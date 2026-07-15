"""
AvoScan — Monitoreo con Evidently AI

Genera 5 reportes comparando datos de referencia (test original)
contra datos de producción (nuevas imágenes procesadas por el modelo):
  1. Data Drift         — drift en características de imagen
  2. Target Drift       — drift en distribución de predicciones
  3. Data Quality       — calidad y consistencia de los datos
  4. Classification     — métricas de clasificación
  5. Data Integrity     — resumen completo de integridad

Uso completo (con modelo y dataset):
    python generate_reports.py --model ../backend/model/modelo_palta.keras \
                               --dataset ../../Dataset_Palta_Maestro \
                               --output ./reports

Uso ligero (con CSV pre-calculado, sin TensorFlow):
    python generate_reports.py --ref-csv ./data/reference.csv --output ./reports
"""

import argparse
import os
import numpy as np
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    ClassificationPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    DatasetCorrelationsMetric,
    ColumnDistributionMetric,
    ColumnDriftMetric,
)


IMG_SIZE = (224, 224)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_features(img_path: str) -> dict:
    from PIL import Image
    img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
    h, w = img.shape[:2]
    gray = np.mean(img, axis=2)
    return {
        "brillo": round(float(img.mean() / 255), 4),
        "contraste": round(float(img.std() / 255), 4),
        "canal_R": round(float(img[:, :, 0].mean() / 255), 4),
        "canal_G": round(float(img[:, :, 1].mean() / 255), 4),
        "canal_B": round(float(img[:, :, 2].mean() / 255), 4),
        "saturacion": round(float((img.max(axis=2) - img.min(axis=2)).mean() / 255), 4),
        "nitidez": round(float(np.abs(np.diff(gray, axis=1)).mean() / 255), 4),
        "aspect_ratio": round(w / h, 4),
        "resolucion": h * w,
    }


def predict_image(model, img_path: str, class_names: list) -> tuple:
    from PIL import Image
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    pred = model.predict(np.expand_dims(img_array, 0), verbose=0)
    pred_idx = int(np.argmax(pred[0]))
    top2 = np.sort(pred[0])[::-1]
    margin = float(top2[0] - top2[1]) if len(top2) > 1 else float(top2[0])
    entropy = float(-np.sum(pred[0] * np.log(pred[0] + 1e-10)))
    return class_names[pred_idx], float(pred[0][pred_idx]), margin, entropy


def build_dataset(model, dataset_path: str, split: str, class_names: list) -> pd.DataFrame:
    img_dir = os.path.join(dataset_path, split, "images")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"No se encontró: {img_dir}")

    records = []
    for fname in sorted(os.listdir(img_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMG_EXTS:
            continue
        img_path = os.path.join(img_dir, fname)
        features = extract_features(img_path)
        prediction, confidence, margin, entropy = predict_image(model, img_path, class_names)
        features["prediccion"] = prediction
        features["confianza"] = round(confidence, 4)
        features["margen"] = round(margin, 4)
        features["entropia"] = round(entropy, 4)
        features["archivo"] = fname
        records.append(features)

    return pd.DataFrame(records)


def simulate_drift(df_ref: pd.DataFrame) -> pd.DataFrame:
    df_prod = df_ref.copy()
    np.random.seed(42)
    df_prod["brillo"] = df_prod["brillo"] * 0.7 + 0.1
    df_prod["contraste"] = df_prod["contraste"] * 1.3
    df_prod["canal_R"] = np.clip(df_prod["canal_R"] + np.random.normal(0, 0.05, len(df_prod)), 0, 1)
    df_prod["canal_G"] = np.clip(df_prod["canal_G"] - 0.05, 0, 1)
    df_prod["canal_B"] = np.clip(df_prod["canal_B"] + np.random.normal(0, 0.08, len(df_prod)), 0, 1)
    df_prod["saturacion"] = np.clip(df_prod["saturacion"] * 0.8, 0, 1)
    df_prod["nitidez"] = np.clip(df_prod["nitidez"] * 0.6, 0, 1)
    df_prod["confianza"] = np.clip(df_prod["confianza"] * 0.6, 0, 1)
    df_prod["margen"] = np.clip(df_prod["margen"] * 0.5, 0, 1)
    df_prod["entropia"] = df_prod["entropia"] * 1.4
    return df_prod


def generate_reports(df_ref: pd.DataFrame, df_prod: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    cols_features = [
        "brillo", "contraste", "canal_R", "canal_G", "canal_B",
        "saturacion", "nitidez", "confianza", "margen", "entropia",
    ]

    print("\n[1/5] Generando reporte de Data Drift...")
    data_drift_report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric(),
    ])
    data_drift_report.run(
        reference_data=df_ref[cols_features],
        current_data=df_prod[cols_features],
    )
    path = os.path.join(output_dir, "01_data_drift.html")
    data_drift_report.save_html(path)
    print(f"  Guardado: {path}")

    print("[2/5] Generando reporte de Target Drift...")
    df_ref_target = df_ref[["prediccion"]].rename(columns={"prediccion": "target"})
    df_prod_target = df_prod[["prediccion"]].rename(columns={"prediccion": "target"})
    target_drift_report = Report(metrics=[TargetDriftPreset()])
    target_drift_report.run(
        reference_data=df_ref_target,
        current_data=df_prod_target,
    )
    path = os.path.join(output_dir, "02_target_drift.html")
    target_drift_report.save_html(path)
    print(f"  Guardado: {path}")

    print("[3/5] Generando reporte de Data Quality...")
    data_quality_report = Report(metrics=[
        DataQualityPreset(),
        DatasetMissingValuesMetric(),
        DatasetCorrelationsMetric(),
    ])
    data_quality_report.run(
        reference_data=df_ref[cols_features],
        current_data=df_prod[cols_features],
    )
    path = os.path.join(output_dir, "03_data_quality.html")
    data_quality_report.save_html(path)
    print(f"  Guardado: {path}")

    print("[4/5] Generando reporte de Clasificación...")
    df_ref_cls = df_ref.rename(columns={"prediccion": "prediction"})
    df_prod_cls = df_prod.rename(columns={"prediccion": "prediction"})
    df_ref_cls["target"] = df_ref_cls["prediction"]
    df_prod_cls["target"] = df_prod_cls["prediction"]
    cls_report = Report(metrics=[ClassificationPreset()])
    cls_report.run(
        reference_data=df_ref_cls[["target", "prediction"]],
        current_data=df_prod_cls[["target", "prediction"]],
    )
    path = os.path.join(output_dir, "04_classification.html")
    cls_report.save_html(path)
    print(f"  Guardado: {path}")

    print("[5/5] Generando reporte de Confianza del Modelo...")
    confidence_report = Report(metrics=[
        ColumnDriftMetric(column_name="confianza"),
        ColumnDistributionMetric(column_name="confianza"),
        ColumnDriftMetric(column_name="margen"),
        ColumnDistributionMetric(column_name="margen"),
        ColumnDriftMetric(column_name="entropia"),
        ColumnDistributionMetric(column_name="entropia"),
    ])
    confidence_report.run(
        reference_data=df_ref[["confianza", "margen", "entropia"]],
        current_data=df_prod[["confianza", "margen", "entropia"]],
    )
    path = os.path.join(output_dir, "05_model_confidence.html")
    confidence_report.save_html(path)
    print(f"  Guardado: {path}")

    print(f"\n{'='*50}")
    print(f"5 reportes generados en: {os.path.abspath(output_dir)}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="AvoScan — Monitoreo con Evidently AI")
    parser.add_argument("--model", default="../backend/model/modelo_palta.keras")
    parser.add_argument("--labels", default="../backend/model/labels.json")
    parser.add_argument("--dataset", default="../../Dataset_Palta_Maestro")
    parser.add_argument("--output", default="./reports")
    parser.add_argument("--ref-csv", default=None,
                        help="CSV con datos de referencia pre-calculados (no requiere TF ni dataset)")
    parser.add_argument("--save-csv", default=None,
                        help="Guardar datos de referencia como CSV para uso futuro")
    parser.add_argument("--prod-dir", default=None,
                        help="Carpeta con imágenes de producción. Si no se indica, simula drift.")
    args = parser.parse_args()

    if args.ref_csv and os.path.isfile(args.ref_csv):
        print(f"Cargando datos de referencia desde CSV: {args.ref_csv}")
        df_ref = pd.read_csv(args.ref_csv)
        print(f"  Referencia: {len(df_ref)} registros")
    else:
        import json
        import tensorflow as tf

        print("Cargando modelo...")
        model = tf.keras.models.load_model(args.model)
        with open(args.labels, "r") as f:
            class_names = json.load(f)
        print(f"Modelo cargado: {len(class_names)} clases")

        print("Construyendo dataset de referencia (test)...")
        df_ref = build_dataset(model, args.dataset, "test", class_names)
        print(f"  Referencia: {len(df_ref)} imágenes")

        if args.save_csv:
            os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
            df_ref.to_csv(args.save_csv, index=False)
            print(f"  CSV guardado: {args.save_csv}")

    if args.prod_dir and os.path.isdir(args.prod_dir):
        import json
        import tensorflow as tf
        from PIL import Image

        print(f"Construyendo dataset de producción desde {args.prod_dir}...")
        model_path = args.model
        labels_path = args.labels
        model = tf.keras.models.load_model(model_path)
        with open(labels_path, "r") as f:
            class_names = json.load(f)

        records = []
        for fname in sorted(os.listdir(args.prod_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in IMG_EXTS:
                continue
            img_path = os.path.join(args.prod_dir, fname)
            features = extract_features(img_path)
            prediction, confidence, margin, entropy = predict_image(model, img_path, class_names)
            features["prediccion"] = prediction
            features["confianza"] = round(confidence, 4)
            features["margen"] = round(margin, 4)
            features["entropia"] = round(entropy, 4)
            features["archivo"] = fname
            records.append(features)
        df_prod = pd.DataFrame(records)
    else:
        print("Simulando drift en datos de producción...")
        df_prod = simulate_drift(df_ref)

    print(f"  Producción: {len(df_prod)} registros")

    generate_reports(df_ref, df_prod, args.output)


if __name__ == "__main__":
    main()
