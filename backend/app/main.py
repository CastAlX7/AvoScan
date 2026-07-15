from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import numpy as np
import tensorflow as tf
import json
import io
import os
from PIL import Image

IMG_SIZE = (224, 224)
MODEL_PATH = "model/modelo_palta.keras"
LABELS_PATH = "model/labels.json"
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "monitoring", "reports")

model = None
class_names = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, class_names
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        class_names = json.load(f)
    print(f"Modelo cargado: {len(class_names)} clases")
    yield


app = FastAPI(title="AvoScan API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def generate_gradcam(img_array: np.ndarray, pred_index: int) -> list:
    last_conv_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_name = layer.name
            break

    if last_conv_name is None:
        return []

    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], IMG_SIZE)
    heatmap_resized = tf.squeeze(heatmap_resized).numpy()

    return heatmap_resized.tolist()


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)

    predictions = model.predict(img_array, verbose=0)
    pred_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][pred_index])

    top_predictions = []
    sorted_indices = np.argsort(predictions[0])[::-1]
    for idx in sorted_indices[:5]:
        top_predictions.append(
            {"class": class_names[idx], "confidence": round(float(predictions[0][idx]) * 100, 2)}
        )

    return {
        "prediction": class_names[pred_index],
        "confidence": round(confidence * 100, 2),
        "top_predictions": top_predictions,
    }


@app.post("/predict/gradcam")
async def predict_gradcam(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)

    predictions = model.predict(img_array, verbose=0)
    pred_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][pred_index])

    heatmap = generate_gradcam(img_array, pred_index)

    top_predictions = []
    sorted_indices = np.argsort(predictions[0])[::-1]
    for idx in sorted_indices[:5]:
        top_predictions.append(
            {"class": class_names[idx], "confidence": round(float(predictions[0][idx]) * 100, 2)}
        )

    return {
        "prediction": class_names[pred_index],
        "confidence": round(confidence * 100, 2),
        "top_predictions": top_predictions,
        "gradcam_heatmap": heatmap,
    }


REPORT_METADATA = {
    "01_data_drift.html": {"name": "Data Drift", "description": "Drift en brillo, contraste, canales RGB, saturación y nitidez"},
    "02_target_drift.html": {"name": "Target Drift", "description": "Cambios en la distribución de predicciones"},
    "03_data_quality.html": {"name": "Data Quality", "description": "Valores faltantes, correlaciones y consistencia"},
    "04_classification.html": {"name": "Classification", "description": "Precisión, recall, F1-score por clase"},
    "05_model_confidence.html": {"name": "Model Confidence", "description": "Drift en confianza, margen y entropía"},
}


@app.get("/monitoring/reports")
async def list_reports():
    reports = []
    for filename, meta in REPORT_METADATA.items():
        path = os.path.join(REPORTS_DIR, filename)
        reports.append({
            "filename": filename,
            "name": meta["name"],
            "description": meta["description"],
            "available": os.path.isfile(path),
        })
    return {"reports": reports}


@app.get("/monitoring/reports/{filename}")
async def get_report(filename: str):
    if filename not in REPORT_METADATA:
        raise HTTPException(status_code=404, detail="Reporte no encontrado")
    path = os.path.join(REPORTS_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="El reporte aún no ha sido generado. Ejecuta generate_reports.py primero.")
    return FileResponse(path, media_type="text/html")
