"use client";

import { useState, useRef, useCallback } from "react";
import Link from "next/link";
import GradCAMCanvas from "@/components/GradCAMCanvas";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface TopPrediction {
  class: string;
  confidence: number;
}

interface PredictionResult {
  prediction: string;
  confidence: number;
  top_predictions: TopPrediction[];
  gradcam_heatmap?: number[][];
}

export default function Home() {
  const [image, setImage] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((f: File) => {
    if (!f.type.startsWith("image/")) {
      setError("Por favor selecciona una imagen");
      return;
    }
    setFile(f);
    setResult(null);
    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => setImage(e.target?.result as string);
    reader.readAsDataURL(f);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    },
    [handleFile]
  );

  const handleScan = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_URL}/predict/gradcam`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const data = await res.json().catch(() => null);
        throw new Error(data?.detail || `Error del servidor (${res.status})`);
      }

      const data: PredictionResult = await res.json();
      setResult(data);
    } catch (err) {
      const message =
        err instanceof TypeError
          ? "No se pudo conectar con el servidor. Verifica que el backend esté corriendo."
          : (err as Error).message;
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setImage(null);
    setFile(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="container">
      <header className="header">
        <h1>
          Avo<span>Scan</span>
        </h1>
        <p>Detección de fitopatologías en hojas de palta</p>
      </header>

      {!image ? (
        <>
          <button
            className="btn-camera"
            onClick={() => cameraInputRef.current?.click()}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
              <circle cx="12" cy="13" r="4" />
            </svg>
            Tomar foto
          </button>
          <input
            ref={cameraInputRef}
            type="file"
            accept="image/*"
            capture="environment"
            style={{ display: "none" }}
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
          />

          <div
            className={`upload-zone ${dragOver ? "drag-over" : ""}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            <p>
              <span className="btn-text">Seleccionar imagen</span> o arrastrar
              aquí
            </p>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
          />
        </>
      ) : (
        <>
          <div className="preview-container">
            <img src={image} alt="Hoja seleccionada" />
            <button className="btn-remove" onClick={reset}>
              ×
            </button>
          </div>

          <button
            className={`btn-scan ${loading ? "loading" : ""}`}
            onClick={handleScan}
            disabled={loading}
          >
            {loading ? "Analizando..." : "Escanear hoja"}
          </button>
        </>
      )}

      {error && <div className="error-msg">{error}</div>}

      {result && (
        <div className="results">
          <div className="result-card">
            <div className="result-main">
              <span className="result-class">
                {result.prediction.replace(/_/g, " ")}
              </span>
              <span className="result-confidence">{result.confidence}%</span>
            </div>

            {result.top_predictions.map((pred) => (
              <div key={pred.class} className="prob-bar">
                <div className="prob-bar-header">
                  <span>{pred.class.replace(/_/g, " ")}</span>
                  <span>{pred.confidence}%</span>
                </div>
                <div className="prob-bar-track">
                  <div
                    className="prob-bar-fill"
                    style={{ width: `${pred.confidence}%` }}
                  />
                </div>
              </div>
            ))}
          </div>

          {result.gradcam_heatmap && image && (
            <div className="result-card gradcam-section">
              <h3>Mapa de atención (Grad-CAM)</h3>
              <GradCAMCanvas
                imageSrc={image}
                heatmap={result.gradcam_heatmap}
              />
            </div>
          )}
        </div>
      )}

      <Link href="/monitoring" className="btn-monitoring">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M18 20V10M12 20V4M6 20v-6" />
        </svg>
        Monitoreo del Modelo
      </Link>

      <footer className="footer">
        AvoScan v1.0 — UPAO Grupo 02
      </footer>
    </div>
  );
}
