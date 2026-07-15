"use client";

import { useState, useEffect } from "react";
import Link from "next/link";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Report {
  filename: string;
  name: string;
  description: string;
  available: boolean;
}

const REPORT_ICONS: Record<string, string> = {
  "01_data_drift.html": "M2 12h4l3-9 4 18 3-9h4",
  "02_target_drift.html": "M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5",
  "03_data_quality.html": "M9 11l3 3L22 4M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11",
  "04_classification.html": "M18 20V10M12 20V4M6 20v-6",
  "05_model_confidence.html": "M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z",
};

export default function MonitoringPage() {
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_URL}/monitoring/reports`)
      .then((res) => res.json())
      .then((data) => setReports(data.reports))
      .catch(() => setError("No se pudo conectar con el servidor"))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="container">
      <header className="header">
        <h1>
          Avo<span>Scan</span>
        </h1>
        <p>Monitoreo del Modelo — Evidently AI</p>
      </header>

      <Link href="/" className="btn-back">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M19 12H5M12 19l-7-7 7-7" />
        </svg>
        Volver al escáner
      </Link>

      {loading && <p className="monitoring-status">Cargando reportes...</p>}
      {error && <div className="error-msg">{error}</div>}

      <div className="reports-grid">
        {reports.map((report) => (
          <a
            key={report.filename}
            href={report.available ? `${API_URL}/monitoring/reports/${report.filename}` : undefined}
            target="_blank"
            rel="noopener noreferrer"
            className={`report-card ${!report.available ? "report-disabled" : ""}`}
            onClick={(e) => {
              if (!report.available) e.preventDefault();
            }}
          >
            <div className="report-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d={REPORT_ICONS[report.filename] || "M12 2v20M2 12h20"} />
              </svg>
            </div>
            <div className="report-info">
              <h3>{report.name}</h3>
              <p>{report.description}</p>
            </div>
            {report.available ? (
              <span className="report-badge available">Disponible</span>
            ) : (
              <span className="report-badge pending">Pendiente</span>
            )}
          </a>
        ))}
      </div>

      {reports.length > 0 && !reports.some((r) => r.available) && (
        <div className="monitoring-hint">
          <p>Los reportes aún no han sido generados.</p>
          <code>python generate_reports.py</code>
        </div>
      )}

      <footer className="footer">AvoScan v1.0 — UPAO Grupo 02</footer>
    </div>
  );
}
