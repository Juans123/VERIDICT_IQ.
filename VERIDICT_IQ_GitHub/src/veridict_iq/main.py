from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.build_features import add_adjusted_features

MODEL_PATH = Path(os.getenv("VERIDICT_MODEL_PATH", "models/adjusted_b2.joblib"))
app = FastAPI(title="VERIDICT IQ adjusted prototype", version="0.2.0")
_model = None


class CaseInput(BaseModel):
    case_id: str = Field(default="INFERENCE")
    fecha_cierre: str = Field(default="2026-01-01")
    materia: str
    tipo_proceso: str
    cuantia_inicial_usd: float | None = None
    rol_cliente: str
    instancia_en_t0: str
    jurisdiccion: str | None = None
    tipo_contraparte: str | None = None
    medida_cautelar_solicitada: float | None = None
    prueba_pericial_prevista: float | None = None


@app.on_event("startup")
def load_model() -> None:
    global _model
    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model_loaded": _model is not None, "model_path": str(MODEL_PATH)}


@app.post("/predict")
def predict(case: CaseInput) -> dict[str, object]:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model artifact not loaded.")
    frame = pd.DataFrame([case.model_dump()])
    adjusted = add_adjusted_features(frame)
    probability = float(_model.predict_proba(adjusted)[0, 1])
    return {
        "probabilidad_favorable": probability,
        "advertencia": "Salida experimental de apoyo; no constituye asesoría ni decisión jurídica.",
        "model_version": "B2-adjusted-0.2.0",
    }
