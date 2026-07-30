from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

REQUIRED_COLUMNS = {
    "case_id",
    "fecha_cierre",
    "materia",
    "tipo_proceso",
    "cuantia_inicial_usd",
    "rol_cliente",
    "instancia_en_t0",
    "jurisdiccion",
    "tipo_contraparte",
    "medida_cautelar_solicitada",
    "prueba_pericial_prevista",
    "resultado_favorable",
}


@dataclass(frozen=True)
class ValidationSummary:
    rows: int
    columns: int
    duplicated_case_ids: int
    missing_percentage: float
    positive_rate: float


def validate_dataset(df: pd.DataFrame) -> ValidationSummary:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("The dataset is empty.")
    if not set(df["resultado_favorable"].dropna().unique()).issubset({0, 1}):
        raise ValueError("resultado_favorable must contain only 0/1 values.")
    parsed_dates = pd.to_datetime(df["fecha_cierre"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("fecha_cierre contains invalid dates.")
    duplicated = int(df["case_id"].duplicated().sum())
    if duplicated:
        raise ValueError(f"Duplicate case_id values detected: {duplicated}")
    return ValidationSummary(
        rows=len(df),
        columns=df.shape[1],
        duplicated_case_ids=duplicated,
        missing_percentage=float(df.isna().sum().sum() / df.size * 100),
        positive_rate=float(df["resultado_favorable"].mean()),
    )
