from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from veridict_iq.pipeline import generate_synthetic_legal_data  # noqa: E402

EXPECTED_COLUMNS = {
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


def test_versioned_dataset_contract() -> None:
    path = REPO_ROOT / "data" / "synthetic" / "dataset_sintetico_veridict_iq.csv"
    df = pd.read_csv(path)
    assert set(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 360
    assert set(df["resultado_favorable"].dropna().unique()).issubset({0, 1})
    assert df["case_id"].is_unique


def test_generator_is_deterministic() -> None:
    first = generate_synthetic_legal_data(n=40, seed=42)
    second = generate_synthetic_legal_data(n=40, seed=42)
    pd.testing.assert_frame_equal(first, second)
