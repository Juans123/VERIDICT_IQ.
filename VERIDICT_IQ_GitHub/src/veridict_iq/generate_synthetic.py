from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42


def generate_synthetic_legal_data(n: int = 360, seed: int = SEED) -> pd.DataFrame:
    """Generate non-identifying legal-like data for engineering validation only.

    The simulated relationships must never be interpreted as evidence about GRUND
    Abogados LLP or the Ecuadorian justice system.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-15", "2026-06-30", periods=n)

    materia = rng.choice(["Civil", "Laboral"], n, p=[0.60, 0.40])
    process_options = {
        "Civil": ["Ordinario", "Ejecutivo", "Sumario", "Monitorio"],
        "Laboral": ["Despido intempestivo", "Visto bueno", "Haberes laborales"],
    }
    tipo_proceso = [rng.choice(process_options[m]) for m in materia]
    cuantia = np.exp(rng.normal(np.log(18000), 1.0, n)).clip(800, 500000)
    rol_cliente = rng.choice(["Actor", "Demandado"], n, p=[0.62, 0.38])
    instancia = rng.choice(["Primera", "Segunda"], n, p=[0.87, 0.13])
    jurisdiccion = rng.choice(
        ["Quito Norte", "Quito Sur", "Rumiñahui", "Mejía"],
        n,
        p=[0.38, 0.32, 0.20, 0.10],
    )
    contraparte = rng.choice(
        ["Persona natural", "Persona jurídica", "Entidad pública"],
        n,
        p=[0.46, 0.44, 0.10],
    )
    medida = rng.binomial(1, 0.27, n)
    pericial = rng.binomial(1, 0.35, n)

    log_cuantia = np.log1p(cuantia)
    z_cuantia = (log_cuantia - log_cuantia.mean()) / log_cuantia.std()
    score = (
        -1.80
        + 2.40
        * (
            0.70 * (rol_cliente == "Actor")
            + 0.50 * (materia == "Laboral")
            - 0.40 * (instancia == "Segunda")
            + 0.55 * pericial
            + 0.35 * medida
            - 0.30 * (contraparte == "Entidad pública")
            + 0.30 * (jurisdiccion == "Rumiñahui")
            + 0.25 * z_cuantia
        )
        - 0.10 * np.linspace(0, 1, n)
        + rng.normal(0, 0.15, n)
    )
    probability = 1 / (1 + np.exp(-score))
    outcome = rng.binomial(1, probability)

    df = pd.DataFrame(
        {
            "case_id": [f"SYN-{i + 1:04d}" for i in range(n)],
            "fecha_cierre": dates,
            "materia": materia,
            "tipo_proceso": tipo_proceso,
            "cuantia_inicial_usd": cuantia.round(2),
            "rol_cliente": rol_cliente,
            "instancia_en_t0": instancia,
            "jurisdiccion": jurisdiccion,
            "tipo_contraparte": contraparte,
            "medida_cautelar_solicitada": medida.astype(float),
            "prueba_pericial_prevista": pericial.astype(float),
            "resultado_favorable": outcome,
        }
    )

    missingness = {
        "cuantia_inicial_usd": 0.07,
        "jurisdiccion": 0.04,
        "tipo_contraparte": 0.05,
        "medida_cautelar_solicitada": 0.03,
        "prueba_pericial_prevista": 0.04,
    }
    for column, fraction in missingness.items():
        indices = rng.choice(n, size=max(1, int(n * fraction)), replace=False)
        df.loc[indices, column] = np.nan
    return df


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_synthetic_snapshot(path: Path, n: int = 360, seed: int = SEED) -> tuple[pd.DataFrame, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_legal_data(n=n, seed=seed)
    df.to_csv(path, index=False)
    return df, sha256_file(path)
