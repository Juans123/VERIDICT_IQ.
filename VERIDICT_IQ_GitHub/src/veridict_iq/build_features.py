from __future__ import annotations

import numpy as np
import pandas as pd


def add_adjusted_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add only features available at t0 and computed row-by-row.

    No target statistics or information from the temporal test set are used.
    """
    out = df.copy()
    out["log_cuantia"] = np.log1p(out["cuantia_inicial_usd"])
    out["cuantia_missing"] = out["cuantia_inicial_usd"].isna().astype(float)
    out["is_actor"] = (out["rol_cliente"] == "Actor").astype(float)
    out["is_laboral"] = (out["materia"] == "Laboral").astype(float)
    out["is_segunda"] = (out["instancia_en_t0"] == "Segunda").astype(float)
    out["is_publica"] = (out["tipo_contraparte"] == "Entidad pública").astype(float)
    out["is_ruminahui"] = (out["jurisdiccion"] == "Rumiñahui").astype(float)
    pericial = out["prueba_pericial_prevista"].fillna(0)
    medida = out["medida_cautelar_solicitada"].fillna(0)
    out["actor_pericial"] = out["is_actor"] * pericial
    out["laboral_pericial"] = out["is_laboral"] * pericial
    out["actor_medida"] = out["is_actor"] * medida
    return out
