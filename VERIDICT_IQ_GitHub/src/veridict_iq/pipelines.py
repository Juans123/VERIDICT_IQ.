from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_CATEGORICAL = [
    "materia",
    "tipo_proceso",
    "rol_cliente",
    "instancia_en_t0",
    "jurisdiccion",
    "tipo_contraparte",
]
BASE_NUMERIC = [
    "cuantia_inicial_usd",
    "medida_cautelar_solicitada",
    "prueba_pericial_prevista",
]
ADJUSTED_NUMERIC = [
    "log_cuantia",
    "medida_cautelar_solicitada",
    "prueba_pericial_prevista",
    "cuantia_missing",
    "is_actor",
    "is_laboral",
    "is_segunda",
    "is_publica",
    "is_ruminahui",
    "actor_pericial",
    "laboral_pericial",
    "actor_medida",
]


def _preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, numeric_features),
            ("categorical", categorical, BASE_CATEGORICAL),
        ]
    )


def build_baseline(C: float = 1.0) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", _preprocessor(BASE_NUMERIC)),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    l1_ratio=0.0,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def build_adjusted() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", _preprocessor(ADJUSTED_NUMERIC)),
            (
                "classifier",
                LogisticRegression(
                    solver="liblinear",
                    l1_ratio=0.0,
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def adjusted_param_grid() -> dict[str, list[object]]:
    return {
        "classifier__C": [0.05, 0.2, 1.0],
        "classifier__l1_ratio": [0.0, 1.0],
        "classifier__class_weight": [None, "balanced"],
    }
