from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(os.environ.get('VERIDICT_OUTPUT_DIR', str(Path.cwd() / 'artifacts' / 'latest'))).resolve()
DATA_DIR = ROOT / 'data'
FIG_DIR = ROOT / 'reports' / 'figures'
METRIC_DIR = ROOT / 'reports' / 'metrics'
MODEL_DIR = ROOT / 'models'
DOC_DIR = ROOT / 'docs'
MODEL_CARD_DIR = ROOT / 'reports' / 'model_card'
for p in [DATA_DIR, FIG_DIR, METRIC_DIR, MODEL_DIR, DOC_DIR, MODEL_CARD_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def generate_synthetic_legal_data(n: int = 360, seed: int = SEED) -> pd.DataFrame:
    """Generate a synthetic, non-identifying dataset for engineering validation only."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2020-01-15', '2026-06-30', periods=n)

    materia = rng.choice(['Civil', 'Laboral'], n, p=[0.60, 0.40])
    tipo_options = {
        'Civil': ['Ordinario', 'Ejecutivo', 'Sumario', 'Monitorio'],
        'Laboral': ['Despido intempestivo', 'Visto bueno', 'Haberes laborales'],
    }
    tipo_proceso = [rng.choice(tipo_options[m]) for m in materia]
    cuantia = np.exp(rng.normal(np.log(18000), 1.0, n)).clip(800, 500000)
    rol_cliente = rng.choice(['Actor', 'Demandado'], n, p=[0.62, 0.38])
    instancia = rng.choice(['Primera', 'Segunda'], n, p=[0.87, 0.13])
    jurisdiccion = rng.choice(['Quito Norte', 'Quito Sur', 'Rumiñahui', 'Mejía'], n, p=[0.38, 0.32, 0.20, 0.10])
    contraparte = rng.choice(['Persona natural', 'Persona jurídica', 'Entidad pública'], n, p=[0.46, 0.44, 0.10])
    medida = rng.binomial(1, 0.27, n)
    pericial = rng.binomial(1, 0.35, n)

    # Moderate, intentionally imperfect signal. The temporal component introduces mild drift.
    log_cuantia = np.log1p(cuantia)
    z_cuantia = (log_cuantia - log_cuantia.mean()) / log_cuantia.std()
    score = (
        -1.80
        + 2.40 * (
            0.70 * (rol_cliente == 'Actor')
            + 0.50 * (materia == 'Laboral')
            - 0.40 * (instancia == 'Segunda')
            + 0.55 * pericial
            + 0.35 * medida
            - 0.30 * (contraparte == 'Entidad pública')
            + 0.30 * (jurisdiccion == 'Rumiñahui')
            + 0.25 * z_cuantia
        )
        - 0.10 * np.linspace(0, 1, n)
        + rng.normal(0, 0.15, n)
    )
    prob = 1 / (1 + np.exp(-score))
    resultado = rng.binomial(1, prob)

    df = pd.DataFrame({
        'case_id': [f'SYN-{i+1:04d}' for i in range(n)],
        'fecha_cierre': dates,
        'materia': materia,
        'tipo_proceso': tipo_proceso,
        'cuantia_inicial_usd': cuantia.round(2),
        'rol_cliente': rol_cliente,
        'instancia_en_t0': instancia,
        'jurisdiccion': jurisdiccion,
        'tipo_contraparte': contraparte,
        'medida_cautelar_solicitada': medida.astype(float),
        'prueba_pericial_prevista': pericial.astype(float),
        'resultado_favorable': resultado,
    })

    # Add realistic missingness, avoiding target/date/ID.
    for col, frac in {
        'cuantia_inicial_usd': 0.07,
        'jurisdiccion': 0.04,
        'tipo_contraparte': 0.05,
        'medida_cautelar_solicitada': 0.03,
        'prueba_pericial_prevista': 0.04,
    }.items():
        idx = rng.choice(n, size=max(1, int(n * frac)), replace=False)
        df.loc[idx, col] = np.nan
    return df


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def build_preprocessor(df: pd.DataFrame):
    numeric_features = ['cuantia_inicial_usd', 'medida_cautelar_solicitada', 'prueba_pericial_prevista']
    categorical_features = ['materia', 'tipo_proceso', 'rol_cliente', 'instancia_en_t0', 'jurisdiccion', 'tipo_contraparte']

    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scaler', StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', min_frequency=3)),
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_pipe, numeric_features),
        ('cat', categorical_pipe, categorical_features),
    ])
    return preprocessor, numeric_features, categorical_features


def get_score_dict():
    return {
        'roc_auc': 'roc_auc',
        'pr_auc': 'average_precision',
        'f1_macro': 'f1_macro',
        'balanced_accuracy': 'balanced_accuracy',
        'neg_brier': 'neg_brier_score',
        'neg_log_loss': 'neg_log_loss',
    }


def cv_summary(model, X, y, cv) -> dict:
    res = cross_validate(model, X, y, cv=cv, scoring=get_score_dict(), n_jobs=1, return_train_score=False)
    out = {}
    for key, values in res.items():
        if not key.startswith('test_'):
            continue
        metric = key.replace('test_', '')
        vals = np.asarray(values, dtype=float)
        if metric in {'neg_brier', 'neg_log_loss'}:
            vals = -vals
            metric = metric.replace('neg_', '')
        out[metric] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals, ddof=1)),
            'n': int(len(vals)),
        }
    return out


def choose_threshold(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, pd.DataFrame]:
    thresholds = np.linspace(0.20, 0.80, 121)
    rows = []
    for t in thresholds:
        pred = (prob >= t).astype(int)
        rows.append({
            'threshold': float(t),
            'f1_macro': float(f1_score(y_true, pred, average='macro')),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, pred)),
            'precision_positive': float(precision_score(y_true, pred, zero_division=0)),
            'recall_positive': float(recall_score(y_true, pred, zero_division=0)),
        })
    table = pd.DataFrame(rows)
    best = table.sort_values(['f1_macro', 'balanced_accuracy'], ascending=False).iloc[0]
    return float(best['threshold']), table


def metric_values(y_true, prob, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    return {
        'roc_auc': float(roc_auc_score(y_true, prob)),
        'pr_auc': float(average_precision_score(y_true, prob)),
        'f1_macro': float(f1_score(y_true, pred, average='macro')),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, pred)),
        'precision_0': float(precision_score(y_true, pred, pos_label=0, zero_division=0)),
        'recall_0': float(recall_score(y_true, pred, pos_label=0, zero_division=0)),
        'precision_1': float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        'recall_1': float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        'brier': float(brier_score_loss(y_true, prob)),
        'log_loss': float(log_loss(y_true, np.column_stack([1-prob, prob]), labels=[0,1])),
        'threshold': float(threshold),
        'n': int(len(y_true)),
        'positive_rate': float(np.mean(y_true)),
    }


def bootstrap_ci(y_true, prob, threshold, metric_name, n_boot=500, seed=SEED):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y_true[idx]
        pb = prob[idx]
        if len(np.unique(yb)) < 2 and metric_name in {'roc_auc', 'pr_auc'}:
            continue
        pred = (pb >= threshold).astype(int)
        try:
            if metric_name == 'roc_auc':
                val = roc_auc_score(yb, pb)
            elif metric_name == 'pr_auc':
                val = average_precision_score(yb, pb)
            elif metric_name == 'f1_macro':
                val = f1_score(yb, pred, average='macro')
            elif metric_name == 'balanced_accuracy':
                val = balanced_accuracy_score(yb, pred)
            elif metric_name == 'brier':
                val = brier_score_loss(yb, pb)
            else:
                raise ValueError(metric_name)
            vals.append(val)
        except ValueError:
            continue
    if not vals:
        return [None, None]
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def plot_class_distribution(df):
    counts = df['resultado_favorable'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    labels = ['No favorable (0)', 'Favorable (1)']
    ax.bar(labels, [counts.get(0,0), counts.get(1,0)])
    ax.set_ylabel('Número de causas')
    ax.set_title('Distribución de la variable objetivo (datos sintéticos)')
    for i, v in enumerate([counts.get(0,0), counts.get(1,0)]):
        ax.text(i, v + 3, str(v), ha='center')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'figura_1_distribucion_clases.png', dpi=200)
    plt.close(fig)


def plot_roc_pr(y, probs: dict):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
    ax.plot([0,1],[0,1], linestyle='--', label='Azar')
    ax.set_xlabel('Tasa de falsos positivos')
    ax.set_ylabel('Tasa de verdaderos positivos')
    ax.set_title('Curva ROC en test temporal')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'figura_2_curva_roc.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, p in probs.items():
        precision, recall, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax.plot(recall, precision, label=f'{name} (PR-AUC={ap:.3f})')
    ax.axhline(np.mean(y), linestyle='--', label=f'Prevalencia={np.mean(y):.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precisión')
    ax.set_title('Curva precisión-recall en test temporal')
    ax.legend(loc='lower left')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'figura_3_curva_pr.png', dpi=200)
    plt.close(fig)


def plot_confusion(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    im = ax.imshow(cm)
    ax.set_xticks([0,1], ['No favorable', 'Favorable'])
    ax.set_yticks([0,1], ['No favorable', 'Favorable'])
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Valor real')
    ax.set_title(f'Matriz de confusión B1 (umbral={threshold:.2f})')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i,j], ha='center', va='center')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'figura_4_matriz_confusion.png', dpi=200)
    plt.close(fig)


def plot_calibration(y, probs: dict):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot([0,1],[0,1], linestyle='--', label='Calibración perfecta')
    for name, p in probs.items():
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=6, strategy='quantile')
        ax.plot(mean_pred, frac_pos, marker='o', label=name)
    ax.set_xlabel('Probabilidad predicha media')
    ax.set_ylabel('Frecuencia observada')
    ax.set_title('Curva de calibración en test temporal')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'figura_5_calibracion.png', dpi=200)
    plt.close(fig)


def plot_coefficients(fitted_pipeline):
    pre = fitted_pipeline.named_steps['preprocess']
    model = fitted_pipeline.named_steps['model']
    feature_names = pre.get_feature_names_out()
    coefs = model.coef_[0]
    coef_df = pd.DataFrame({'feature': feature_names, 'coef': coefs, 'odds_ratio': np.exp(coefs)})
    coef_df['abs_coef'] = coef_df['coef'].abs()
    coef_df = coef_df.sort_values('abs_coef', ascending=False)
    coef_df.to_csv(METRIC_DIR / 'coeficientes_odds_ratios.csv', index=False)
    top = coef_df.head(12).sort_values('coef')
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.barh(top['feature'].str.replace('num__','', regex=False).str.replace('cat__','', regex=False), top['coef'])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel('Coeficiente logístico estandarizado')
    ax.set_title('Variables con mayor magnitud de coeficiente en B1')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'figura_6_coeficientes.png', dpi=200)
    plt.close(fig)
    return coef_df


def load_input_dataset(input_file: str | None) -> tuple[pd.DataFrame, str]:
    if not input_file:
        return generate_synthetic_legal_data(), 'synthetic_demo'
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f'No se encontró el archivo: {path}')
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
    elif path.suffix.lower() in {'.xlsx', '.xls'}:
        df = pd.read_excel(path)
    else:
        raise ValueError('Formato no soportado. Use CSV o XLSX.')
    required = {
        'case_id', 'fecha_cierre', 'materia', 'tipo_proceso', 'cuantia_inicial_usd',
        'rol_cliente', 'instancia_en_t0', 'jurisdiccion', 'tipo_contraparte',
        'medida_cautelar_solicitada', 'prueba_pericial_prevista', 'resultado_favorable'
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'Faltan columnas obligatorias: {missing}')
    df['fecha_cierre'] = pd.to_datetime(df['fecha_cierre'], errors='coerce')
    if df['fecha_cierre'].isna().any():
        raise ValueError('fecha_cierre contiene valores no convertibles a fecha.')
    df['resultado_favorable'] = pd.to_numeric(df['resultado_favorable'], errors='coerce')
    if df['resultado_favorable'].isna().any() or not set(df['resultado_favorable'].unique()).issubset({0, 1}):
        raise ValueError('resultado_favorable debe contener únicamente 0 y 1.')
    if df['case_id'].duplicated().any():
        raise ValueError('case_id contiene duplicados.')
    return df, 'real_anonimizado'


def main():
    input_file = os.environ.get('VERIDICT_INPUT_FILE')
    df, data_mode = load_input_dataset(input_file)
    data_path = DATA_DIR / ('snapshot_modelado.csv' if input_file else 'dataset_sintetico_veridict_iq.csv')
    df.to_csv(data_path, index=False)
    data_hash = file_sha256(data_path)

    # Temporal holdout: oldest 80% dev, newest 20% test.
    df = df.sort_values('fecha_cierre').reset_index(drop=True)
    cut = int(len(df) * 0.80)
    dev, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    target = 'resultado_favorable'
    drop_cols = ['case_id', 'fecha_cierre', target]
    X_dev, y_dev = dev.drop(columns=drop_cols), dev[target].to_numpy()
    X_test, y_test = test.drop(columns=drop_cols), test[target].to_numpy()

    preprocessor, numeric_features, categorical_features = build_preprocessor(df)
    b0_mf = Pipeline([('preprocess', clone(preprocessor)), ('model', DummyClassifier(strategy='most_frequent'))])
    b0_st = Pipeline([('preprocess', clone(preprocessor)), ('model', DummyClassifier(strategy='stratified', random_state=SEED))])
    b1 = Pipeline([
        ('preprocess', clone(preprocessor)),
        ('model', LogisticRegression(C=1.0, l1_ratio=0.0, solver='lbfgs', max_iter=1000, random_state=SEED)),
    ])

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    cv_results = {
        'B0 most_frequent': cv_summary(b0_mf, X_dev, y_dev, cv),
        'B0 stratified': cv_summary(b0_st, X_dev, y_dev, cv),
        'B1 regresion_logistica': cv_summary(b1, X_dev, y_dev, cv),
    }
    with (METRIC_DIR / 'resultados_cv.json').open('w', encoding='utf-8') as f:
        json.dump(cv_results, f, indent=2, ensure_ascii=False)

    cv_rows = []
    for model, metrics in cv_results.items():
        row = {'modelo': model}
        for m, s in metrics.items():
            row[f'{m}_mean'] = s['mean']
            row[f'{m}_std'] = s['std']
        cv_rows.append(row)
    pd.DataFrame(cv_rows).to_csv(METRIC_DIR / 'resultados_cv.csv', index=False)

    # OOF threshold selection on development only.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_prob = cross_val_predict(b1, X_dev, y_dev, cv=skf, method='predict_proba', n_jobs=1)[:, 1]
    threshold, threshold_table = choose_threshold(y_dev, oof_prob)
    threshold_table.to_csv(METRIC_DIR / 'seleccion_umbral_oof.csv', index=False)

    fitted_models = {}
    test_probs = {}
    test_metrics = {}
    for name, model in [('B0 most_frequent', b0_mf), ('B0 stratified', b0_st), ('B1 regresion_logistica', b1)]:
        fitted = clone(model).fit(X_dev, y_dev)
        prob = fitted.predict_proba(X_test)[:, 1]
        use_threshold = threshold if name.startswith('B1') else 0.50
        test_metrics[name] = metric_values(y_test, prob, use_threshold)
        fitted_models[name] = fitted
        test_probs[name] = prob

    # Add bootstrap CIs for B1 key metrics.
    b1_metrics = test_metrics['B1 regresion_logistica']
    b1_metrics['ci95'] = {
        m: bootstrap_ci(y_test, test_probs['B1 regresion_logistica'], threshold, m)
        for m in ['roc_auc', 'pr_auc', 'f1_macro', 'balanced_accuracy', 'brier']
    }
    with (METRIC_DIR / 'resultados_test.json').open('w', encoding='utf-8') as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    test_rows = []
    for name, mets in test_metrics.items():
        row = {'modelo': name}
        for k, v in mets.items():
            if k != 'ci95':
                row[k] = v
        test_rows.append(row)
    pd.DataFrame(test_rows).to_csv(METRIC_DIR / 'resultados_test.csv', index=False)

    # Error analysis for B1.
    b1_prob = test_probs['B1 regresion_logistica']
    b1_pred = (b1_prob >= threshold).astype(int)
    err = test[['case_id', 'fecha_cierre', 'materia', 'tipo_proceso', 'rol_cliente', 'jurisdiccion', target]].copy()
    err['probabilidad_favorable'] = b1_prob
    err['prediccion'] = b1_pred
    err['tipo_error'] = np.where((err[target] == 0) & (err['prediccion'] == 1), 'Falso positivo',
                                 np.where((err[target] == 1) & (err['prediccion'] == 0), 'Falso negativo', 'Correcto'))
    err['confianza_error'] = np.where(err['tipo_error'] == 'Falso positivo', err['probabilidad_favorable'],
                                      np.where(err['tipo_error'] == 'Falso negativo', 1-err['probabilidad_favorable'], 0))
    errors_only = err[err['tipo_error'] != 'Correcto'].sort_values('confianza_error', ascending=False)
    errors_only.to_csv(METRIC_DIR / 'analisis_errores.csv', index=False)

    # Figures.
    plot_class_distribution(df)
    plot_roc_pr(y_test, {'B0 most frequent': test_probs['B0 most_frequent'], 'B1 logística': b1_prob})
    plot_confusion(y_test, b1_prob, threshold)
    plot_calibration(y_test, {'B1 logística': b1_prob})
    coef_df = plot_coefficients(fitted_models['B1 regresion_logistica'])

    # Save model and metadata.
    model_path = MODEL_DIR / 'baseline_b1_logistic.joblib'
    joblib.dump(fitted_models['B1 regresion_logistica'], model_path)
    metadata = {
        'data_mode': data_mode,
        'disclaimer': ('Métricas exclusivas de validación técnica; no representan datos ni desempeño de GRUND Abogados LLP®.' if data_mode == 'synthetic_demo' else 'Ejecución con snapshot anonimizado cargado por el equipo; su validez depende de permisos, etiquetado y revisión jurídica.'),
        'seed': SEED,
        'data_sha256': data_hash,
        'n_total': len(df),
        'n_dev': len(dev),
        'n_test': len(test),
        'period_start': str(df['fecha_cierre'].min().date()),
        'period_end': str(df['fecha_cierre'].max().date()),
        'target_positive_rate_total': float(df[target].mean()),
        'target_positive_rate_dev': float(dev[target].mean()),
        'target_positive_rate_test': float(test[target].mean()),
        'missing_percentage': float(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'threshold_oof': threshold,
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'versions': {
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'scikit_learn': sklearn.__version__,
        },
        'model_file_sha256': file_sha256(model_path),
    }
    with (METRIC_DIR / 'run_metadata.json').open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Model card.
    model_card = f"""# Model Card - VERIDICT IQ Baseline B1\n\n## Estado\nValidación técnica preliminar. Revise el campo data_mode antes de interpretar resultados.\n\n## Uso previsto\nVerificar la ejecución reproducible del pipeline de clasificación binaria y preparar la sustitución por el snapshot anonimizado autorizado.\n\n## Modelo\nRegresión logística L2, C=1.0, max_iter=1000, preprocesamiento encapsulado.\n\n## Datos\n{len(df)} registros (modo: {data_mode}); desarrollo={len(dev)} y test temporal={len(test)}. SHA-256: `{data_hash}`.\n\n## Métricas de demostración\nROC-AUC test: {b1_metrics['roc_auc']:.3f}; PR-AUC: {b1_metrics['pr_auc']:.3f}; macro-F1: {b1_metrics['f1_macro']:.3f}; Brier: {b1_metrics['brier']:.3f}.\n\n## Limitaciones\n- En modo synthetic_demo, los datos no provienen de GRUND Abogados LLP®.
- En modo real_anonimizado, se requiere evidencia de autorización, anonimización y validación jurídica.\n- Las relaciones fueron simuladas y no sustentan conclusiones jurídicas u organizacionales.\n- La calibración y los intervalos son inestables por el tamaño del test.\n- Requiere validación jurídica de etiquetas, variables t0 y sesgos antes de cualquier piloto real.\n\n## Usos no permitidos\nNo usar para aceptar, rechazar, valorar, priorizar o aconsejar litigios reales.\n"""
    (ROOT / 'reports' / 'model_card' / 'MODEL_CARD.md').write_text(model_card, encoding='utf-8')

    # README, bitacora, backlog.
    readme = """# VERIDICT IQ - Resultados preliminares\n\nEste paquete contiene un notebook reproducible para ejecutar los controles B0 y el baseline B1.\n\n## Advertencia\nLa ejecución incluida usa datos sintéticos exclusivamente para validar el pipeline. Antes de entregar resultados organizacionales, sustituya el archivo por el dataset anonimizado y autorizado.\n\n## Ejecución en Colab\n1. Abra `VERIDICT_IQ_Resultados_Preliminares_Colab.ipynb`.\n2. Seleccione `MODO_DATOS = 'subir'` para cargar CSV/XLSX real o conserve `sintetico` para demostración.\n3. Ejecute todas las celdas en orden.\n4. Descargue el ZIP de artefactos generado por el notebook.\n\n## Esquema mínimo esperado\n- `case_id`\n- `fecha_cierre`\n- `materia`\n- `tipo_proceso`\n- `cuantia_inicial_usd`\n- `rol_cliente`\n- `instancia_en_t0`\n- `jurisdiccion`\n- `tipo_contraparte`\n- `medida_cautelar_solicitada`\n- `prueba_pericial_prevista`\n- `resultado_favorable` (0/1)\n\n## Principios de evaluación\nEl test temporal se sella antes del preprocesamiento. El umbral se selecciona solo con predicciones OOF de desarrollo. Los resultados de test se calculan una sola vez.\n"""
    (ROOT / 'README.md').write_text(readme, encoding='utf-8')

    backlog = pd.DataFrame([
        [1, 'P0', 'DAT-01', 'Sustituir dataset sintético por snapshot anonimizado y autorizado', 'Datos', 'Equipo + GRUND', 'Dataset con hash, diccionario y permiso'],
        [2, 'P0', 'LAB-01', 'Congelar manual de etiquetado y medir acuerdo interevaluador', 'Ground truth', 'Abogados revisores', 'Kappa y reglas de casos ambiguos'],
        [3, 'P0', 'QA-01', 'Auditar variables disponibles en t0 y excluir fugas temporales', 'Calidad', 'Data scientist + abogado', 'Checklist de leakage aprobado'],
        [4, 'P0', 'EXP-01', 'Reejecutar B0 y B1 con splits versionados', 'Modelado', 'Data scientist', 'Métricas CV/test y artefactos'],
        [5, 'P1', 'MOD-01', 'Comparar class_weight y transformación log1p de cuantía', 'Modelado', 'Data scientist', 'Experimento controlado E3/E4'],
        [6, 'P1', 'CAL-01', 'Evaluar calibración sigmoid y banda de abstención', 'Evaluación', 'Data scientist + despacho', 'Brier/log-loss y cobertura'],
        [7, 'P1', 'ERR-01', 'Revisar al menos 20 FP/FN con abogados', 'Análisis de errores', 'Equipo jurídico-técnico', 'Matriz de errores y acciones'],
        [8, 'P1', 'MLOPS-01', 'Registrar run en MLflow y vincular git commit/data hash', 'MLOps', 'MLOps engineer', 'Run reproducible'],
        [9, 'P2', 'MOD-02', 'Comparar Random Forest y XGBoost contra B1', 'Modelado', 'Data scientist', 'Promoción solo con mejora consistente'],
        [10, 'P2', 'XAI-01', 'Incorporar SHAP para el modelo candidato', 'Explicabilidad', 'Data scientist', 'Explicaciones globales/locales'],
    ], columns=['orden', 'prioridad', 'id', 'tarea', 'componente', 'responsable', 'criterio_aceptacion'])
    backlog.to_csv(DOC_DIR / 'backlog_actualizado.csv', index=False)

    bitacora = pd.DataFrame([
        ['2026-07-15', 'feat', 'Se implementó generador sintético de contingencia y esquema tabular', 'Permitir validar el pipeline sin exponer datos sensibles', 'run_demo.py'],
        ['2026-07-15', 'feat', 'Se ejecutaron B0 most_frequent, B0 stratified y B1 logística con los mismos splits', 'Establecer referencia honesta y comparable', 'run_demo.py'],
        ['2026-07-15', 'fix', 'Preprocesamiento encapsulado dentro de Pipeline/ColumnTransformer', 'Evitar data leakage', 'run_demo.py'],
        ['2026-07-15', 'feat', 'Umbral seleccionado con predicciones OOF de desarrollo', 'Evitar optimización sobre test', 'run_demo.py'],
        ['2026-07-15', 'docs', 'Se generaron README, model card, métricas, figuras y backlog', 'Asegurar trazabilidad y evidencia', 'reports/ y docs/'],
    ], columns=['fecha', 'tipo_commit', 'cambio', 'justificacion', 'artefacto'])
    bitacora.to_csv(DOC_DIR / 'bitacora_cambios.csv', index=False)

    summary = {
        'metadata': metadata,
        'cv_results': cv_results,
        'test_results': test_metrics,
        'n_errors_b1': int(len(errors_only)),
        'top_errors': errors_only.head(10).to_dict(orient='records'),
        'top_coefficients': coef_df.head(10).to_dict(orient='records'),
    }
    with (METRIC_DIR / 'resumen_completo.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(json.dumps({
        'data_hash': data_hash,
        'threshold': threshold,
        'b1_test': b1_metrics,
        'n_errors': len(errors_only),
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
