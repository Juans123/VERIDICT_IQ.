from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
import warnings
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn
from sklearn.base import clone
from sklearn.calibration import calibration_curve
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
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
)

from src.data.generate_synthetic import save_synthetic_snapshot, sha256_file
from src.data.validate import validate_dataset
from src.features.build_features import add_adjusted_features
from src.models.pipelines import adjusted_param_grid, build_adjusted, build_baseline

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*penalty.*deprecated.*")

SEED = 42
TARGET = "resultado_favorable"
DATE = "fecha_cierre"
ID = "case_id"


def metric_bundle(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "f1_macro": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision_0": float(precision_score(y_true, predictions, pos_label=0, zero_division=0)),
        "recall_0": float(recall_score(y_true, predictions, pos_label=0, zero_division=0)),
        "precision_1": float(precision_score(y_true, predictions, pos_label=1, zero_division=0)),
        "recall_1": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
    }


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.20, 0.80, 121)
    scores = [
        f1_score(y_true, (probabilities >= threshold).astype(int), average="macro", zero_division=0)
        for threshold in thresholds
    ]
    index = int(np.argmax(scores))
    return float(thresholds[index]), float(scores[index])


def descriptive_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    mean = array.mean()
    if len(array) < 2:
        return mean, mean
    sem = stats.sem(array)
    margin = stats.t.ppf((1 + confidence) / 2, len(array) - 1) * sem
    return float(mean - margin), float(mean + margin)


def fit_threshold_from_training(model, X: pd.DataFrame, y: pd.Series, folds: int, seed: int) -> float:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    threshold, _ = choose_threshold(y, oof)
    return threshold


def nested_repeated_evaluation(
    X_baseline: pd.DataFrame,
    X_adjusted: pd.DataFrame,
    y: pd.Series,
    outer_folds: int = 5,
    outer_repeats: int = 3,
    inner_folds: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outer = RepeatedStratifiedKFold(
        n_splits=outer_folds, n_repeats=outer_repeats, random_state=SEED
    )
    rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []

    for fold_id, (train_idx, validation_idx) in enumerate(outer.split(X_baseline, y), start=1):
        y_train = y.iloc[train_idx]
        y_validation = y.iloc[validation_idx]
        Xb_train, Xb_validation = X_baseline.iloc[train_idx], X_baseline.iloc[validation_idx]
        Xa_train, Xa_validation = X_adjusted.iloc[train_idx], X_adjusted.iloc[validation_idx]

        baseline = build_baseline()
        baseline_threshold = fit_threshold_from_training(
            baseline, Xb_train, y_train, folds=inner_folds, seed=SEED + fold_id
        )
        start = time.perf_counter()
        baseline.fit(Xb_train, y_train)
        baseline_fit_time = time.perf_counter() - start
        baseline_prob = baseline.predict_proba(Xb_validation)[:, 1]
        baseline_metrics = metric_bundle(y_validation.to_numpy(), baseline_prob, baseline_threshold)
        rows.append(
            {
                "fold": fold_id,
                "model": "B1 baseline",
                "threshold": baseline_threshold,
                "fit_seconds": baseline_fit_time,
                **baseline_metrics,
            }
        )

        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=SEED + fold_id)
        search = GridSearchCV(
            estimator=build_adjusted(),
            param_grid=adjusted_param_grid(),
            scoring="roc_auc",
            cv=inner,
            refit=True,
            n_jobs=1,
            return_train_score=False,
        )
        search.fit(Xa_train, y_train)
        adjusted_model = search.best_estimator_
        adjusted_threshold = fit_threshold_from_training(
            adjusted_model, Xa_train, y_train, folds=inner_folds, seed=1000 + fold_id
        )
        start = time.perf_counter()
        adjusted_model.fit(Xa_train, y_train)
        adjusted_fit_time = time.perf_counter() - start
        adjusted_prob = adjusted_model.predict_proba(Xa_validation)[:, 1]
        adjusted_metrics = metric_bundle(y_validation.to_numpy(), adjusted_prob, adjusted_threshold)
        rows.append(
            {
                "fold": fold_id,
                "model": "B2 ajustado",
                "threshold": adjusted_threshold,
                "fit_seconds": adjusted_fit_time,
                **adjusted_metrics,
            }
        )
        tuning_rows.append(
            {
                "fold": fold_id,
                "best_inner_auc": float(search.best_score_),
                **search.best_params_,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(tuning_rows)


def summarize_cv(cv_results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "roc_auc",
        "average_precision",
        "f1_macro",
        "balanced_accuracy",
        "brier",
        "log_loss",
        "fit_seconds",
        "threshold",
    ]
    output: list[dict[str, object]] = []
    for model, group in cv_results.groupby("model", sort=False):
        for metric in metrics:
            values = group[metric].astype(float).tolist()
            low, high = descriptive_ci(values)
            output.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "n_folds": len(values),
                }
            )
    return pd.DataFrame(output)


def paired_cv_differences(cv_results: pd.DataFrame) -> pd.DataFrame:
    baseline = cv_results[cv_results.model == "B1 baseline"].sort_values("fold")
    adjusted = cv_results[cv_results.model == "B2 ajustado"].sort_values("fold")
    metrics = ["roc_auc", "average_precision", "f1_macro", "balanced_accuracy", "brier", "log_loss"]
    rows = []
    for metric in metrics:
        difference = adjusted[metric].to_numpy() - baseline[metric].to_numpy()
        low, high = descriptive_ci(difference.tolist())
        try:
            statistic, p_value = stats.wilcoxon(difference, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            statistic, p_value = math.nan, math.nan
        rows.append(
            {
                "metric": metric,
                "mean_difference_adjusted_minus_baseline": float(difference.mean()),
                "sd_difference": float(difference.std(ddof=1)),
                "ci95_low": low,
                "ci95_high": high,
                "wilcoxon_statistic_exploratory": float(statistic),
                "p_value_exploratory": float(p_value),
            }
        )
    return pd.DataFrame(rows)


def stratified_bootstrap(
    y_true: np.ndarray,
    baseline_prob: np.ndarray,
    adjusted_prob: np.ndarray,
    baseline_threshold: float,
    adjusted_threshold: float,
    iterations: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    class_indices = {value: np.flatnonzero(y_true == value) for value in [0, 1]}
    records: list[dict[str, object]] = []
    for iteration in range(iterations):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices.values()]
        )
        rng.shuffle(sampled)
        y_sample = y_true[sampled]
        for model_name, probability, threshold in [
            ("B1 baseline", baseline_prob[sampled], baseline_threshold),
            ("B2 ajustado", adjusted_prob[sampled], adjusted_threshold),
        ]:
            prediction = (probability >= threshold).astype(int)
            metrics = {
                "roc_auc": float(roc_auc_score(y_sample, probability)),
                "average_precision": float(average_precision_score(y_sample, probability)),
                "f1_macro": float(f1_score(y_sample, prediction, average="macro", zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y_sample, prediction)),
                "brier": float(brier_score_loss(y_sample, probability)),
                "log_loss": float(log_loss(y_sample, probability, labels=[0, 1])),
            }
            records.append({"iteration": iteration, "model": model_name, **metrics})
    raw = pd.DataFrame(records)
    summaries: list[dict[str, object]] = []
    metrics = ["roc_auc", "average_precision", "f1_macro", "balanced_accuracy", "brier", "log_loss"]
    for model, group in raw.groupby("model", sort=False):
        for metric in metrics:
            values = group[metric].to_numpy()
            summaries.append(
                {
                    "model": model,
                    "metric": metric,
                    "bootstrap_mean": float(values.mean()),
                    "ci95_low": float(np.quantile(values, 0.025)),
                    "ci95_high": float(np.quantile(values, 0.975)),
                }
            )
    summary = pd.DataFrame(summaries)
    differences = []
    base = raw[raw.model == "B1 baseline"].sort_values("iteration")
    adj = raw[raw.model == "B2 ajustado"].sort_values("iteration")
    for metric in metrics:
        values = adj[metric].to_numpy() - base[metric].to_numpy()
        differences.append(
            {
                "metric": metric,
                "mean_difference_adjusted_minus_baseline": float(values.mean()),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "probability_difference_gt_zero": float(np.mean(values > 0)),
            }
        )
    return summary, pd.DataFrame(differences)


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    indices = np.digitize(probabilities, edges[1:-1], right=True)
    ece = 0.0
    for bin_id in range(bins):
        mask = indices == bin_id
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - probabilities[mask].mean())
    return float(ece)


def plot_cv_distribution(cv_results: pd.DataFrame, metric: str, output: Path, ylabel: str) -> None:
    models = ["B1 baseline", "B2 ajustado"]
    data = [cv_results.loc[cv_results.model == model, metric].to_numpy() for model in models]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.boxplot(data, tick_labels=models, showmeans=True)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Distribución en validación cruzada anidada: {ylabel}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_roc_pr(y_test, base_prob, adj_prob, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for label, probability in [("B1 baseline", base_prob), ("B2 ajustado", adj_prob)]:
        fpr, tpr, _ = roc_curve(y_test, probability)
        auc = roc_auc_score(y_test, probability)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Sensibilidad")
    ax.set_title("Curvas ROC en test temporal sellado")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_test.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    prevalence = float(np.mean(y_test))
    for label, probability in [("B1 baseline", base_prob), ("B2 ajustado", adj_prob)]:
        precision, recall, _ = precision_recall_curve(y_test, probability)
        ap = average_precision_score(y_test, probability)
        ax.plot(recall, precision, label=f"{label} (PR-AUC={ap:.3f})")
    ax.axhline(prevalence, linestyle="--", linewidth=1, label=f"Prevalencia={prevalence:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisión")
    ax.set_title("Curvas precisión-recall en test temporal sellado")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "pr_test.png", dpi=220)
    plt.close(fig)


def plot_calibration(y_test, base_prob, adj_prob, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for label, probability in [("B1 baseline", base_prob), ("B2 ajustado", adj_prob)]:
        observed, predicted = calibration_curve(y_test, probability, n_bins=7, strategy="quantile")
        ax.plot(predicted, observed, marker="o", label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Calibración perfecta")
    ax.set_xlabel("Probabilidad media predicha")
    ax.set_ylabel("Frecuencia observada")
    ax.set_title("Calibración en test temporal")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["No favorable", "Favorable"])
    ax.set_yticks([0, 1], labels=["No favorable", "Favorable"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Observado")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def coefficient_table(adjusted_model) -> pd.DataFrame:
    preprocessor = adjusted_model.named_steps["preprocessor"]
    classifier = adjusted_model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "odds_ratio": np.exp(coefficients),
            "absolute_coefficient": np.abs(coefficients),
        }
    )
    return frame.sort_values("absolute_coefficient", ascending=False).reset_index(drop=True)


def plot_coefficients(coefficients: pd.DataFrame, output: Path, top_n: int = 15) -> None:
    data = coefficients.head(top_n).sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.barh(data["feature"], data["coefficient"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Coeficiente log-odds")
    ax.set_title("Variables con mayor magnitud de coeficiente en B2")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def segment_metrics(test: pd.DataFrame, y_test: pd.Series, probability: np.ndarray, threshold: float) -> pd.DataFrame:
    rows = []
    for variable in ["materia", "rol_cliente"]:
        for value, indices in test.groupby(variable).groups.items():
            positions = test.index.get_indexer(indices)
            y_group = y_test.iloc[positions].to_numpy()
            p_group = probability[positions]
            if len(np.unique(y_group)) < 2:
                auc = math.nan
            else:
                auc = float(roc_auc_score(y_group, p_group))
            prediction = (p_group >= threshold).astype(int)
            rows.append(
                {
                    "segment_variable": variable,
                    "segment": value,
                    "n": len(y_group),
                    "positive_rate": float(y_group.mean()),
                    "roc_auc": auc,
                    "f1_macro": float(f1_score(y_group, prediction, average="macro", zero_division=0)),
                    "recall_0": float(recall_score(y_group, prediction, pos_label=0, zero_division=0)),
                    "recall_1": float(recall_score(y_group, prediction, pos_label=1, zero_division=0)),
                }
            )
    return pd.DataFrame(rows)


def final_fit_and_test(dev: pd.DataFrame, test: pd.DataFrame, output_dir: Path):
    Xb_dev = dev.drop(columns=[TARGET])
    Xb_test = test.drop(columns=[TARGET])
    Xa_dev = add_adjusted_features(Xb_dev)
    Xa_test = add_adjusted_features(Xb_test)
    y_dev = dev[TARGET].astype(int)
    y_test = test[TARGET].astype(int)

    baseline = build_baseline()
    baseline_threshold = fit_threshold_from_training(baseline, Xb_dev, y_dev, folds=5, seed=SEED)
    start = time.perf_counter()
    baseline.fit(Xb_dev, y_dev)
    base_fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    base_prob = baseline.predict_proba(Xb_test)[:, 1]
    base_inference_ms = (time.perf_counter() - start) / len(Xb_test) * 1000

    search = GridSearchCV(
        build_adjusted(),
        adjusted_param_grid(),
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        refit=True,
        n_jobs=1,
    )
    search.fit(Xa_dev, y_dev)
    adjusted = search.best_estimator_
    adjusted_threshold = fit_threshold_from_training(adjusted, Xa_dev, y_dev, folds=5, seed=SEED + 1)
    start = time.perf_counter()
    adjusted.fit(Xa_dev, y_dev)
    adjusted_fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    adjusted_prob = adjusted.predict_proba(Xa_test)[:, 1]
    adjusted_inference_ms = (time.perf_counter() - start) / len(Xa_test) * 1000

    test_metrics = []
    for model, probability, threshold, fit_time, inference_time in [
        ("B1 baseline", base_prob, baseline_threshold, base_fit_seconds, base_inference_ms),
        ("B2 ajustado", adjusted_prob, adjusted_threshold, adjusted_fit_seconds, adjusted_inference_ms),
    ]:
        metrics = metric_bundle(y_test.to_numpy(), probability, threshold)
        metrics.update(
            {
                "model": model,
                "threshold": threshold,
                "fit_seconds": fit_time,
                "inference_ms_per_case": inference_time,
                "ece_10_bins": expected_calibration_error(y_test.to_numpy(), probability),
            }
        )
        test_metrics.append(metrics)
    test_metrics_df = pd.DataFrame(test_metrics)

    boot_summary, boot_deltas = stratified_bootstrap(
        y_test.to_numpy(),
        base_prob,
        adjusted_prob,
        baseline_threshold,
        adjusted_threshold,
        iterations=500,
    )

    base_pred = (base_prob >= baseline_threshold).astype(int)
    adjusted_pred = (adjusted_prob >= adjusted_threshold).astype(int)
    base_cm = confusion_matrix(y_test, base_pred)
    adjusted_cm = confusion_matrix(y_test, adjusted_pred)

    coefficients = coefficient_table(adjusted)
    segments = segment_metrics(test.reset_index(drop=True), y_test.reset_index(drop=True), adjusted_prob, adjusted_threshold)

    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(baseline, models_dir / "baseline_b1.joblib")
    joblib.dump(adjusted, models_dir / "adjusted_b2.joblib")

    predictions = pd.DataFrame(
        {
            "case_id": test[ID].to_numpy(),
            "fecha_cierre": test[DATE].astype(str).to_numpy(),
            "y_true": y_test.to_numpy(),
            "p_baseline": base_prob,
            "pred_baseline": base_pred,
            "p_adjusted": adjusted_prob,
            "pred_adjusted": adjusted_pred,
        }
    )
    return {
        "baseline": baseline,
        "adjusted": adjusted,
        "best_params": search.best_params_,
        "test_metrics": test_metrics_df,
        "bootstrap_summary": boot_summary,
        "bootstrap_deltas": boot_deltas,
        "baseline_cm": base_cm,
        "adjusted_cm": adjusted_cm,
        "coefficients": coefficients,
        "segments": segments,
        "predictions": predictions,
        "y_test": y_test.to_numpy(),
        "base_prob": base_prob,
        "adjusted_prob": adjusted_prob,
        "baseline_threshold": baseline_threshold,
        "adjusted_threshold": adjusted_threshold,
    }


def main(project_dir: Path) -> None:
    project_dir = project_dir.resolve()
    figure_dir = project_dir / "reports" / "figures"
    metric_dir = project_dir / "reports" / "metrics"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    raw_path = project_dir / "data" / "raw" / "veridict_synthetic_demo.csv"
    df, data_hash = save_synthetic_snapshot(raw_path, n=360, seed=SEED)
    summary = validate_dataset(df)
    df[DATE] = pd.to_datetime(df[DATE])
    df = df.sort_values(DATE).reset_index(drop=True)
    split_index = int(len(df) * 0.80)
    dev = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()
    dev.to_csv(project_dir / "data" / "processed" / "development.csv", index=False)
    test.to_csv(project_dir / "data" / "processed" / "temporal_test.csv", index=False)

    X_baseline = dev.drop(columns=[TARGET])
    X_adjusted = add_adjusted_features(X_baseline)
    y_dev = dev[TARGET].astype(int)

    print("START nested", flush=True)
    cv_results, tuning_results = nested_repeated_evaluation(
        X_baseline, X_adjusted, y_dev, outer_folds=5, outer_repeats=2, inner_folds=3
    )
    print("END nested", flush=True)
    cv_summary = summarize_cv(cv_results)
    cv_differences = paired_cv_differences(cv_results)
    print("START final", flush=True)
    final = final_fit_and_test(dev, test, project_dir)
    print("END final", flush=True)

    cv_results.to_csv(metric_dir / "nested_cv_folds.csv", index=False)
    tuning_results.to_csv(metric_dir / "nested_cv_tuning.csv", index=False)
    cv_summary.to_csv(metric_dir / "nested_cv_summary.csv", index=False)
    cv_differences.to_csv(metric_dir / "nested_cv_paired_differences.csv", index=False)
    final["test_metrics"].to_csv(metric_dir / "temporal_test_metrics.csv", index=False)
    final["bootstrap_summary"].to_csv(metric_dir / "temporal_test_bootstrap_ci.csv", index=False)
    final["bootstrap_deltas"].to_csv(metric_dir / "temporal_test_bootstrap_deltas.csv", index=False)
    final["coefficients"].to_csv(metric_dir / "adjusted_coefficients.csv", index=False)
    final["segments"].to_csv(metric_dir / "segment_metrics_adjusted.csv", index=False)
    final["predictions"].to_csv(metric_dir / "temporal_test_predictions.csv", index=False)

    plot_cv_distribution(cv_results, "roc_auc", figure_dir / "cv_auc_boxplot.png", "AUC-ROC")
    plot_cv_distribution(cv_results, "f1_macro", figure_dir / "cv_f1_boxplot.png", "F1 macro")
    plot_roc_pr(final["y_test"], final["base_prob"], final["adjusted_prob"], figure_dir)
    plot_calibration(final["y_test"], final["base_prob"], final["adjusted_prob"], figure_dir / "calibration_test.png")
    plot_confusion(final["baseline_cm"], "Matriz de confusión - B1 baseline", figure_dir / "confusion_baseline.png")
    plot_confusion(final["adjusted_cm"], "Matriz de confusión - B2 ajustado", figure_dir / "confusion_adjusted.png")
    plot_coefficients(final["coefficients"], figure_dir / "coefficients_adjusted.png")

    metadata = {
        "run_date": pd.Timestamp.now(tz="UTC").isoformat(),
        "data_mode": "synthetic_demo",
        "warning": "Engineering validation only; not evidence about GRUND or Ecuadorian courts.",
        "data_sha256": data_hash,
        "dataset_validation": asdict(summary),
        "n_total": len(df),
        "n_development": len(dev),
        "n_temporal_test": len(test),
        "development_period": [str(dev[DATE].min().date()), str(dev[DATE].max().date())],
        "test_period": [str(test[DATE].min().date()), str(test[DATE].max().date())],
        "best_adjusted_params": final["best_params"],
        "baseline_threshold": final["baseline_threshold"],
        "adjusted_threshold": final["adjusted_threshold"],
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": stats.__version__ if hasattr(stats, "__version__") else "see scipy package",
        },
        "model_hashes": {
            "baseline_b1": sha256_file(project_dir / "models" / "baseline_b1.joblib"),
            "adjusted_b2": sha256_file(project_dir / "models" / "adjusted_b2.joblib"),
        },
    }
    with (metric_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2, default=str)

    print(json.dumps({
        "data_sha256": data_hash,
        "best_adjusted_params": final["best_params"],
        "test_metrics": final["test_metrics"].to_dict(orient="records"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VERIDICT IQ adjusted validation experiment.")
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    main(arguments.project_dir)
