from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_predict

from src.features.build_features import add_adjusted_features
from src.models.pipelines import _preprocessor, ADJUSTED_NUMERIC, build_adjusted, build_baseline

SEED = 42
TARGET = "resultado_favorable"
THRESHOLD = 0.49
ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "reports" / "figures" / "s7"
METRIC_DIR = ROOT / "reports" / "metrics" / "s7"
FIG_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    dev = pd.read_csv(ROOT / "data" / "processed" / "development.csv")
    test = pd.read_csv(ROOT / "data" / "processed" / "temporal_test.csv")
    model = joblib.load(ROOT / "models" / "adjusted_b2.joblib")
    return dev, test, model


def get_transformed(model, df):
    engineered = add_adjusted_features(df.drop(columns=[TARGET], errors="ignore"))
    pre = model.named_steps["preprocessor"]
    transformed = pre.transform(engineered)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    names = pre.get_feature_names_out()
    return engineered, np.asarray(transformed, dtype=float), np.asarray(names, dtype=str)


def shap_analysis(dev, test, model):
    _, X_dev_t, names = get_transformed(model, dev)
    _, X_test_t, _ = get_transformed(model, test)
    clf = model.named_steps["classifier"]

    background = shap.maskers.Independent(X_dev_t)
    explainer = shap.LinearExplainer(clf, background)
    explanation = explainer(X_test_t)
    shap_values = np.asarray(explanation.values)

    importance = pd.DataFrame({
        "feature": names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "mean_shap": shap_values.mean(axis=0),
        "coefficient": clf.coef_[0],
        "odds_ratio": np.exp(clf.coef_[0]),
    }).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(METRIC_DIR / "shap_global_importance.csv", index=False)

    top = importance.head(15).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.barh(top["feature"], top["mean_abs_shap"])
    ax.set_xlabel("Media de |SHAP| (log-odds)")
    ax.set_title("Importancia global SHAP del modelo B2")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "shap_global_bar.png", dpi=220)
    plt.close(fig)

    shap.summary_plot(shap_values, X_test_t, feature_names=names, show=False, max_display=15)
    plt.title("Resumen SHAP del modelo B2")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_beeswarm.png", dpi=220, bbox_inches="tight")
    plt.close()

    probs = model.predict_proba(add_adjusted_features(test.drop(columns=[TARGET])))[:, 1]
    preds = (probs >= THRESHOLD).astype(int)
    y = test[TARGET].to_numpy()
    categories = {
        "verdadero_positivo": np.flatnonzero((y == 1) & (preds == 1)),
        "verdadero_negativo": np.flatnonzero((y == 0) & (preds == 0)),
        "falso_positivo": np.flatnonzero((y == 0) & (preds == 1)),
        "falso_negativo": np.flatnonzero((y == 1) & (preds == 0)),
    }
    selected = []
    for label, idxs in categories.items():
        if len(idxs) == 0:
            continue
        if label in {"falso_positivo", "verdadero_positivo"}:
            idx = idxs[np.argmax(probs[idxs])]
        else:
            idx = idxs[np.argmin(probs[idxs])]
        selected.append({"case_type": label, "row_index": int(idx), "case_id": test.iloc[idx]["case_id"],
                         "actual": int(y[idx]), "predicted": int(preds[idx]), "probability": float(probs[idx])})
        shap.plots.waterfall(explanation[idx], max_display=12, show=False)
        plt.title(f"SHAP local: {label} ({test.iloc[idx]['case_id']})")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"shap_waterfall_{label}.png", dpi=220, bbox_inches="tight")
        plt.close()
    pd.DataFrame(selected).to_csv(METRIC_DIR / "selected_local_cases.csv", index=False)
    np.save(METRIC_DIR / "shap_values_test.npy", shap_values)
    return explanation, importance, selected, X_dev_t, X_test_t, names, probs


def lime_style_local_surrogate(X_train, x, model_predict, feature_names, seed=42, n_samples=3000):
    """LIME-style local linear surrogate in transformed feature space.

    This is dependency-free and follows the core LIME idea: local perturbations,
    proximity weighting, and a sparse-ish linear surrogate. It is not the external lime package.
    """
    rng = np.random.default_rng(seed)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    samples = x + rng.normal(0, 0.35 * std, size=(n_samples, X_train.shape[1]))
    samples[0] = x
    distances = np.sqrt(np.sum(((samples - x) / std) ** 2, axis=1))
    kernel_width = math.sqrt(X_train.shape[1]) * 0.75
    weights = np.exp(-(distances ** 2) / (kernel_width ** 2))
    y_local = model_predict(samples)
    surrogate = Ridge(alpha=1.0)
    surrogate.fit(samples, y_local, sample_weight=weights)
    local_pred = surrogate.predict(x.reshape(1, -1))[0]
    fidelity = surrogate.score(samples, y_local, sample_weight=weights)
    table = pd.DataFrame({"feature": feature_names, "local_weight": surrogate.coef_})
    table["abs_weight"] = table["local_weight"].abs()
    return table.sort_values("abs_weight", ascending=False), float(local_pred), float(fidelity)


def local_surrogate_analysis(model, X_dev_t, X_test_t, names, selected):
    clf = model.named_steps["classifier"]
    rows = []
    for item in selected:
        idx = item["row_index"]
        table, pred, fidelity = lime_style_local_surrogate(
            X_dev_t, X_test_t[idx], lambda z: clf.predict_proba(z)[:, 1], names, seed=SEED + idx
        )
        table.head(15).to_csv(METRIC_DIR / f"local_surrogate_{item['case_type']}.csv", index=False)
        top = table.head(12).sort_values("local_weight")
        fig, ax = plt.subplots(figsize=(8.5, 5.4))
        ax.barh(top["feature"], top["local_weight"])
        ax.axvline(0, linewidth=0.8)
        ax.set_xlabel("Peso del sustituto local")
        ax.set_title(f"Sustituto local tipo LIME: {item['case_type']}")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"local_surrogate_{item['case_type']}.png", dpi=220)
        plt.close(fig)
        rows.append({**item, "surrogate_probability": pred, "local_fidelity_r2": fidelity})
    pd.DataFrame(rows).to_csv(METRIC_DIR / "local_surrogate_summary.csv", index=False)


def fairness_metrics(test, probabilities, threshold=THRESHOLD):
    y = test[TARGET].to_numpy()
    pred = (probabilities >= threshold).astype(int)
    rows = []
    for group_name, group in test.groupby("rol_cliente"):
        idx = group.index.to_numpy() - test.index.min()
        yg, pg, probg = y[idx], pred[idx], probabilities[idx]
        tn, fp, fn, tp = confusion_matrix(yg, pg, labels=[0, 1]).ravel()
        rows.append({
            "group": group_name, "n": len(idx), "prevalence": float(yg.mean()),
            "tpr_recall_1": float(tp / (tp + fn)) if tp + fn else np.nan,
            "fpr": float(fp / (fp + tn)) if fp + tn else np.nan,
            "tnr_recall_0": float(tn / (tn + fp)) if tn + fp else np.nan,
            "selection_rate": float(pg.mean()),
            "auc": float(roc_auc_score(yg, probg)) if len(np.unique(yg)) == 2 else np.nan,
            "f1_macro": float(f1_score(yg, pg, average="macro", zero_division=0)),
        })
    out = pd.DataFrame(rows)
    out.to_csv(METRIC_DIR / "fairness_by_role.csv", index=False)
    if len(out) >= 2:
        eq_odds = max(out.tpr_recall_1.max() - out.tpr_recall_1.min(), out.fpr.max() - out.fpr.min())
        summary = {
            "equalized_odds_difference": float(eq_odds),
            "tpr_difference": float(out.tpr_recall_1.max() - out.tpr_recall_1.min()),
            "fpr_difference": float(out.fpr.max() - out.fpr.min()),
            "selection_rate_difference": float(out.selection_rate.max() - out.selection_rate.min()),
        }
    else:
        summary = {}
    (METRIC_DIR / "fairness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.3, 4.5))
    plot_df = out.set_index("group")[["tpr_recall_1", "fpr", "selection_rate"]]
    plot_df.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Proporción")
    ax.set_title("Métricas de equidad por rol del cliente")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fairness_by_role.png", dpi=220)
    plt.close(fig)

    # Stratified-within-group bootstrap CIs for fairness gaps
    rng = np.random.default_rng(SEED)
    records = []
    group_indices = {g: np.flatnonzero(test["rol_cliente"].to_numpy() == g) for g in out.group}
    for b in range(2000):
        metrics = {}
        for g, idx in group_indices.items():
            sampled = rng.choice(idx, len(idx), replace=True)
            yg, pg = y[sampled], pred[sampled]
            tn, fp, fn, tp = confusion_matrix(yg, pg, labels=[0, 1]).ravel()
            metrics[g] = {
                "tpr": tp / (tp + fn) if tp + fn else np.nan,
                "fpr": fp / (fp + tn) if fp + tn else np.nan,
            }
        if len(metrics) == 2:
            vals = list(metrics.values())
            if not any(np.isnan([vals[0]["tpr"], vals[1]["tpr"], vals[0]["fpr"], vals[1]["fpr"]])):
                tpr_gap = abs(vals[0]["tpr"] - vals[1]["tpr"])
                fpr_gap = abs(vals[0]["fpr"] - vals[1]["fpr"])
                records.append({"iteration": b, "tpr_gap": tpr_gap, "fpr_gap": fpr_gap,
                                "equalized_odds_difference": max(tpr_gap, fpr_gap)})
    boot = pd.DataFrame(records)
    boot.to_csv(METRIC_DIR / "fairness_bootstrap_2000.csv", index=False)
    if not boot.empty:
        ci = []
        for c in ["tpr_gap", "fpr_gap", "equalized_odds_difference"]:
            ci.append({"metric": c, "mean": boot[c].mean(), "ci95_low": boot[c].quantile(.025),
                       "ci95_high": boot[c].quantile(.975)})
        pd.DataFrame(ci).to_csv(METRIC_DIR / "fairness_bootstrap_ci.csv", index=False)
    return out, summary


def perturbation_stability(dev, test, model, base_shap, base_probs, names, repetitions=100):
    rng = np.random.default_rng(SEED)
    mean_q = abs(dev["cuantia_inicial_usd"].mean(skipna=True))
    sigma = 0.05 * mean_q
    clf = model.named_steps["classifier"]
    pre = model.named_steps["preprocessor"]
    base_rank = np.argsort(-np.abs(base_shap).mean(axis=0))
    top5 = set(base_rank[:5])
    rows = []
    for r in range(repetitions):
        pert = test.copy()
        mask = pert["cuantia_inicial_usd"].notna()
        pert.loc[mask, "cuantia_inicial_usd"] = np.maximum(
            0, pert.loc[mask, "cuantia_inicial_usd"] + rng.normal(0, sigma, mask.sum())
        )
        engineered = add_adjusted_features(pert.drop(columns=[TARGET]))
        probs = model.predict_proba(engineered)[:, 1]
        Xt = pre.transform(engineered)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        explainer = shap.LinearExplainer(clf, shap.maskers.Independent(pre.transform(add_adjusted_features(dev.drop(columns=[TARGET])))))
        sv = np.asarray(explainer(np.asarray(Xt)).values)
        rank = np.argsort(-np.abs(sv).mean(axis=0))
        spear = stats.spearmanr(np.argsort(base_rank), np.argsort(rank)).statistic
        jacc = len(top5 & set(rank[:5])) / len(top5 | set(rank[:5]))
        rows.append({
            "repetition": r,
            "sigma": sigma,
            "mean_abs_probability_change": float(np.mean(np.abs(probs - base_probs))),
            "max_abs_probability_change": float(np.max(np.abs(probs - base_probs))),
            "classification_flip_rate": float(np.mean((probs >= THRESHOLD) != (base_probs >= THRESHOLD))),
            "shap_rank_spearman": float(spear),
            "top5_jaccard": float(jacc),
        })
    out = pd.DataFrame(rows)
    out.to_csv(METRIC_DIR / "perturbation_stability.csv", index=False)
    summary = out.drop(columns=["repetition"]).agg(["mean", "std", "min", "max"]).T.reset_index().rename(columns={"index":"metric"})
    summary.to_csv(METRIC_DIR / "perturbation_stability_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.hist(out["mean_abs_probability_change"], bins=15)
    ax.set_xlabel("Cambio absoluto medio de probabilidad")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Estabilidad ante ruido gaussiano controlado")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "perturbation_probability_change.png", dpi=220)
    plt.close(fig)
    return out


def evaluate_smote(dev, test):
    X_dev = add_adjusted_features(dev.drop(columns=[TARGET]))
    y_dev = dev[TARGET]
    X_test = add_adjusted_features(test.drop(columns=[TARGET]))
    y_test = test[TARGET].to_numpy()

    pre = _preprocessor(ADJUSTED_NUMERIC)
    smote_model = ImbPipeline([
        ("preprocessor", pre),
        ("smote", SMOTE(random_state=SEED, k_neighbors=5)),
        ("classifier", LogisticRegression(C=0.2, solver="liblinear", l1_ratio=1.0,
                                          max_iter=2000, random_state=SEED)),
    ])
    current = build_adjusted().set_params(classifier__C=0.2, classifier__l1_ratio=1.0,
                                          classifier__class_weight="balanced")
    models = {"B2_class_weight": current, "B2_SMOTE": smote_model}
    rows = []
    for name, mod in models.items():
        mod.fit(X_dev, y_dev)
        p = mod.predict_proba(X_test)[:, 1]
        pred = (p >= THRESHOLD).astype(int)
        role = test["rol_cliente"].to_numpy()
        actor_idx = role == "Actor"
        demand_idx = role == "Demandado"
        rows.append({
            "model": name,
            "roc_auc": roc_auc_score(y_test, p),
            "pr_auc": average_precision_score(y_test, p),
            "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "brier": brier_score_loss(y_test, p),
            "log_loss": log_loss(y_test, p),
            "recall_1_actor": recall_score(y_test[actor_idx], pred[actor_idx], pos_label=1, zero_division=0),
            "recall_1_demandado": recall_score(y_test[demand_idx], pred[demand_idx], pos_label=1, zero_division=0),
        })
    out = pd.DataFrame(rows)
    out["recall_1_gap"] = (out["recall_1_actor"] - out["recall_1_demandado"]).abs()
    out.to_csv(METRIC_DIR / "smote_comparison.csv", index=False)
    return out


def bootstrap_model_metrics(test, probabilities, iterations=2000):
    y = test[TARGET].to_numpy()
    rng = np.random.default_rng(SEED)
    cls = {v: np.flatnonzero(y == v) for v in [0,1]}
    rows=[]
    for b in range(iterations):
        idx=np.concatenate([rng.choice(v,len(v),replace=True) for v in cls.values()]); rng.shuffle(idx)
        yy=y[idx]; pp=probabilities[idx]; pred=(pp>=THRESHOLD).astype(int)
        rows.append({"iteration":b,"roc_auc":roc_auc_score(yy,pp),"pr_auc":average_precision_score(yy,pp),
                     "f1_macro":f1_score(yy,pred,average='macro',zero_division=0),
                     "balanced_accuracy":balanced_accuracy_score(yy,pred),"brier":brier_score_loss(yy,pp),
                     "log_loss":log_loss(yy,pp)})
    raw=pd.DataFrame(rows); raw.to_csv(METRIC_DIR/'b2_bootstrap_2000.csv',index=False)
    ci=[]
    for c in raw.columns[1:]:
        ci.append({'metric':c,'mean':raw[c].mean(),'ci95_low':raw[c].quantile(.025),'ci95_high':raw[c].quantile(.975)})
    pd.DataFrame(ci).to_csv(METRIC_DIR/'b2_bootstrap_2000_ci.csv',index=False)


def dietterich_5x2cv(dev):
    Xb = dev.drop(columns=[TARGET])
    Xa = add_adjusted_features(Xb)
    y = dev[TARGET].to_numpy()
    splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=SEED)
    diffs=[]; rows=[]
    for i,(a,b) in enumerate(splitter.split(Xb,y),start=1):
        pair=[]
        for direction,(tr,te) in enumerate([(a,b),(b,a)],start=1):
            b1=build_baseline(); b2=build_adjusted().set_params(classifier__C=.2,classifier__l1_ratio=1.0,classifier__class_weight='balanced')
            b1.fit(Xb.iloc[tr],y[tr]); b2.fit(Xa.iloc[tr],y[tr])
            p1=(b1.predict_proba(Xb.iloc[te])[:,1]>=THRESHOLD).astype(int)
            p2=(b2.predict_proba(Xa.iloc[te])[:,1]>=THRESHOLD).astype(int)
            s1=balanced_accuracy_score(y[te],p1); s2=balanced_accuracy_score(y[te],p2); d=s2-s1
            pair.append(d); rows.append({'replication':i,'direction':direction,'b1_balanced_accuracy':s1,'b2_balanced_accuracy':s2,'difference':d})
        diffs.append(pair)
    diffs=np.asarray(diffs)
    variances=(diffs[:,0]-diffs[:,1])**2
    denom=math.sqrt(np.mean(variances)) if np.mean(variances)>0 else np.nan
    t_stat=diffs[0,0]/denom if denom and not np.isnan(denom) else np.nan
    p=2*(1-stats.t.cdf(abs(t_stat),df=5)) if not np.isnan(t_stat) else np.nan
    pd.DataFrame(rows).to_csv(METRIC_DIR/'dietterich_5x2cv_folds.csv',index=False)
    pd.DataFrame([{'t_statistic':t_stat,'df':5,'p_value_two_sided':p,'metric':'balanced_accuracy',
                   'note':'Adaptación exploratoria del 5x2cv paired t-test; no sustituye validación temporal.'}]).to_csv(METRIC_DIR/'dietterich_5x2cv_summary.csv',index=False)


def main():
    dev, test, model = load_data()
    explanation, importance, selected, Xdevt, Xtestt, names, probs = shap_analysis(dev,test,model)
    local_surrogate_analysis(model,Xdevt,Xtestt,names,selected)
    fairness_metrics(test,probs)
    perturbation_stability(dev,test,model,np.asarray(explanation.values),probs,names)
    evaluate_smote(dev,test)
    bootstrap_model_metrics(test,probs,2000)
    dietterich_5x2cv(dev)
    metadata={
        'seed':SEED,'threshold':THRESHOLD,'development_rows':len(dev),'temporal_test_rows':len(test),
        'shap_version':shap.__version__,'note':'All analyses use synthetic data and are engineering validation only.'
    }
    (METRIC_DIR/'s7_run_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('S7 analysis completed:', METRIC_DIR)

if __name__=='__main__':
    main()
