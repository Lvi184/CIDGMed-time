"""Evaluate CIDGMed outputs.

This script evaluates three parts of the project:
1) multi-label medication regimen prediction;
2) LOS and step-duration regression;
3) causal-effect diagnostics for single-medication ATE estimates.

It is designed to run after scripts/run_pipeline.sh has generated model predictions
and causal outputs in the same processed directory.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# matplotlib is only imported inside plotting functions so metric computation can
# still run in minimal environments.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    coverage_error,
    f1_score,
    hamming_loss,
    jaccard_score,
    label_ranking_average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
    zero_one_loss,
)

from configs import data_config as cfg
from src.utils.causal_utils import doubly_robust_ate, stabilized_ipw_binary
from src.utils.io_utils import ensure_dir, load_pickle
from src.utils.preprocess_utils import split_codes


def _safe_float(x) -> Optional[float]:
    try:
        if x is None or not np.isfinite(float(x)):
            return None
        return float(x)
    except Exception:
        return None


def _json_ready(obj):
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return _json_ready(obj.tolist())
    return obj


def _select_subset(arr, indices):
    if indices is None:
        return arr
    return arr[indices]


def load_subset_indices(processed_dir: Path, subset: str) -> Optional[np.ndarray]:
    subset = subset.lower()
    if subset == "all":
        return None
    path = processed_dir / f"{subset}_indices.npy"
    if path.exists():
        return np.load(path).astype(int)
    print(f"[WARN] {path.name} not found; evaluating on all rows instead.")
    return None


def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, top_ks: Iterable[int]) -> Tuple[Dict, np.ndarray]:
    y_true = y_true.astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # Avoid empty predictions: for rows with no label above threshold, select the top-1 candidate.
    empty_rows = np.where(y_pred.sum(axis=1) == 0)[0]
    if len(empty_rows) > 0 and y_prob.shape[1] > 0:
        y_pred[empty_rows, np.argmax(y_prob[empty_rows], axis=1)] = 1

    out = {
        "threshold": threshold,
        "n_samples": int(y_true.shape[0]),
        "n_labels": int(y_true.shape[1]),
        "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "samples_f1": f1_score(y_true, y_pred, average="samples", zero_division=0),
        "jaccard_samples": jaccard_score(y_true, y_pred, average="samples", zero_division=0),
        "jaccard_micro": jaccard_score(y_true, y_pred, average="micro", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "subset_exact_match": 1.0 - zero_one_loss(y_true, y_pred),
        "mean_true_meds_per_step": float(y_true.sum(axis=1).mean()),
        "mean_pred_meds_per_step": float(y_pred.sum(axis=1).mean()),
    }

    for k in top_ks:
        k = int(k)
        if k <= 0 or y_prob.shape[1] == 0:
            continue
        kk = min(k, y_prob.shape[1])
        order = np.argsort(-y_prob, axis=1)[:, :kk]
        hits = []
        precision_hits = []
        exact_any = []
        for i in range(y_true.shape[0]):
            true_set = set(np.where(y_true[i] == 1)[0])
            pred_set = set(order[i].tolist())
            if len(true_set) == 0:
                continue
            n_hit = len(true_set & pred_set)
            hits.append(n_hit / len(true_set))
            precision_hits.append(n_hit / kk)
            exact_any.append(1.0 if n_hit > 0 else 0.0)
        out[f"recall_at_{k}"] = float(np.mean(hits)) if hits else None
        out[f"precision_at_{k}"] = float(np.mean(precision_hits)) if precision_hits else None
        out[f"hit_rate_at_{k}"] = float(np.mean(exact_any)) if exact_any else None

    try:
        out["label_ranking_average_precision"] = label_ranking_average_precision_score(y_true, y_prob)
        out["coverage_error"] = coverage_error(y_true, y_prob)
    except Exception:
        out["label_ranking_average_precision"] = None
        out["coverage_error"] = None

    # Global AUROC/AUPRC can fail if a label has only one class; handle gracefully.
    try:
        out["micro_roc_auc"] = roc_auc_score(y_true.ravel(), y_prob.ravel())
    except Exception:
        out["micro_roc_auc"] = None
    try:
        out["micro_average_precision"] = average_precision_score(y_true.ravel(), y_prob.ravel())
    except Exception:
        out["micro_average_precision"] = None

    return out, y_pred


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> Dict:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"target": name, "n_samples": 0}
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    medae = float(np.median(np.abs(y_true - y_pred)))
    mape_mask = np.abs(y_true) > 1e-8
    mape = float(np.mean(np.abs((y_true[mape_mask] - y_pred[mape_mask]) / y_true[mape_mask]))) if np.any(mape_mask) else None
    return {
        "target": name,
        "n_samples": int(len(y_true)),
        "mae": mae,
        "median_absolute_error": medae,
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(y_true, y_pred) if len(y_true) > 1 else None,
        "mape": mape,
        "mean_true": float(np.mean(y_true)),
        "mean_pred": float(np.mean(y_pred)),
        "bias_mean_pred_minus_true": float(np.mean(y_pred - y_true)),
    }


def weighted_smd_summary(X: np.ndarray, t: np.ndarray, w: Optional[np.ndarray] = None) -> Dict:
    X = np.asarray(X, dtype=float)
    t = np.asarray(t, dtype=int)
    mask_t = t == 1
    mask_c = t == 0
    if mask_t.sum() == 0 or mask_c.sum() == 0:
        return {"mean_abs_smd": None, "max_abs_smd": None, "pct_abs_smd_lt_0_1": None}

    def _wm(a, weights=None):
        if weights is None:
            return np.mean(a, axis=0)
        return np.average(a, axis=0, weights=weights)

    def _wv(a, mean, weights=None):
        if weights is None:
            return np.var(a, axis=0)
        return np.average((a - mean) ** 2, axis=0, weights=weights)

    wt = w[mask_t] if w is not None else None
    wc = w[mask_c] if w is not None else None
    mt = _wm(X[mask_t], wt)
    mc = _wm(X[mask_c], wc)
    vt = _wv(X[mask_t], mt, wt)
    vc = _wv(X[mask_c], mc, wc)
    denom = np.sqrt((vt + vc) / 2.0)
    smd = np.divide(mt - mc, denom, out=np.zeros_like(mt), where=denom > 1e-12)
    abs_smd = np.abs(smd[np.isfinite(smd)])
    if len(abs_smd) == 0:
        return {"mean_abs_smd": None, "max_abs_smd": None, "pct_abs_smd_lt_0_1": None}
    return {
        "mean_abs_smd": float(np.mean(abs_smd)),
        "max_abs_smd": float(np.max(abs_smd)),
        "pct_abs_smd_lt_0_1": float(np.mean(abs_smd < 0.1)),
    }


def effective_sample_size(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    denom = np.sum(w ** 2)
    if denom <= 0:
        return np.nan
    return float((np.sum(w) ** 2) / denom)


def bootstrap_ci_for_ate(X: np.ndarray, t: np.ndarray, y: np.ndarray, n_bootstrap: int, seed: int) -> Tuple[Optional[float], Optional[float]]:
    if n_bootstrap <= 0:
        return None, None
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(t[idx])) < 2:
            continue
        try:
            vals.append(doubly_robust_ate(X[idx], t[idx], y[idx], random_state=seed).get("ate_dr", np.nan))
        except Exception:
            continue
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < max(10, n_bootstrap // 5):
        return None, None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def causal_diagnostics(
    processed_dir: Path,
    data_csv: Path,
    causal_csv: Path,
    top_n: int,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    if not causal_csv.exists():
        print(f"[WARN] {causal_csv.name} not found; skipping causal diagnostics.")
        return pd.DataFrame()

    df_effect = pd.read_csv(causal_csv)
    if df_effect.empty or "medication" not in df_effect.columns:
        return pd.DataFrame()
    if "ate_dr" in df_effect.columns:
        df_effect = df_effect.assign(_rank_abs_ate=df_effect["ate_dr"].abs()).sort_values("_rank_abs_ate", ascending=False)
    df_effect = df_effect.head(top_n)

    df = pd.read_csv(data_csv)
    X = np.load(processed_dir / "X_confounders.npy").astype(float)
    y = pd.to_numeric(df[cfg.OUTCOME_COL], errors="coerce").fillna(0).to_numpy(dtype=float)
    med_sets = df[cfg.TARGET_MED_COL].fillna("").astype(str).apply(split_codes)

    rows = []
    for _, row in df_effect.iterrows():
        med = row["medication"]
        t = med_sets.apply(lambda xs: int(med in xs)).to_numpy(dtype=int)
        if t.sum() < 10 or (1 - t).sum() < 10:
            continue
        try:
            ps_model = LogisticRegression(max_iter=1000, random_state=seed)
            ps_model.fit(X, t)
            p = ps_model.predict_proba(X)[:, 1]
            w = stabilized_ipw_binary(t, p)
            smd_before = weighted_smd_summary(X, t, None)
            smd_after = weighted_smd_summary(X, t, w)
            ci_low, ci_high = bootstrap_ci_for_ate(X, t, y, n_bootstrap, seed)
            rows.append({
                "medication": med,
                "ate_dr": row.get("ate_dr", np.nan),
                "ate_ci95_low": ci_low,
                "ate_ci95_high": ci_high,
                "treated_n": int(t.sum()),
                "control_n": int((1 - t).sum()),
                "propensity_min": float(np.min(p)),
                "propensity_p05": float(np.percentile(p, 5)),
                "propensity_p50": float(np.percentile(p, 50)),
                "propensity_p95": float(np.percentile(p, 95)),
                "propensity_max": float(np.max(p)),
                "ess_total": effective_sample_size(w),
                "ess_treated": effective_sample_size(w[t == 1]),
                "ess_control": effective_sample_size(w[t == 0]),
                "mean_abs_smd_before": smd_before["mean_abs_smd"],
                "max_abs_smd_before": smd_before["max_abs_smd"],
                "pct_abs_smd_lt_0_1_before": smd_before["pct_abs_smd_lt_0_1"],
                "mean_abs_smd_after": smd_after["mean_abs_smd"],
                "max_abs_smd_after": smd_after["max_abs_smd"],
                "pct_abs_smd_lt_0_1_after": smd_after["pct_abs_smd_lt_0_1"],
            })
        except Exception as exc:
            rows.append({"medication": med, "ate_dr": row.get("ate_dr", np.nan), "error": str(exc)})
    return pd.DataFrame(rows)


def per_medication_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray, med_vocab: List[str]) -> pd.DataFrame:
    rows = []
    for j, med in enumerate(med_vocab):
        yt = y_true[:, j].astype(int)
        yp = y_pred[:, j].astype(int)
        prob = y_prob[:, j]
        rows.append({
            "medication": med,
            "support": int(yt.sum()),
            "predicted_positive": int(yp.sum()),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
            "f1": f1_score(yt, yp, zero_division=0),
            "average_probability": float(np.mean(prob)),
        })
    return pd.DataFrame(rows).sort_values(["support", "f1"], ascending=[False, False])


def write_metric_tables(metrics: Dict, out_dir: Path) -> None:
    rows = []
    for group, value in metrics.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, (dict, list)):
                    continue
                rows.append({"metric_group": group, "metric": k, "value": _safe_float(v) if isinstance(v, (int, float, np.number)) else v})
    pd.DataFrame(rows).to_csv(out_dir / "evaluation_summary.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(_json_ready(metrics), f, ensure_ascii=False, indent=2)


def make_plots(out_dir: Path, y_true_med: np.ndarray, y_prob: np.ndarray, pred_df: pd.DataFrame, causal_diag: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable, skip plots: {exc}")
        return

    plot_dir = ensure_dir(out_dir / "evaluation_plots")

    # Regression: predicted vs true LOS and step duration.
    for true_col, pred_col, title, fname in [
        ("true_los", "predicted_los", "LOS prediction", "los_pred_vs_true.png"),
        ("true_step_duration", "predicted_step_duration", "Step duration prediction", "step_duration_pred_vs_true.png"),
    ]:
        if true_col in pred_df.columns and pred_col in pred_df.columns:
            x = pd.to_numeric(pred_df[true_col], errors="coerce")
            y = pd.to_numeric(pred_df[pred_col], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() > 0:
                plt.figure(figsize=(6, 5))
                plt.scatter(x[mask], y[mask], s=12, alpha=0.65)
                lo = float(min(x[mask].min(), y[mask].min()))
                hi = float(max(x[mask].max(), y[mask].max()))
                plt.plot([lo, hi], [lo, hi], linestyle="--")
                plt.xlabel("True")
                plt.ylabel("Predicted")
                plt.title(title)
                plt.tight_layout()
                plt.savefig(plot_dir / fname, dpi=160)
                plt.close()

    # Medication probability distribution for positive vs negative labels.
    if y_true_med.size and y_prob.size:
        pos_prob = y_prob[y_true_med == 1]
        neg_prob = y_prob[y_true_med == 0]
        plt.figure(figsize=(7, 5))
        if len(neg_prob):
            plt.hist(neg_prob, bins=40, alpha=0.6, label="negative labels")
        if len(pos_prob):
            plt.hist(pos_prob, bins=40, alpha=0.6, label="positive labels")
        plt.xlabel("Predicted medication probability")
        plt.ylabel("Count")
        plt.title("Medication probability separation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "med_probability_histogram.png", dpi=160)
        plt.close()

    # Causal diagnostics: balance before/after weighting.
    if causal_diag is not None and not causal_diag.empty and "mean_abs_smd_before" in causal_diag.columns:
        keep = causal_diag.dropna(subset=["mean_abs_smd_before", "mean_abs_smd_after"]).head(20)
        if not keep.empty:
            labels = keep["medication"].astype(str).tolist()
            x = np.arange(len(labels))
            width = 0.38
            plt.figure(figsize=(max(8, len(labels) * 0.45), 5))
            plt.bar(x - width / 2, keep["mean_abs_smd_before"], width, label="before IPW")
            plt.bar(x + width / 2, keep["mean_abs_smd_after"], width, label="after IPW")
            plt.axhline(0.1, linestyle="--")
            plt.xticks(x, labels, rotation=60, ha="right")
            plt.ylabel("Mean absolute SMD")
            plt.title("Covariate balance diagnostics")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / "causal_smd_balance.png", dpi=160)
            plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True, help="Step-level CSV, usually data/processed/visit_level_step.csv")
    ap.add_argument("--feature_dir", required=True, help="Directory containing X/Y npy files and vocab.pkl")
    ap.add_argument("--prediction_csv", required=True, help="prediction_preview.csv generated by train_step_cidgmed.py")
    ap.add_argument("--causal_csv", default=None, help="single_med_effect_on_los.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--subset", choices=["all", "train", "val"], default="all", help="Evaluate all rows or saved train/val split")
    ap.add_argument("--med_threshold", type=float, default=0.5)
    ap.add_argument("--top_k", type=str, default="3,5,10", help="Comma-separated K values for Recall@K/Precision@K")
    ap.add_argument("--causal_top_n", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=100, help="Bootstrap repetitions for ATE 95% CI. Set 0 to skip.")
    args = ap.parse_args()

    processed_dir = Path(args.feature_dir)
    out_dir = ensure_dir(args.out_dir)
    pred_df = pd.read_csv(args.prediction_csv)
    meta = load_pickle(processed_dir / "vocab.pkl")
    med_vocab = meta.get("med_vocab", [])

    indices = load_subset_indices(processed_dir, args.subset)
    y_true_med = np.load(processed_dir / "Y_med.npy").astype(int)
    y_prob = np.load(out_dir / "predicted_med_prob.npy") if (out_dir / "predicted_med_prob.npy").exists() else np.load(Path(args.prediction_csv).parent / "predicted_med_prob.npy")
    y_true_med = _select_subset(y_true_med, indices)
    y_prob = _select_subset(y_prob, indices)
    pred_eval_df = pred_df.iloc[indices].copy() if indices is not None else pred_df.copy()

    top_ks = [int(x.strip()) for x in args.top_k.split(",") if x.strip()]
    cls_metrics, y_pred_med = multilabel_metrics(y_true_med, y_prob, args.med_threshold, top_ks)

    los_metrics = regression_metrics(pred_eval_df["true_los"].to_numpy(), pred_eval_df["predicted_los"].to_numpy(), "los")
    duration_metrics = regression_metrics(
        pred_eval_df["true_step_duration"].to_numpy(),
        pred_eval_df["predicted_step_duration"].to_numpy(),
        "step_duration",
    )

    per_med_df = per_medication_metrics(y_true_med, y_prob, y_pred_med, med_vocab)
    per_med_df.to_csv(out_dir / f"per_medication_metrics_{args.subset}.csv", index=False, encoding="utf-8-sig")

    causal_path = Path(args.causal_csv) if args.causal_csv else (out_dir / "single_med_effect_on_los.csv")
    causal_diag = causal_diagnostics(
        processed_dir=processed_dir,
        data_csv=Path(args.data_csv),
        causal_csv=causal_path,
        top_n=args.causal_top_n,
        n_bootstrap=args.bootstrap,
        seed=cfg.RANDOM_STATE,
    )
    if not causal_diag.empty:
        causal_diag.to_csv(out_dir / "causal_diagnostics.csv", index=False, encoding="utf-8-sig")

    metrics = {
        "evaluation_config": {
            "subset": args.subset,
            "med_threshold": args.med_threshold,
            "top_k": top_ks,
            "causal_top_n": args.causal_top_n,
            "bootstrap": args.bootstrap,
        },
        "medication_prediction": cls_metrics,
        "los_regression": los_metrics,
        "step_duration_regression": duration_metrics,
    }
    if not causal_diag.empty:
        metrics["causal_diagnostics_mean"] = {
            "n_medications_evaluated": int(len(causal_diag)),
            "mean_abs_smd_before": causal_diag["mean_abs_smd_before"].mean(skipna=True),
            "mean_abs_smd_after": causal_diag["mean_abs_smd_after"].mean(skipna=True),
            "mean_ess_total": causal_diag["ess_total"].mean(skipna=True),
            "median_ess_total": causal_diag["ess_total"].median(skipna=True),
        }

    write_metric_tables(metrics, out_dir)
    make_plots(out_dir, y_true_med, y_prob, pred_eval_df, causal_diag)

    print("Saved evaluation outputs:")
    print(f"- {out_dir / 'evaluation_metrics.json'}")
    print(f"- {out_dir / 'evaluation_summary.csv'}")
    print(f"- {out_dir / ('per_medication_metrics_' + args.subset + '.csv')}")
    if not causal_diag.empty:
        print(f"- {out_dir / 'causal_diagnostics.csv'}")
    print(f"- {out_dir / 'evaluation_plots'}")


if __name__ == "__main__":
    main()
