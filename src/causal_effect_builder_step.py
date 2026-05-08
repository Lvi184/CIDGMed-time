import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from configs import data_config as cfg
from src.utils.io_utils import ensure_dir, load_pickle
from src.utils.preprocess_utils import split_codes
from src.utils.causal_utils import doubly_robust_ate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--feature_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    df = pd.read_csv(args.data_csv)
    X = np.load(Path(args.feature_dir) / "X_confounders.npy")
    meta = load_pickle(Path(args.feature_dir) / "vocab.pkl")
    med_vocab = meta.get("med_vocab", [])

    y = pd.to_numeric(df[cfg.OUTCOME_COL], errors="coerce").fillna(0).to_numpy(dtype=float)

    results = []
    target_meds_series = df[cfg.TARGET_MED_COL].fillna("").astype(str).apply(split_codes)
    for med in med_vocab:
        t = target_meds_series.apply(lambda xs: int(med in xs)).to_numpy(dtype=int)
        treated_n = int(t.sum())
        control_n = int(len(t) - treated_n)
        if treated_n < 10 or control_n < 10:
            continue
        try:
            stat = doubly_robust_ate(X, t, y, random_state=cfg.RANDOM_STATE)
            stat["medication"] = med
            results.append(stat)
        except Exception as e:
            results.append({
                "medication": med,
                "ate_dr": np.nan,
                "treated_n": treated_n,
                "control_n": control_n,
                "ps_logloss": np.nan,
                "ipw_treated_mean": np.nan,
                "ipw_control_mean": np.nan,
                "error": str(e),
            })

    if results:
        effect_df = pd.DataFrame(results).sort_values("ate_dr", ascending=False, na_position="last")
    else:
        effect_df = pd.DataFrame(columns=["medication","ate_dr","treated_n","control_n","ps_logloss","ipw_treated_mean","ipw_control_mean"])
    effect_df.to_csv(out_dir / "single_med_effect_on_los.csv", index=False, encoding="utf-8-sig")

    duration_df = (
        df[[cfg.STEP_DURATION_COL, cfg.OUTCOME_COL]]
        .copy()
        .rename(columns={cfg.STEP_DURATION_COL: "step_duration", cfg.OUTCOME_COL: "outcome"})
    )
    duration_df["step_duration"] = pd.to_numeric(duration_df["step_duration"], errors="coerce")
    duration_df["outcome"] = pd.to_numeric(duration_df["outcome"], errors="coerce")
    duration_stats = duration_df.groupby("step_duration", dropna=True)["outcome"].agg(["count", "mean", "median"]).reset_index()
    duration_stats.to_csv(out_dir / "step_duration_effect_on_los.csv", index=False, encoding="utf-8-sig")

    regimen_freq = (
        df[cfg.TARGET_MED_COL]
        .fillna("")
        .astype(str)
        .value_counts()
        .rename_axis("regimen")
        .reset_index(name="count")
    )
    regimen_freq.to_csv(out_dir / "regimen_frequency.csv", index=False, encoding="utf-8-sig")

    print(f"Saved causal outputs to {out_dir}")


if __name__ == "__main__":
    main()
