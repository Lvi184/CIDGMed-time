import argparse
from pathlib import Path
import pandas as pd
from configs import data_config as cfg
from src.utils.io_utils import ensure_dir
from src.utils.preprocess_utils import explode_path_row, split_codes


def build_step_table(df: pd.DataFrame) -> pd.DataFrame:
    if cfg.ID_COL not in df.columns:
        raise ValueError(f"Missing required column: {cfg.ID_COL}")

    rows = []
    for _, row in df.iterrows():
        exploded = explode_path_row(
            row,
            id_col=cfg.ID_COL,
            seq_col=cfg.STEP_COL,
            time_col=cfg.TIME_COL,
            path_col=cfg.MEDICATION_SOURCE_COL,
        )
        for rec in exploded:
            rec["diag_codes"] = "|".join(split_codes(row.get(cfg.DIAGNOSIS_COL, "")))
            rec["proc_codes"] = "|".join(split_codes(row.get(cfg.PROCEDURE_COL, "")))
            rows.append(rec)

    out = pd.DataFrame(rows)
    out[cfg.STEP_INDEX_COL] = pd.to_numeric(out[cfg.STEP_INDEX_COL], errors="coerce").fillna(0).astype(int)
    if cfg.STEP_DURATION_COL in out.columns:
        out[cfg.STEP_DURATION_COL] = pd.to_numeric(out[cfg.STEP_DURATION_COL], errors="coerce")

    keep_cols = [
        cfg.ID_COL,
        cfg.STEP_COL,
        cfg.TIME_COL,
        cfg.MEDICATION_SOURCE_COL,
        cfg.STEP_INDEX_COL,
        cfg.STEP_DURATION_COL,
        cfg.RAW_REGIMEN_COL,
        cfg.TARGET_MED_COL,
        cfg.PREV_MED_COL,
        cfg.ADDED_MED_COL,
        cfg.REMOVED_MED_COL,
        cfg.TIME_LABEL_COL,
        "diag_codes",
        "proc_codes",
    ]
    keep_cols += cfg.DEMOGRAPHIC_COLS
    keep_cols += cfg.PHYSICAL_EXAM_COLS
    keep_cols += cfg.READMISSION_COLS
    keep_cols += cfg.DISEASE_COLS
    keep_cols += cfg.COMORBIDITY_COLS
    keep_cols += cfg.HISTORY_COLS
    keep_cols += cfg.SYMPTOM_COLS
    keep_cols += cfg.LAB_VALUE_COLS
    keep_cols += cfg.LAB_LEVEL_COLS
    keep_cols += [cfg.SURGERY_COL]

    keep_cols = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols].copy()
    out = out.sort_values([cfg.ID_COL, cfg.STEP_INDEX_COL]).reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    out = build_step_table(df)
    ensure_dir(Path(args.output).parent)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved step table to {args.output}, shape={out.shape}")


if __name__ == "__main__":
    main()
