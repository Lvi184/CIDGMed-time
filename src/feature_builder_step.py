import argparse
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from configs import data_config as cfg
from src.utils.io_utils import ensure_dir, save_pickle
from src.utils.preprocess_utils import split_codes, fill_and_cast_categorical, fill_and_cast_numeric


def build_vocab(series, min_support):
    counter = Counter()
    for val in series:
        counter.update(split_codes(val))
    return sorted([k for k, v in counter.items() if v >= min_support])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    df = pd.read_csv(args.input)

    diag_vocab = build_vocab(df.get("diag_codes", pd.Series(dtype=str)), cfg.MIN_SUPPORT_DIAG)
    proc_vocab = build_vocab(df.get("proc_codes", pd.Series(dtype=str)), cfg.MIN_SUPPORT_PROC)
    med_vocab = build_vocab(df.get(cfg.TARGET_MED_COL, pd.Series(dtype=str)), cfg.MIN_SUPPORT_MED)

    mlb_diag = MultiLabelBinarizer(classes=diag_vocab)
    mlb_proc = MultiLabelBinarizer(classes=proc_vocab)
    mlb_prev = MultiLabelBinarizer(classes=med_vocab)
    mlb_med = MultiLabelBinarizer(classes=med_vocab)

    X_diag = mlb_diag.fit_transform(df.get("diag_codes", "").apply(split_codes)) if diag_vocab else np.zeros((len(df), 0), dtype=np.float32)
    X_proc = mlb_proc.fit_transform(df.get("proc_codes", "").apply(split_codes)) if proc_vocab else np.zeros((len(df), 0), dtype=np.float32)
    X_prev_med = mlb_prev.fit_transform(df.get(cfg.PREV_MED_COL, "").apply(split_codes)) if med_vocab else np.zeros((len(df), 0), dtype=np.float32)
    Y_med = mlb_med.fit_transform(df.get(cfg.TARGET_MED_COL, "").apply(split_codes)) if med_vocab else np.zeros((len(df), 0), dtype=np.float32)

    cat_cols = [c for c in cfg.BINARY_OR_CATEGORICAL_COLS if c in df.columns]
    num_cols = [c for c in cfg.NUMERIC_COLS if c in df.columns and c != cfg.OUTCOME_COL]

    extra_numeric = []
    if cfg.STEP_INDEX_COL in df.columns:
        extra_numeric.append(cfg.STEP_INDEX_COL)
    if cfg.STEP_DURATION_COL in df.columns:
        extra_numeric.append(cfg.STEP_DURATION_COL)
    num_cols = list(dict.fromkeys(num_cols + extra_numeric))

    df_cat = fill_and_cast_categorical(df[cat_cols], cat_cols) if cat_cols else pd.DataFrame(index=df.index)
    df_num = fill_and_cast_numeric(df[num_cols], num_cols) if num_cols else pd.DataFrame(index=df.index)

    cat_records = df_cat.to_dict(orient="records") if not df_cat.empty else [{} for _ in range(len(df))]
    cat_vec = DictVectorizer(sparse=False)
    X_cat = cat_vec.fit_transform(cat_records) if cat_cols else np.zeros((len(df), 0), dtype=np.float32)
    X_num = df_num.to_numpy(dtype=np.float32) if not df_num.empty else np.zeros((len(df), 0), dtype=np.float32)

    X_demo = np.concatenate([X_cat, X_num], axis=1).astype(np.float32)
    X_confounders = np.concatenate([X_demo, X_diag, X_proc, X_prev_med], axis=1).astype(np.float32)
    Y_time = pd.to_numeric(df[cfg.OUTCOME_COL], errors="coerce").fillna(0).to_numpy(dtype=np.float32).reshape(-1, 1)
    Y_step_duration = pd.to_numeric(df.get(cfg.STEP_DURATION_COL, 0), errors="coerce").fillna(0).to_numpy(dtype=np.float32).reshape(-1, 1)

    np.save(out_dir / "X_diag.npy", X_diag.astype(np.float32))
    np.save(out_dir / "X_proc.npy", X_proc.astype(np.float32))
    np.save(out_dir / "X_prev_med.npy", X_prev_med.astype(np.float32))
    np.save(out_dir / "X_demo.npy", X_demo.astype(np.float32))
    np.save(out_dir / "X_confounders.npy", X_confounders.astype(np.float32))
    np.save(out_dir / "Y_med.npy", Y_med.astype(np.float32))
    np.save(out_dir / "Y_time.npy", Y_time.astype(np.float32))
    np.save(out_dir / "Y_step_duration.npy", Y_step_duration.astype(np.float32))

    save_pickle({
        "diag_vocab": diag_vocab,
        "proc_vocab": proc_vocab,
        "med_vocab": med_vocab,
        "categorical_feature_names": list(cat_vec.get_feature_names_out()) if cat_cols else [],
        "numeric_feature_names": num_cols,
    }, out_dir / "vocab.pkl")

    print("Saved features to", out_dir)
    print("Shapes:")
    print("  X_confounders", X_confounders.shape)
    print("  Y_med", Y_med.shape)
    print("  Y_time", Y_time.shape)
    print("  Y_step_duration", Y_step_duration.shape)


if __name__ == "__main__":
    main()
