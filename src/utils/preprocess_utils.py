import re
from typing import List, Dict
import numpy as np
import pandas as pd

_SPLIT_RE = re.compile(r"[|,;，；/\\]+")
_STEP_RE = re.compile(r"\s*-\s*")
_TOKEN_RE = re.compile(r"\*")
_COUNT_PREFIX_RE = re.compile(r"^\s*\d+\s*x", flags=re.IGNORECASE)


def is_missing(value) -> bool:
    if value is None:
        return True
    if pd.isna(value):
        return True
    text = str(value).strip().lower()
    return text in {"", "nan", "none", "null"}


def split_codes(value) -> List[str]:
    if is_missing(value):
        return []
    text = str(value).strip()
    return [x.strip() for x in _SPLIT_RE.split(text) if x.strip()]


def split_steps(value) -> List[str]:
    if is_missing(value):
        return []
    text = str(value).strip()
    return [x.strip() for x in _STEP_RE.split(text) if x.strip()]


def parse_durations(value) -> List[float]:
    if is_missing(value):
        return []
    parts = [x.strip() for x in str(value).split("+") if x.strip()]
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            out.append(np.nan)
    return out


def normalize_med_token(token: str) -> str:
    token = str(token).strip()
    token = _COUNT_PREFIX_RE.sub("", token)
    return token.strip()


def parse_regimen_segment(segment) -> List[str]:
    if is_missing(segment):
        return []
    tokens = [normalize_med_token(tok) for tok in _TOKEN_RE.split(str(segment))]
    tokens = [t for t in tokens if t]
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def safe_step_value(values: List, idx: int, default=None):
    return values[idx] if idx < len(values) else default


def explode_path_row(row: pd.Series, id_col: str, seq_col: str, time_col: str, path_col: str) -> List[Dict]:
    seq_steps = split_steps(row.get(seq_col, ""))
    time_steps = parse_durations(row.get(time_col, ""))
    path_steps = split_steps(row.get(path_col, ""))
    n_steps = max(len(seq_steps), len(time_steps), len(path_steps))
    if n_steps == 0:
        n_steps = 1

    out = []
    prev_meds = []
    prev_set = set()
    prev_step_meds = []

    for step_idx in range(n_steps):
        raw_seq = safe_step_value(seq_steps, step_idx, "") or ""
        raw_path = safe_step_value(path_steps, step_idx, raw_seq) or raw_seq
        duration = safe_step_value(time_steps, step_idx, np.nan)
        cur_meds = parse_regimen_segment(raw_path if raw_path else raw_seq)
        cur_set = set(cur_meds)
        added = [m for m in cur_meds if m not in prev_set]
        removed = [m for m in prev_step_meds if m not in cur_set]

        base = row.to_dict()
        base["step_idx"] = step_idx
        base["step_duration"] = duration
        base["step_regimen_raw"] = raw_path
        base["target_med_codes"] = "|".join(cur_meds)
        base["prev_med_codes"] = "|".join(prev_meds)
        base["added_med_codes"] = "|".join(added)
        base["removed_med_codes"] = "|".join(removed)
        out.append(base)

        for med in cur_meds:
            if med not in prev_set:
                prev_set.add(med)
                prev_meds.append(med)
        prev_step_meds = cur_meds

    return out


def fill_and_cast_categorical(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(str)
            out[c] = out[c].replace({"nan": "UNK", "None": "UNK", "": "UNK"})
    return out


def fill_and_cast_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            fill_val = out[c].median() if out[c].notna().any() else 0.0
            out[c] = out[c].fillna(fill_val)
    return out
