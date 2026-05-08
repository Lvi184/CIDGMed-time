from pathlib import Path
import json
import joblib


def ensure_dir(path_like):
    path = Path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_pickle(obj, path_like):
    path = Path(path_like)
    ensure_dir(path.parent)
    joblib.dump(obj, path)


def load_pickle(path_like):
    return joblib.load(path_like)


def save_json(obj, path_like, indent=2):
    path = Path(path_like)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
