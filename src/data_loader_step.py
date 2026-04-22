
import json
import pandas as pd
import torch
from torch.utils.data import Dataset


class StepMultilabelDataset(Dataset):
    def __init__(self, x_file, y_med_file, y_time_file, meta_file):
        self.X = pd.read_csv(x_file)
        self.Y_med = pd.read_csv(y_med_file)
        self.y_time = pd.read_csv(y_time_file)

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.clinical_feature_cols = meta["clinical_feature_cols"]
        self.prev_med_cols = meta["prev_med_cols"]
        self.time_cols = meta["time_cols"]

        # ✅ 长度一致性检查（重要）
        assert len(self.X) == len(self.Y_med) == len(self.y_time), \
            f"Length mismatch: X={len(self.X)}, Y_med={len(self.Y_med)}, y_time={len(self.y_time)}"

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        row = self.X.iloc[idx]

        x_clinical = torch.tensor(row[self.clinical_feature_cols].values, dtype=torch.float32)
        x_prev_med = torch.tensor(row[self.prev_med_cols].values, dtype=torch.float32)
        x_time_ctx = torch.tensor(row[self.time_cols].values, dtype=torch.float32)

        y_med = torch.tensor(self.Y_med.iloc[idx].values, dtype=torch.float32)

        # ✅ 修复关键 bug
        y_time = torch.tensor(float(self.y_time.iloc[idx, 0]), dtype=torch.float32)

        return {
            "x_clinical": x_clinical,
            "x_prev_med": x_prev_med,
            "x_time_ctx": x_time_ctx,
            "y_med": y_med,
            "y_time": y_time,
        }


def build_dataset_from_dir(processed_dir, use_scaled_time=True):
    x_file = f"{processed_dir}/X_multilabel.csv"
    y_med_file = f"{processed_dir}/Y_multilabel_drugs.csv"

    if use_scaled_time:
        y_time_file = f"{processed_dir}/y_step_time_scaled.csv"
    else:
        y_time_file = f"{processed_dir}/y_step_time.csv"

    meta_file = f"{processed_dir}/meta_columns.json"

    return StepMultilabelDataset(x_file, y_med_file, y_time_file, meta_file)

