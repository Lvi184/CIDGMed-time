
import json
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


class StepMultiTaskDataset(Dataset):
    def __init__(self, processed_dir, use_scaled_time=True):
        processed_dir = Path(processed_dir)
        self.processed_dir = processed_dir

        # 读取元信息
        with open(processed_dir / "meta_columns.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.clinical_feature_cols = self.meta["clinical_feature_cols"]
        self.prev_med_cols = self.meta["prev_med_cols"]
        self.time_cols = self.meta["time_cols"]
        self.target_cols = self.meta["target_cols"]

        # 读取数据
        self.X = pd.read_csv(processed_dir / "X_multilabel.csv")
        self.Y_med = pd.read_csv(processed_dir / "Y_multilabel_drugs.csv")
        
        if use_scaled_time:
            self.y_time = pd.read_csv(processed_dir / "y_step_time_scaled.csv")
        else:
            self.y_time = pd.read_csv(processed_dir / "y_step_time.csv")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        row = self.X.iloc[idx]

        x_clinical = torch.tensor(row[self.clinical_feature_cols].values, dtype=torch.float32)
        x_prev_med = torch.tensor(row[self.prev_med_cols].values, dtype=torch.float32)
        x_time_ctx = torch.tensor(row[self.time_cols].values, dtype=torch.float32)
        
        y_med = torch.tensor(self.Y_med.iloc[idx].values, dtype=torch.float32)
        y_time = torch.tensor(float(self.y_time.iloc[idx]), dtype=torch.float32)

        return {
            "x_clinical": x_clinical,
            "x_prev_med": x_prev_med,
            "x_time_ctx": x_time_ctx,
            "y_med": y_med,
            "y_time": y_time,
        }


def build_dataset_from_dir(processed_dir, use_scaled_time=True):
    return StepMultiTaskDataset(processed_dir, use_scaled_time)


def get_dataloaders(dataset, train_idx, val_idx, batch_size=32, num_workers=0):
    from torch.utils.data import Subset

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

