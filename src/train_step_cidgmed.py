
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    f1_score, 
    jaccard_score, 
    average_precision_score, 
    mean_absolute_error, 
    mean_squared_error,
)
import pandas as pd
from pathlib import Path

from src.data_loader_step import build_dataset_from_dir
from src.modules.CIDGMed_step import StepCIDGMedFull


def compute_loss(med_logits, time_pred, y_med, y_time, alpha=0.5):
    loss_med = F.binary_cross_entropy_with_logits(med_logits, y_med)
    loss_time = F.mse_loss(time_pred, y_time)
    return loss_med + alpha * loss_time


def extract_causal_features(x_batch, meta):
    """Extract all features used for causal effect matrix"""
    diagnostic_like_features = [
        'out_diagnosis_code',
        'severity', 
        'first_episode',
        'psychiatric_comorbidity',
        'endocrine_comorbidity',
        'nervous_comorbidity',
        'digestive_comorbidity',
        'circulatory_comorbidity',
        'respiratory_comorbidity',
        'cancer_comorbidity',
    ]
    
    procedure_like_features = [
        'surgery_NO',
        'operation_NO',
        'history_surgery',
    ]
    
    all_causal_features = diagnostic_like_features + procedure_like_features
    
    features = []
    for feat in all_causal_features:
        if feat in meta['all_x_cols']:
            col_idx = meta['all_x_cols'].index(feat)
            features.append(x_batch[:, col_idx].unsqueeze(1))
        else:
            features.append(torch.zeros((x_batch.shape[0], 1), device=x_batch.device))
    
    return torch.cat(features, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_scaled_time", action="store_true")
    parser.add_argument("--use_causal_bias", action="store_true")

    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    dataset = build_dataset_from_dir(processed_dir, args.use_scaled_time)

    groups = pd.read_csv(processed_dir / "groups_patient_id.csv", header=None).values.flatten()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(np.arange(len(dataset)), groups=groups))

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size)

    sample = dataset[0]
    meta = json.load(open(processed_dir / "meta_columns.json", "r"))
    
    # Load causal effect matrix
    causal_effect_mat = None
    if args.use_causal_bias:
        causal_effect_path = processed_dir / "causal_effect_matrix.npy"
        if causal_effect_path.exists():
            causal_effect_mat = torch.tensor(np.load(causal_effect_path), dtype=torch.float32)

    model = StepCIDGMedFull(
        clinical_dim=sample["x_clinical"].shape[0],
        prev_med_dim=sample["x_prev_med"].shape[0],
        time_ctx_dim=sample["x_time_ctx"].shape[0],
        num_med=sample["y_med"].shape[0],
        num_diag=10,
        num_proc=3,
        use_causal_bias=args.use_causal_bias,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            # Move data to device
            x_clinical = batch["x_clinical"]
            x_prev_med = batch["x_prev_med"]
            x_time_ctx = batch["x_time_ctx"]
            y_med = batch["y_med"]
            y_time = batch["y_time"]
            
            # Get causal features
            causal_features = extract_causal_features(x_clinical, meta)
            
            if args.use_causal_bias and causal_effect_mat is not None:
                causal_effect_mat = causal_effect_mat.to(x_clinical.device)
            
            med_logits, time_pred = model(
                x_clinical,
                x_prev_med,
                x_time_ctx,
                causal_features,
                causal_effect_mat,
            )

            loss = compute_loss(med_logits, time_pred, y_med, y_time)
            total_loss += loss.item() * x_clinical.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch} done, avg train loss: {avg_train_loss:.4f}")


if __name__ == "__main__":
    main()
