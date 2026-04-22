
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, jaccard_score, average_precision_score, mean_absolute_error, mean_squared_error
import pandas as pd

from src.data_loader_step import build_dataset_from_dir
from src.modules.step_cidgmed import StepCIDGMed


def compute_loss(med_logits, time_pred, y_med, y_time, alpha=0.5):
    loss_med = F.binary_cross_entropy_with_logits(med_logits, y_med)
    loss_time = F.mse_loss(time_pred, y_time)
    return loss_med + alpha * loss_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_scaled_time", action="store_true")
    parser.add_argument("--use_causal_bias", action="store_true")
    parser.add_argument("--causal_matrix_path", type=str, default=None)

    args = parser.parse_args()

    dataset = build_dataset_from_dir(args.processed_dir, args.use_scaled_time)

    groups = pd.read_csv(f"{args.processed_dir}/groups_patient_id.csv").iloc[:, 0].values

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(np.arange(len(dataset)), groups=groups))

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size)

    sample = dataset[0]

    causal_matrix = None
    if args.use_causal_bias and args.causal_matrix_path:
        causal_matrix = np.load(args.causal_matrix_path)

    model = StepCIDGMed(
        clinical_dim=sample["x_clinical"].shape[0],
        prev_med_dim=sample["x_prev_med"].shape[0],
        time_ctx_dim=sample["x_time_ctx"].shape[0],
        num_drugs=sample["y_med"].shape[0],
        use_causal_bias=args.use_causal_bias,
        causal_effect_matrix=causal_matrix,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            med_logits, time_pred = model(
                batch["x_clinical"],
                batch["x_prev_med"],
                batch["x_time_ctx"],
            )

            loss = compute_loss(med_logits, time_pred, batch["y_med"], batch["y_time"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} done")


if __name__ == "__main__":
    main()

