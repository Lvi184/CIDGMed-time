
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
)

from data_loader_step import build_dataset_from_dir, get_dataloaders
from step_cidgmed import StepCIDGMed


def compute_loss(med_logits, time_pred, y_med, y_time, alpha=0.5):
    loss_med = F.binary_cross_entropy_with_logits(med_logits, y_med)
    loss_time = F.mse_loss(time_pred, y_time)
    loss = loss_med + alpha * loss_time
    return loss, loss_med, loss_time


def evaluate(model, loader, device, time_scaler=None, threshold=0.5):
    model.eval()
    all_med_probs = []
    all_med_true = []
    all_time_pred = []
    all_time_true = []

    with torch.no_grad():
        for batch in loader:
            x_clinical = batch["x_clinical"].to(device)
            x_prev_med = batch["x_prev_med"].to(device)
            x_time_ctx = batch["x_time_ctx"].to(device)
            y_med = batch["y_med"].to(device)
            y_time = batch["y_time"].to(device)

            med_logits, time_pred = model(x_clinical, x_prev_med, x_time_ctx)
            med_probs = torch.sigmoid(med_logits)

            all_med_probs.append(med_probs.cpu().numpy())
            all_med_true.append(y_med.cpu().numpy())
            all_time_pred.append(time_pred.cpu().numpy())
            all_time_true.append(y_time.cpu().numpy())

    all_med_probs = np.vstack(all_med_probs)
    all_med_true = np.vstack(all_med_true)
    all_time_pred = np.concatenate(all_time_pred)
    all_time_true = np.concatenate(all_time_true)

    # 反标准化时间（如果有）
    if time_scaler is not None:
        all_time_pred = time_scaler.inverse_transform(all_time_pred.reshape(-1, 1)).reshape(-1)
        all_time_true = time_scaler.inverse_transform(all_time_true.reshape(-1, 1)).reshape(-1)

    # 药物多标签指标
    med_pred_bin = (all_med_probs > threshold).astype(int)

    precision_micro = precision_score(all_med_true, med_pred_bin, average="micro", zero_division=0)
    recall_micro = recall_score(all_med_true, med_pred_bin, average="micro", zero_division=0)
    micro_f1 = f1_score(all_med_true, med_pred_bin, average="micro", zero_division=0)
    macro_f1 = f1_score(all_med_true, med_pred_bin, average="macro", zero_division=0)
    jaccard_samples = jaccard_score(all_med_true, med_pred_bin, average="samples", zero_division=0)
    
    try:
        prauc_macro = average_precision_score(all_med_true, all_med_probs, average="macro")
    except:
        prauc_macro = np.nan

    # 时间回归指标
    mae = mean_absolute_error(all_time_true, all_time_pred)
    rmse = np.sqrt(mean_squared_error(all_time_true, all_time_pred))

    return {
        "precision": precision_micro,
        "recall": recall_micro,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "jaccard_samples": jaccard_samples,
        "prauc_macro": prauc_macro,
        "mae": mae,
        "rmse": rmse,
    }


def train_one_epoch(model, loader, optimizer, device, alpha=0.5):
    model.train()
    total_loss = 0.0
    total_loss_med = 0.0
    total_loss_time = 0.0

    for batch in loader:
        optimizer.zero_grad()

        x_clinical = batch["x_clinical"].to(device)
        x_prev_med = batch["x_prev_med"].to(device)
        x_time_ctx = batch["x_time_ctx"].to(device)
        y_med = batch["y_med"].to(device)
        y_time = batch["y_time"].to(device)

        med_logits, time_pred = model(x_clinical, x_prev_med, x_time_ctx)
        loss, loss_med, loss_time = compute_loss(med_logits, time_pred, y_med, y_time, alpha=alpha)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch["x_clinical"])
        total_loss_med += loss_med.item() * len(batch["x_clinical"])
        total_loss_time += loss_time.item() * len(batch["x_clinical"])

    num_samples = len(loader.dataset)
    return {
        "loss": total_loss / num_samples,
        "loss_med": total_loss_med / num_samples,
        "loss_time": total_loss_time / num_samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default="processed_stepcidgmed")
    parser.add_argument("--output_dir", type=str, default="saved/stepcidgmed")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.3, help="time loss weight")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_scaled_time", action="store_true", default=True)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    processed_dir = Path(args.processed_dir)
    dataset = build_dataset_from_dir(processed_dir, use_scaled_time=args.use_scaled_time)
    groups = np.loadtxt(processed_dir / "groups_patient_id.csv", delimiter=",", dtype=str, skiprows=1)

    # 按病人分组划分
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(np.arange(len(dataset)), groups=groups))
    train_loader, val_loader = get_dataloaders(dataset, train_idx, val_idx, batch_size=args.batch_size)

    # 加载时间 scaler（用于反标准化）
    import joblib
    time_scaler = None
    if args.use_scaled_time:
        time_scaler_path = processed_dir / "y_time_scaler.joblib"
        if time_scaler_path.exists():
            time_scaler = joblib.load(time_scaler_path)

    # 初始化模型
    sample = dataset[0]
    model = StepCIDGMed(
        clinical_dim=sample["x_clinical"].shape[0],
        prev_med_dim=sample["x_prev_med"].shape[0],
        time_ctx_dim=sample["x_time_ctx"].shape[0],
        num_drugs=sample["y_med"].shape[0],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # 训练循环
    best_metric = -1.0
    best_epoch = 0
    best_state = None
    metrics_history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, alpha=args.alpha)
        val_metrics = evaluate(model, val_loader, device, time_scaler=time_scaler)

        print(f"Train loss: {train_metrics['loss']:.4f} (med: {train_metrics['loss_med']:.4f}, time: {train_metrics['loss_time']:.4f})")
        print(f"Val micro-F1: {val_metrics['micro_f1']:.4f}, recall: {val_metrics['recall']:.4f}, MAE: {val_metrics['mae']:.4f}")

        # 保存最佳模型（按 micro-F1）
        score = val_metrics["micro_f1"]
        if score > best_metric:
            best_metric = score
            best_epoch = epoch
            best_state = model.state_dict().copy()
            torch.save(best_state, output_dir / "best_model.pt")
            print(f"New best model at epoch {epoch} (micro-F1: {best_metric:.4f})")

        # 记录指标
        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        metrics_history.append(epoch_record)

    # 保存最终指标和配置
    print(f"\nTraining complete! Best epoch {best_epoch}, micro-F1: {best_metric:.4f}")
    with open(output_dir / "metrics_history.json", "w", encoding="utf-8") as f:
        json.dump(metrics_history, f, indent=2)

    config = vars(args)
    config["best_micro_f1"] = best_metric
    config["best_epoch"] = best_epoch
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()

