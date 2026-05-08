import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from configs import data_config as cfg
from src.models.step_cidgmed import StepCIDGMedNet
from src.utils.io_utils import ensure_dir, load_pickle


def evaluate(model, loader, med_loss_fn, time_loss_fn, duration_loss_fn, device, duration_weight: float):
    model.eval()
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for xb, y_med, y_time, y_duration in loader:
            xb = xb.to(device)
            y_med = y_med.to(device)
            y_time = y_time.to(device)
            y_duration = y_duration.to(device)
            med_logits, time_pred, duration_pred = model(xb)
            med_loss = med_loss_fn(med_logits, y_med)
            time_loss = time_loss_fn(time_pred, y_time)
            duration_loss = duration_loss_fn(duration_pred, y_duration)
            loss = med_loss + time_loss + duration_weight * duration_loss
            batch_n = xb.size(0)
            total_loss += loss.item() * batch_n
            total_n += batch_n
    return total_loss / max(total_n, 1)


def decode_regimen(prob_row, med_vocab, threshold: float, top_k: int):
    if len(med_vocab) == 0:
        return "", ""
    order = np.argsort(prob_row)[::-1]
    selected = [i for i in order if prob_row[i] >= threshold]
    if not selected and top_k > 0:
        selected = list(order[:top_k])
    elif top_k > 0:
        selected = selected[:top_k]
    meds = [med_vocab[i] for i in selected]
    probs = [f"{med_vocab[i]}:{prob_row[i]:.4f}" for i in selected]
    return "|".join(meds), ";".join(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--feature_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--duration_weight", type=float, default=0.5)
    ap.add_argument("--med_threshold", type=float, default=0.5)
    ap.add_argument("--top_k_meds", type=int, default=5)
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    feature_dir = Path(args.feature_dir)
    data_csv = Path(args.data_csv)

    X = np.load(feature_dir / "X_confounders.npy").astype(np.float32)
    Y_med = np.load(feature_dir / "Y_med.npy").astype(np.float32)
    Y_time = np.load(feature_dir / "Y_time.npy").astype(np.float32)
    duration_path = feature_dir / "Y_step_duration.npy"
    if duration_path.exists():
        Y_duration = np.load(duration_path).astype(np.float32)
    else:
        df_tmp = pd.read_csv(data_csv)
        Y_duration = pd.to_numeric(df_tmp.get(cfg.STEP_DURATION_COL, 0), errors="coerce").fillna(0).to_numpy(dtype=np.float32).reshape(-1, 1)
    df = pd.read_csv(data_csv)
    meta = load_pickle(feature_dir / "vocab.pkl")
    med_vocab = meta.get("med_vocab", [])

    X_t = torch.tensor(X, dtype=torch.float32)
    Y_med_t = torch.tensor(Y_med, dtype=torch.float32)
    Y_time_t = torch.tensor(Y_time, dtype=torch.float32)
    Y_duration_t = torch.tensor(Y_duration, dtype=torch.float32)

    ds = TensorDataset(X_t, Y_med_t, Y_time_t, Y_duration_t)
    n_total = len(ds)
    n_val = max(1, int(0.2 * n_total)) if n_total > 1 else 0
    n_train = n_total - n_val
    if n_val > 0:
        train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.RANDOM_STATE))
        np.save(out_dir / "train_indices.npy", np.asarray(train_ds.indices, dtype=np.int64))
        np.save(out_dir / "val_indices.npy", np.asarray(val_ds.indices, dtype=np.int64))
    else:
        train_ds, val_ds = ds, ds
        np.save(out_dir / "train_indices.npy", np.arange(n_total, dtype=np.int64))
        np.save(out_dir / "val_indices.npy", np.arange(n_total, dtype=np.int64))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StepCIDGMedNet(input_dim=X.shape[1], med_dim=Y_med.shape[1], hidden_dim=args.hidden_dim).to(device)

    med_loss_fn = torch.nn.BCEWithLogitsLoss()
    time_loss_fn = torch.nn.MSELoss()
    duration_loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    history = []
    best_path = out_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for xb, y_med, y_time, y_duration in train_loader:
            xb = xb.to(device)
            y_med = y_med.to(device)
            y_time = y_time.to(device)
            y_duration = y_duration.to(device)

            optimizer.zero_grad()
            med_logits, time_pred, duration_pred = model(xb)
            med_loss = med_loss_fn(med_logits, y_med)
            time_loss = time_loss_fn(time_pred, y_time)
            duration_loss = duration_loss_fn(duration_pred, y_duration)
            loss = med_loss + time_loss + args.duration_weight * duration_loss
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            running += loss.item() * bs
            seen += bs

        train_loss = running / max(seen, 1)
        val_loss = evaluate(model, val_loader, med_loss_fn, time_loss_fn, duration_loss_fn, device, args.duration_weight)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)

    with open(out_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_val_loss": best_val, "history": history}, f, ensure_ascii=False, indent=2)

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    with torch.no_grad():
        med_logits, time_pred, duration_pred = model(X_t.to(device))
        med_prob = torch.sigmoid(med_logits).cpu().numpy()
        time_hat = time_pred.cpu().numpy().reshape(-1)
        duration_hat = duration_pred.cpu().numpy().reshape(-1)

    decoded = [decode_regimen(row, med_vocab, args.med_threshold, args.top_k_meds) for row in med_prob]
    pred_regimen = [x[0] for x in decoded]
    pred_regimen_probs = [x[1] for x in decoded]

    pred_df = pd.DataFrame({
        "row_index": np.arange(len(df)),
        cfg.ID_COL: df[cfg.ID_COL].values if cfg.ID_COL in df.columns else np.arange(len(df)),
        cfg.STEP_INDEX_COL: df[cfg.STEP_INDEX_COL].values if cfg.STEP_INDEX_COL in df.columns else 0,
        "predicted_los": time_hat,
        "true_los": pd.to_numeric(df[cfg.OUTCOME_COL], errors="coerce").fillna(0).values,
        "predicted_step_duration": duration_hat,
        "true_step_duration": pd.to_numeric(df.get(cfg.STEP_DURATION_COL, 0), errors="coerce").fillna(0).values,
        "predicted_med_regimen": pred_regimen,
        "predicted_med_probabilities": pred_regimen_probs,
        "true_med_regimen": df[cfg.TARGET_MED_COL].fillna("").astype(str).values if cfg.TARGET_MED_COL in df.columns else "",
    })
    pred_df.to_csv(out_dir / "prediction_preview.csv", index=False, encoding="utf-8-sig")
    np.save(out_dir / "predicted_med_prob.npy", med_prob.astype(np.float32))
    print(f"Saved best model and predictions to {out_dir}")


if __name__ == "__main__":
    main()
