"""One-seed COG/HDG circular-encoding sensitivity experiment.

Variant A is evaluated from the existing rev4 seed-42 checkpoint. Variant B is
trained independently with 12 input channels and writes only below this folder.
"""

from __future__ import annotations

import csv
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_provider import TRAIN_RATIO, VAL_RATIO, USVDataset  # noqa: E402
from experiment_config import ALL_BOAT_FILES, SOURCE_BOAT_FILES, TARGET_BOAT_FILES, boat_ids_for, boat_tag  # noqa: E402
from model import EdgeWind_Mamba_Model, build_loss  # noqa: E402


SEED = 42
SEQ_LEN = 36
PRED_LEN = 6
HIDDEN_DIM = 96
D_STATE = 16
MAMBA_LAYERS = 3
BATCH_SIZE = 32
PRETRAIN_EPOCHS = 60
FINETUNE_EPOCHS = 20
PRETRAIN_LR = 2e-4
FINETUNE_LR = 5e-5
WEIGHT_DECAY = 1e-4
PATIENCE = 8
GRAD_MAX_NORM = 1.0
OUTPUT_BASE = Path(
    os.getenv("WIND_MAMBA_REVISION_OUTPUT", str(REPO_ROOT / "outputs" / "revision"))
).resolve()
OUTPUT_ROOT = OUTPUT_BASE / "circular_encoding"
CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"
BACKUP_ROOTS = [
    Path(os.getenv("WIND_MAMBA_CHECKPOINT_ROOT", str(REPO_ROOT))).resolve()
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ORIGINAL_FEATURES = [
    "latitude",
    "longitude",
    "SOG",
    "COG",
    "HDG",
    "UWND_MEAN",
    "VWND_MEAN",
    "GUST_WND_MEAN",
    "TEMP_AIR_MEAN",
    "BARO_PRES_MEAN",
]
CIRCULAR_FEATURES = [
    "latitude",
    "longitude",
    "SOG",
    "COG_sin",
    "COG_cos",
    "HDG_sin",
    "HDG_cos",
    "UWND_MEAN",
    "VWND_MEAN",
    "GUST_WND_MEAN",
    "TEMP_AIR_MEAN",
    "BARO_PRES_MEAN",
]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def segment_reader():
    reader = object.__new__(USVDataset)
    reader.seq_len = SEQ_LEN
    reader.pred_len = PRED_LEN
    reader.feature_cols = ORIGINAL_FEATURES
    reader.target_cols = ["WS_TRUE", "WD_TRUE"]
    return reader


def circular_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    cog = np.deg2rad(result["COG"].to_numpy(dtype=float))
    hdg = np.deg2rad(result["HDG"].to_numpy(dtype=float))
    result["COG_sin"] = np.sin(cog)
    result["COG_cos"] = np.cos(cog)
    result["HDG_sin"] = np.sin(hdg)
    result["HDG_cos"] = np.cos(hdg)
    return result


class CircularInputDataset(Dataset):
    def __init__(self, file_paths, global_boat_ids, flag, scaler=None):
        self.file_paths = file_paths
        self.global_boat_ids = global_boat_ids
        self.flag = flag
        self.reader = segment_reader()
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.x_data = []
        self.y_data = []
        self.boat_ids = []

        selected = [(file_path, self.reader._load_continuous_segments(file_path)[0]) for file_path in file_paths]
        if scaler is None:
            training = []
            for _, frame in selected:
                train_end = int(len(frame) * TRAIN_RATIO)
                training.append(circular_frame(frame.iloc[:train_end])[CIRCULAR_FEATURES])
            self.scaler.fit(pd.concat(training, axis=0).to_numpy(dtype=float))

        for local_index, (_, frame) in enumerate(selected):
            train_end = int(len(frame) * TRAIN_RATIO)
            val_end = train_end + int(len(frame) * VAL_RATIO)
            if flag == "train":
                split = frame.iloc[:train_end].copy()
            elif flag == "val":
                split = frame.iloc[train_end:val_end].copy()
            else:
                split = frame.iloc[val_end:].copy()
            split = circular_frame(split)
            x = self.scaler.transform(split[CIRCULAR_FEATURES].to_numpy(dtype=float)).astype(np.float32)
            ws = split[["WS_TRUE"]].to_numpy(dtype=float)
            wd = np.deg2rad(split[["WD_TRUE"]].to_numpy(dtype=float))
            y = np.concatenate([ws, np.sin(wd), np.cos(wd)], axis=1).astype(np.float32)
            self.x_data.append(x)
            self.y_data.append(y)
            self.boat_ids.append(global_boat_ids[local_index])
        self.counts = [max(len(x) - SEQ_LEN - PRED_LEN + 1, 0) for x in self.x_data]
        self.total = sum(self.counts)

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        for segment_index, count in enumerate(self.counts):
            if index < count:
                end = index + SEQ_LEN
                return (
                    torch.from_numpy(self.x_data[segment_index][index:end]),
                    torch.from_numpy(self.y_data[segment_index][end : end + PRED_LEN]),
                    torch.tensor(self.boat_ids[segment_index], dtype=torch.long),
                )
            index -= count
        raise IndexError(index)


def locate_backup_root() -> Path:
    for root in BACKUP_ROOTS:
        checkpoint = root / "weights" / "transfer_two_targets_v1" / "edgewind" / "rev4_seed_42" / "transfer_sd1042_best.pth"
        if checkpoint.exists():
            return root
    raise FileNotFoundError("Could not locate rev4 seed-42 checkpoints.")


def original_scaler():
    return USVDataset(
        SOURCE_BOAT_FILES,
        global_boat_ids=boat_ids_for(SOURCE_BOAT_FILES),
        flag="train",
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
    ).scaler


def original_model(scaler):
    stats = {
        "u_mean": scaler.mean_[5],
        "u_scale": scaler.scale_[5],
        "v_mean": scaler.mean_[6],
        "v_scale": scaler.scale_[6],
    }
    return EdgeWind_Mamba_Model(
        in_dim=10,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        hidden_dim=HIDDEN_DIM,
        d_state=D_STATE,
        n_mamba_layers=MAMBA_LAYERS,
        num_boats=len(ALL_BOAT_FILES),
        feature_stats=stats,
        mamba_backend="custom",
    ).to(DEVICE)


def circular_model(scaler):
    stats = {
        "u_mean": scaler.mean_[7],
        "u_scale": scaler.scale_[7],
        "v_mean": scaler.mean_[8],
        "v_scale": scaler.scale_[8],
    }
    model = EdgeWind_Mamba_Model(
        in_dim=12,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        hidden_dim=HIDDEN_DIM,
        d_state=D_STATE,
        n_mamba_layers=MAMBA_LAYERS,
        num_boats=len(ALL_BOAT_FILES),
        feature_stats=stats,
        mamba_backend="custom",
    ).to(DEVICE)
    model.u_idx = 7
    model.v_idx = 8
    return model


def loader(dataset, shuffle):
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


def batch_statistics(ws_pred, sin_pred, cos_pred, y):
    ws_true = y[..., 0:1]
    sin_true = y[..., 1:2]
    cos_true = y[..., 2:3]
    ws_error = ws_pred - ws_true
    pred_deg = torch.remainder(torch.rad2deg(torch.atan2(sin_pred, cos_pred)), 360.0)
    true_deg = torch.remainder(torch.rad2deg(torch.atan2(sin_true, cos_true)), 360.0)
    wd_diff = torch.abs(pred_deg - true_deg)
    return {
        "sq": float(torch.sum(ws_error**2).item()),
        "abs": float(torch.sum(torch.abs(ws_error)).item()),
        "wd": float(torch.sum(torch.minimum(wd_diff, 360.0 - wd_diff)).item()),
        "true_sum": float(torch.sum(ws_true).item()),
        "true_sq_sum": float(torch.sum(ws_true**2).item()),
        "count": int(ws_true.numel()),
    }


def run_epoch(model, data_loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {name: 0.0 for name in ("loss", "sq", "abs", "wd", "true_sum", "true_sq_sum")}
    total_points = 0
    total_windows = 0
    for x, y, boat_id in data_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        boat_id = boat_id.to(DEVICE)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            ws, sin_pred, cos_pred = model(x, boat_id)
            loss, _, _ = criterion(ws, sin_pred, cos_pred, y[..., 0:1], y[..., 1:2], y[..., 2:3])
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_MAX_NORM)
                optimizer.step()
        stats = batch_statistics(ws.detach(), sin_pred.detach(), cos_pred.detach(), y)
        totals["loss"] += float(loss.item()) * len(x)
        for name in ("sq", "abs", "wd", "true_sum", "true_sq_sum"):
            totals[name] += stats[name]
        total_points += stats["count"]
        total_windows += len(x)
    mean_true = totals["true_sum"] / total_points
    total_variance = totals["true_sq_sum"] - total_points * mean_true**2
    return {
        "loss": totals["loss"] / total_windows,
        "ws_rmse": math.sqrt(totals["sq"] / total_points),
        "ws_mae": totals["abs"] / total_points,
        "wd_mae": totals["wd"] / total_points,
        "r2": 1.0 - totals["sq"] / total_variance,
    }


def score(metrics):
    return 0.7 * metrics["ws_rmse"] + 0.3 * metrics["wd_mae"] / 100.0


def fit_stage(model, name, train_data, val_data, criterion, epochs, learning_rate, checkpoint_path, log_rows):
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Reusing isolated checkpoint: {checkpoint_path}")
        return
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_score = float("inf")
    stagnant = 0
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, loader(train_data, True), criterion, optimizer)
        val_metrics = run_epoch(model, loader(val_data, False), criterion)
        scheduler.step()
        current = score(val_metrics)
        log_rows.append({"stage": name, "epoch": epoch, "score": current, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}})
        print(f"{name} epoch={epoch:02d} val_ws={val_metrics['ws_rmse']:.4f} val_wd={val_metrics['wd_mae']:.2f} score={current:.4f}", flush=True)
        if current < best_score:
            best_score = current
            stagnant = 0
            torch.save({"model_state_dict": model.state_dict(), "best_score": best_score, "variant": "circular_12d", "seed": SEED}, checkpoint_path)
        else:
            stagnant += 1
            if stagnant >= PATIENCE:
                break
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])


def evaluate_variant_a(backup_root, scaler):
    rows = []
    for target_file in TARGET_BOAT_FILES:
        target = boat_tag(target_file)
        dataset = USVDataset(
            [target_file],
            global_boat_ids=boat_ids_for([target_file]),
            flag="test",
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN,
            scaler=scaler,
        )
        model = original_model(scaler)
        path = backup_root / "weights" / "transfer_two_targets_v1" / "edgewind" / "rev4_seed_42" / f"transfer_{target}_best.pth"
        model.load_state_dict(torch.load(path, map_location=DEVICE)["model_state_dict"])
        metrics = run_epoch(model, loader(dataset, False), build_loss("smoothl1_dircos").to(DEVICE))
        rows.append({"variant": "A_degree_10d", "seed": SEED, "target": target, "checkpoint": str(path), **metrics})
    return rows


def train_variant_b():
    log_rows = []
    source_ids = boat_ids_for(SOURCE_BOAT_FILES)
    source_train = CircularInputDataset(SOURCE_BOAT_FILES, source_ids, "train")
    scaler = source_train.scaler
    source_val = CircularInputDataset(SOURCE_BOAT_FILES, source_ids, "val", scaler=scaler)
    criterion = build_loss("smoothl1_dircos", wd_weight=1.0).to(DEVICE)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    pretrain_path = CHECKPOINT_ROOT / "circular12d_transfer_pretrain_source8_seed42.pth"
    pretrain_model = circular_model(scaler)
    fit_stage(pretrain_model, "circular-pretrain", source_train, source_val, criterion, PRETRAIN_EPOCHS, PRETRAIN_LR, pretrain_path, log_rows)
    rows = []
    for target_file in TARGET_BOAT_FILES:
        target = boat_tag(target_file)
        target_id = boat_ids_for([target_file])
        train_data = CircularInputDataset([target_file], target_id, "train", scaler=scaler)
        val_data = CircularInputDataset([target_file], target_id, "val", scaler=scaler)
        test_data = CircularInputDataset([target_file], target_id, "test", scaler=scaler)
        model = circular_model(scaler)
        model.load_state_dict(torch.load(pretrain_path, map_location=DEVICE)["model_state_dict"])
        target_path = CHECKPOINT_ROOT / f"circular12d_transfer_{target}_seed42.pth"
        fit_stage(model, f"circular-finetune-{target}", train_data, val_data, criterion, FINETUNE_EPOCHS, FINETUNE_LR, target_path, log_rows)
        metrics = run_epoch(model, loader(test_data, False), criterion)
        rows.append({"variant": "B_circular_12d", "seed": SEED, "target": target, "checkpoint": str(target_path), **metrics})
    pd.DataFrame(log_rows).to_csv(OUTPUT_ROOT / "training_log.csv", index=False)
    return rows


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    backup_root = locate_backup_root()
    rows = evaluate_variant_a(backup_root, original_scaler())
    rows.extend(train_variant_b())
    raw = pd.DataFrame(rows)
    raw.to_csv(OUTPUT_ROOT / "circular_sensitivity_target_metrics.csv", index=False)
    means = raw.groupby(["variant", "seed"], as_index=False)[["ws_rmse", "ws_mae", "wd_mae", "r2"]].mean()
    means["target"] = "target_mean"
    pd.concat([raw, means], ignore_index=True).to_csv(OUTPUT_ROOT / "circular_sensitivity_results.csv", index=False)
    with (OUTPUT_ROOT / "configuration.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["setting", "value"])
        for key, value in {
            "seed": SEED,
            "sequence_length": SEQ_LEN,
            "prediction_length": PRED_LEN,
            "hidden_dim": HIDDEN_DIM,
            "state_dimension": D_STATE,
            "mamba_layers": MAMBA_LAYERS,
            "lambda_d": 1.0,
            "batch_size": BATCH_SIZE,
            "pretrain_epochs": PRETRAIN_EPOCHS,
            "finetune_epochs": FINETUNE_EPOCHS,
            "pretrain_lr": PRETRAIN_LR,
            "finetune_lr": FINETUNE_LR,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "residual_bound_mps": 4.0,
            "mamba_backend": "custom",
            "loss": "SmoothL1 + directional cosine",
            "device": DEVICE,
        }.items():
            writer.writerow([key, value])
    print(pd.concat([raw, means], ignore_index=True).to_string(index=False))


if __name__ == "__main__":
    main()
