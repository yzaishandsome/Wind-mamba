"""Evaluate the deterministic last-wind-vector persistence baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_provider import USVDataset
from experiment_config import SOURCE_BOAT_FILES, TARGET_BOAT_FILES, boat_ids_for, boat_tag


SEQ_LEN = 36
PRED_LEN = 6
BATCH_SIZE = 256


def circular_error(pred_sin, pred_cos, true_sin, true_cos):
    pred = np.mod(np.degrees(np.arctan2(pred_sin, pred_cos)), 360.0)
    true = np.mod(np.degrees(np.arctan2(true_sin, true_cos)), 360.0)
    difference = np.abs(pred - true)
    return np.minimum(difference, 360.0 - difference)


def evaluate(target_file, scaler):
    dataset = USVDataset(
        [target_file],
        global_boat_ids=boat_ids_for([target_file]),
        flag="test",
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        scaler=scaler,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    true_ws, true_sin, true_cos, pred_ws, pred_sin, pred_cos = ([] for _ in range(6))
    for batch_x, batch_y, _ in loader:
        u = batch_x[:, -1, 5].numpy() * scaler.scale_[5] + scaler.mean_[5]
        v = batch_x[:, -1, 6].numpy() * scaler.scale_[6] + scaler.mean_[6]
        ws = np.sqrt(u**2 + v**2 + 1e-6)
        angle = np.remainder(np.arctan2(u, v) + np.pi, 2 * np.pi)
        true_ws.append(batch_y[:, :, 0].numpy())
        true_sin.append(batch_y[:, :, 1].numpy())
        true_cos.append(batch_y[:, :, 2].numpy())
        pred_ws.append(np.repeat(ws[:, None], PRED_LEN, axis=1))
        pred_sin.append(np.repeat(np.sin(angle)[:, None], PRED_LEN, axis=1))
        pred_cos.append(np.repeat(np.cos(angle)[:, None], PRED_LEN, axis=1))

    true_ws = np.concatenate(true_ws)
    true_sin = np.concatenate(true_sin)
    true_cos = np.concatenate(true_cos)
    pred_ws = np.concatenate(pred_ws)
    pred_sin = np.concatenate(pred_sin)
    pred_cos = np.concatenate(pred_cos)
    error = pred_ws - true_ws
    denominator = np.sum((true_ws - np.mean(true_ws)) ** 2)
    return {
        "target": boat_tag(target_file),
        "samples": int(true_ws.size),
        "ws_rmse": float(np.sqrt(np.mean(error**2))),
        "ws_mae": float(np.mean(np.abs(error))),
        "wd_mae": float(np.mean(circular_error(pred_sin, pred_cos, true_sin, true_cos))),
        "r2": float(1.0 - np.sum(error**2) / denominator),
    }


def main():
    source = USVDataset(
        SOURCE_BOAT_FILES,
        global_boat_ids=boat_ids_for(SOURCE_BOAT_FILES),
        flag="train",
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
    )
    rows = [evaluate(target, source.scaler) for target in TARGET_BOAT_FILES]
    numeric = ("ws_rmse", "ws_mae", "wd_mae", "r2")
    rows.append({"target": "target_mean", "samples": sum(row["samples"] for row in rows), **{
        metric: float(np.mean([row[metric] for row in rows])) for metric in numeric
    }})
    output = Path("comparison_outputs") / "transfer_two_targets_v1" / "persistence_baseline_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(output.resolve())


if __name__ == "__main__":
    main()
