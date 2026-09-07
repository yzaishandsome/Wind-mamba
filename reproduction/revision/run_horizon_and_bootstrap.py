"""Five-seed horizon metrics and paired moving-block bootstrap analyses."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines import Baseline_Autoformer, Baseline_Mamba  # noqa: E402
from data_provider import USVDataset  # noqa: E402
from experiment_config import ALL_BOAT_FILES, SOURCE_BOAT_FILES, TARGET_BOAT_FILES, boat_ids_for, boat_tag  # noqa: E402
from model import EdgeWind_Mamba_Model  # noqa: E402


SEQ_LEN = 36
PRED_LEN = 6
HIDDEN_DIM = 96
BATCH_SIZE = 256
SEEDS = (42, 43, 44, 45, 46)
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260903
OUTPUT_ROOT = Path(
    os.getenv("WIND_MAMBA_HORIZON_OUTPUT", str(REPO_ROOT / "outputs" / "horizon_analysis"))
).resolve()
BACKUP_ROOTS = [
    Path(os.getenv("WIND_MAMBA_CHECKPOINT_ROOT", str(REPO_ROOT))).resolve()
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def locate_backup_root() -> Path:
    for root in BACKUP_ROOTS:
        edge = root / "weights" / "transfer_two_targets_v1" / "edgewind" / "rev4_seed_42"
        base = root / "weights" / "transfer_two_targets_v1" / "baselines_v2" / "rev4_seed_42"
        if (edge / "transfer_sd1042_best.pth").exists() and (base / "autoformer" / "transfer_sd1042_best.pth").exists():
            return root
    raise FileNotFoundError("Could not find the five-seed rev4 checkpoints.")


def source_scaler():
    dataset = USVDataset(
        SOURCE_BOAT_FILES,
        global_boat_ids=boat_ids_for(SOURCE_BOAT_FILES),
        flag="train",
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
    )
    return dataset.scaler


def target_dataset(target_file, scaler):
    return USVDataset(
        [target_file],
        global_boat_ids=boat_ids_for([target_file]),
        flag="test",
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        scaler=scaler,
    )


def wind_mamba(scaler):
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
        num_boats=len(ALL_BOAT_FILES),
        feature_stats=stats,
        mamba_backend="custom",
    ).to(DEVICE)


def make_model(model_key, scaler):
    if model_key == "wind_mamba":
        return wind_mamba(scaler)
    if model_key == "mamba":
        return Baseline_Mamba(seq_len=SEQ_LEN, pred_len=PRED_LEN, hidden_dim=HIDDEN_DIM).to(DEVICE)
    if model_key == "autoformer":
        return Baseline_Autoformer(seq_len=SEQ_LEN, pred_len=PRED_LEN, hidden_dim=HIDDEN_DIM).to(DEVICE)
    raise KeyError(model_key)


def checkpoint_path(root: Path, model_key: str, seed: int, target: str) -> Path:
    if model_key == "wind_mamba":
        return root / "weights" / "transfer_two_targets_v1" / "edgewind" / f"rev4_seed_{seed}" / f"transfer_{target}_best.pth"
    folder = "stdmamba" if model_key == "mamba" else "autoformer"
    return root / "weights" / "transfer_two_targets_v1" / "baselines_v2" / f"rev4_seed_{seed}" / folder / f"transfer_{target}_best.pth"


def collect_truth_inputs(dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    xs, ys, ids = [], [], []
    for x, y, boat_id in loader:
        xs.append(x)
        ys.append(y)
        ids.append(boat_id)
    return torch.cat(xs), torch.cat(ys), torch.cat(ids)


def model_predict(model, x, boat_id):
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), BATCH_SIZE):
            ws, wd_sin, wd_cos = model(x[start : start + BATCH_SIZE].to(DEVICE), boat_id[start : start + BATCH_SIZE].to(DEVICE))
            outputs.append(
                (
                    ws.squeeze(-1).cpu().numpy(),
                    wd_sin.squeeze(-1).cpu().numpy(),
                    wd_cos.squeeze(-1).cpu().numpy(),
                )
            )
    return tuple(np.concatenate([part[index] for part in outputs]) for index in range(3))


def persistence_predict(x, scaler):
    array = x.numpy()
    u = array[:, -1, 5] * scaler.scale_[5] + scaler.mean_[5]
    v = array[:, -1, 6] * scaler.scale_[6] + scaler.mean_[6]
    ws = np.sqrt(u**2 + v**2 + 1e-6)
    angle = np.remainder(np.arctan2(u, v) + np.pi, 2 * np.pi)
    return (
        np.repeat(ws[:, None], PRED_LEN, axis=1),
        np.repeat(np.sin(angle)[:, None], PRED_LEN, axis=1),
        np.repeat(np.cos(angle)[:, None], PRED_LEN, axis=1),
    )


def circular_error(pred_sin, pred_cos, true_sin, true_cos):
    pred = np.mod(np.degrees(np.arctan2(pred_sin, pred_cos)), 360.0)
    true = np.mod(np.degrees(np.arctan2(true_sin, true_cos)), 360.0)
    diff = np.abs(pred - true)
    return np.minimum(diff, 360.0 - diff)


def metric_rows(target, model, seed, true_ws, true_sin, true_cos, pred_ws, pred_sin, pred_cos):
    rows = []
    wd_error = circular_error(pred_sin, pred_cos, true_sin, true_cos)
    for horizon in range(PRED_LEN):
        error = pred_ws[:, horizon] - true_ws[:, horizon]
        rows.append(
            {
                "target": target,
                "model": model,
                "seed": seed,
                "horizon": horizon + 1,
                "ws_rmse": float(np.sqrt(np.mean(error**2))),
                "ws_mae": float(np.mean(np.abs(error))),
                "wd_mae": float(np.mean(wd_error[:, horizon])),
                "windows": len(true_ws),
            }
        )
    return rows


def acf_block_length(series: np.ndarray, minimum: int = PRED_LEN, max_lag: int = 240) -> tuple[int, int, float]:
    centered = np.asarray(series, dtype=float) - np.mean(series)
    variance = np.dot(centered, centered)
    if variance <= 0:
        return minimum, 1, 0.0
    threshold = 1.96 / math.sqrt(len(centered))
    chosen_lag = min(max_lag, len(centered) - 1)
    for lag in range(1, chosen_lag + 1):
        acf = float(np.dot(centered[:-lag], centered[lag:]) / variance)
        if abs(acf) <= threshold:
            chosen_lag = lag
            break
    return max(minimum, chosen_lag), chosen_lag, threshold


def moving_block_means(series: np.ndarray, block_length: int, rng, replicates: int):
    n = len(series)
    blocks = int(math.ceil(n / block_length))
    starts = rng.integers(0, n - block_length + 1, size=(replicates, blocks))
    offsets = np.arange(block_length)
    indices = (starts[..., None] + offsets).reshape(replicates, -1)[:, :n]
    return series[indices].mean(axis=1)


def bootstrap_comparison(prediction_store):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    replicate_store = {}
    for target, payload in prediction_store.items():
        true = payload["true_ws"]
        persistence = payload["persistence_ws"]
        wind_predictions = np.stack(payload["wind_mamba_ws_by_seed"], axis=0)
        wm_sq = np.mean((wind_predictions - true[None, ...]) ** 2, axis=(0, 2))
        ps_sq = np.mean((persistence - true) ** 2, axis=1)
        wm_abs = np.mean(np.abs(wind_predictions - true[None, ...]), axis=(0, 2))
        ps_abs = np.mean(np.abs(persistence - true), axis=1)
        diffs = {"squared_error": wm_sq - ps_sq, "absolute_error": wm_abs - ps_abs}
        block_candidates = [acf_block_length(values)[0] for values in diffs.values()]
        block_length = max(block_candidates)
        replicate_store[target] = {}
        for loss_name, difference in diffs.items():
            boot = moving_block_means(difference, block_length, rng, BOOTSTRAP_REPLICATES)
            replicate_store[target][loss_name] = boot
            persistence_mean = float(np.mean(ps_sq if loss_name == "squared_error" else ps_abs))
            mean_difference = float(np.mean(difference))
            rows.append(
                {
                    "target": target,
                    "loss": loss_name,
                    "mean_difference_wind_mamba_minus_persistence": mean_difference,
                    "relative_improvement_percent": float(-100.0 * mean_difference / persistence_mean),
                    "ci95_lower": float(np.quantile(boot, 0.025)),
                    "ci95_upper": float(np.quantile(boot, 0.975)),
                    "persistence_mean_loss": persistence_mean,
                    "wind_mamba_five_seed_mean_loss": float(persistence_mean + mean_difference),
                    "block_length_origins": block_length,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )
    for loss_name in ("squared_error", "absolute_error"):
        combined = np.mean(np.stack([replicate_store[target][loss_name] for target in sorted(replicate_store)]), axis=0)
        target_rows = [row for row in rows if row["loss"] == loss_name and row["target"] != "target_mean"]
        rows.append(
            {
                "target": "target_mean",
                "loss": loss_name,
                "mean_difference_wind_mamba_minus_persistence": float(np.mean([r["mean_difference_wind_mamba_minus_persistence"] for r in target_rows])),
                "relative_improvement_percent": float(np.mean([r["relative_improvement_percent"] for r in target_rows])),
                "ci95_lower": float(np.quantile(combined, 0.025)),
                "ci95_upper": float(np.quantile(combined, 0.975)),
                "persistence_mean_loss": float(np.mean([r["persistence_mean_loss"] for r in target_rows])),
                "wind_mamba_five_seed_mean_loss": float(np.mean([r["wind_mamba_five_seed_mean_loss"] for r in target_rows])),
                "block_length_origins": "target-specific",
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            }
        )
    return pd.DataFrame(rows)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    prediction_root = OUTPUT_ROOT / "predictions"
    prediction_root.mkdir(exist_ok=True)
    root = locate_backup_root()
    scaler = source_scaler()
    rows = []
    prediction_store = {}

    for target_file in TARGET_BOAT_FILES:
        target = boat_tag(target_file)
        dataset = target_dataset(target_file, scaler)
        x, y, boat_ids = collect_truth_inputs(dataset)
        true_ws = y[..., 0].numpy()
        true_sin = y[..., 1].numpy()
        true_cos = y[..., 2].numpy()
        persistence = persistence_predict(x, scaler)
        rows.extend(metric_rows(target, "Persistence", None, true_ws, true_sin, true_cos, *persistence))
        prediction_store[target] = {
            "true_ws": true_ws,
            "persistence_ws": persistence[0],
            "wind_mamba_ws_by_seed": [],
        }
        np.savez_compressed(
            prediction_root / f"{target}_persistence.npz",
            true_ws=true_ws,
            true_sin=true_sin,
            true_cos=true_cos,
            pred_ws=persistence[0],
            pred_sin=persistence[1],
            pred_cos=persistence[2],
        )

        for seed in SEEDS:
            for model_key, model_name in (("wind_mamba", "Wind-Mamba"), ("mamba", "Mamba"), ("autoformer", "Autoformer")):
                path = checkpoint_path(root, model_key, seed, target)
                model = make_model(model_key, scaler)
                checkpoint = torch.load(path, map_location=DEVICE)
                model.load_state_dict(checkpoint["model_state_dict"])
                prediction = model_predict(model, x, boat_ids)
                rows.extend(metric_rows(target, model_name, seed, true_ws, true_sin, true_cos, *prediction))
                if model_key == "wind_mamba":
                    prediction_store[target]["wind_mamba_ws_by_seed"].append(prediction[0])
                np.savez_compressed(
                    prediction_root / f"{target}_{model_key}_seed{seed}.npz",
                    true_ws=true_ws,
                    true_sin=true_sin,
                    true_cos=true_cos,
                    pred_ws=prediction[0],
                    pred_sin=prediction[1],
                    pred_cos=prediction[2],
                    checkpoint=str(path),
                )
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    raw = pd.DataFrame(rows)
    raw.to_csv(OUTPUT_ROOT / "horizon_metrics_by_seed_target.csv", index=False)
    seeded = raw[raw["seed"].notna()].copy()
    summary = (
        seeded.groupby(["target", "model", "horizon"])[["ws_rmse", "ws_mae", "wd_mae"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(column).strip("_") for column in summary.columns]
    persistence_rows = raw[raw["model"] == "Persistence"].copy()
    for metric in ("ws_rmse", "ws_mae", "wd_mae"):
        persistence_rows[f"{metric}_mean"] = persistence_rows[metric]
        persistence_rows[f"{metric}_std"] = 0.0
    summary = pd.concat(
        [summary, persistence_rows[["target", "model", "horizon", "ws_rmse_mean", "ws_rmse_std", "ws_mae_mean", "ws_mae_std", "wd_mae_mean", "wd_mae_std"]]],
        ignore_index=True,
    )
    target_mean = (
        summary[summary["model"] != "Persistence"]
        .groupby(["model", "horizon"], as_index=False)[["ws_rmse_mean", "ws_mae_mean", "wd_mae_mean"]]
        .mean()
    )
    target_mean["target"] = "target_mean"
    seed_target_mean = (
        seeded.groupby(["seed", "model", "horizon"], as_index=False)[["ws_rmse", "ws_mae", "wd_mae"]]
        .mean()
    )
    target_seed_summary = (
        seed_target_mean.groupby(["model", "horizon"])[["ws_rmse", "ws_mae", "wd_mae"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    target_seed_summary.columns = ["_".join(column).strip("_") for column in target_seed_summary.columns]
    target_seed_summary["target"] = "target_mean"
    target_mean = target_mean.drop(columns=["ws_rmse_mean", "ws_mae_mean", "wd_mae_mean"]).merge(
        target_seed_summary, on=["target", "model", "horizon"], how="left"
    )
    persistence_mean = persistence_rows.groupby("horizon", as_index=False)[["ws_rmse", "ws_mae", "wd_mae"]].mean()
    persistence_mean["target"] = "target_mean"
    persistence_mean["model"] = "Persistence"
    for metric in ("ws_rmse", "ws_mae", "wd_mae"):
        persistence_mean[f"{metric}_mean"] = persistence_mean[metric]
        persistence_mean[f"{metric}_std"] = 0.0
    target_mean = pd.concat(
        [target_mean, persistence_mean[["target", "model", "horizon", "ws_rmse_mean", "ws_rmse_std", "ws_mae_mean", "ws_mae_std", "wd_mae_mean", "wd_mae_std"]]],
        ignore_index=True,
    )
    final_summary = pd.concat([summary, target_mean], ignore_index=True).sort_values(["target", "model", "horizon"])
    final_summary.to_csv(OUTPUT_ROOT / "horizon_metrics_five_seed_summary.csv", index=False)

    bootstrap = bootstrap_comparison(prediction_store)
    bootstrap.to_csv(OUTPUT_ROOT / "wind_mamba_vs_persistence_block_bootstrap.csv", index=False)
    (OUTPUT_ROOT / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_root": str(root),
                "checkpoint_group": "rev4_seed_42 through rev4_seed_46",
                "device": str(DEVICE),
                "bootstrap_method": "paired chronological moving-block bootstrap over forecast origins",
                "bootstrap_series": "per-origin mean over six horizons; Wind-Mamba loss averaged over five seeds before resampling",
                "block_length_rule": "maximum across squared/absolute paired-difference ACF cutoff; first lag inside +/-1.96/sqrt(n), minimum H=6",
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(final_summary.to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
