"""Clean chronological split-conformal analysis using the submitted checkpoint.

The original 20% test block is split at the raw time-series level into equal
chronological calibration and final interval-test blocks. No windows cross the
boundary and the early-stopping validation split is not reused.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_provider import TRAIN_RATIO, VAL_RATIO, USVDataset  # noqa: E402
from experiment_config import (  # noqa: E402
    ALL_BOAT_FILES,
    SOURCE_BOAT_FILES,
    TARGET_BOAT_FILES,
    boat_ids_for,
    boat_tag,
)
from model import EdgeWind_Mamba_Model  # noqa: E402


SEQ_LEN = 36
PRED_LEN = 6
HIDDEN_DIM = 96
ALPHA = 0.10
BATCH_SIZE = 256
OUTPUT_ROOT = Path(
    os.getenv("WIND_MAMBA_CONFORMAL_OUTPUT", str(REPO_ROOT / "outputs" / "conformal_clean"))
).resolve()
CHECKPOINT_BASE = Path(
    os.getenv("WIND_MAMBA_CHECKPOINT_ROOT", str(REPO_ROOT / "weights"))
).resolve()
CHECKPOINT_ROOT_CANDIDATES = [
    CHECKPOINT_BASE / "transfer_two_targets_v1" / "edgewind",
    CHECKPOINT_BASE / "weights" / "transfer_two_targets_v1" / "edgewind" / "rev4_seed_42",
    CHECKPOINT_BASE,
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ChronologicalBlockDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, scaler, boat_id: int):
        self.feature_cols = [
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
        x = scaler.transform(frame[self.feature_cols].to_numpy(dtype=float))
        ws = frame[["WS_TRUE"]].to_numpy(dtype=float)
        wd = np.deg2rad(frame[["WD_TRUE"]].to_numpy(dtype=float))
        self.x = x.astype(np.float32)
        self.y = np.concatenate([ws, np.sin(wd), np.cos(wd)], axis=1).astype(np.float32)
        self.boat_id = int(boat_id)
        self.length = max(len(frame) - SEQ_LEN - PRED_LEN + 1, 0)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        split = index + SEQ_LEN
        return (
            torch.from_numpy(self.x[index:split]),
            torch.from_numpy(self.y[split : split + PRED_LEN]),
            torch.tensor(self.boat_id, dtype=torch.long),
        )


def locate_checkpoint_root() -> Path:
    for root in CHECKPOINT_ROOT_CANDIDATES:
        if all((root / f"transfer_{tag}_best.pth").exists() for tag in ("sd1042", "sd1091")):
            return root
    raise FileNotFoundError("Could not locate the submitted unsuffixed target checkpoints.")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_scaler_and_thresholds():
    source_train = USVDataset(
        SOURCE_BOAT_FILES,
        global_boat_ids=boat_ids_for(SOURCE_BOAT_FILES),
        flag="train",
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
    )
    all_train = USVDataset(
        ALL_BOAT_FILES,
        global_boat_ids=boat_ids_for(ALL_BOAT_FILES),
        flag="train",
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
    )
    source_ws = np.concatenate([segment[:, 0] for segment in source_train.y_data])
    all_ws = np.concatenate([segment[:, 0] for segment in all_train.y_data])
    thresholds = {
        "source_training_only_p95": float(np.percentile(source_ws, 95)),
        "all_training_slices_only_p95": float(np.percentile(all_ws, 95)),
    }
    return source_train.scaler, thresholds


def build_model(scaler):
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
        mamba_backend="legacy",
    ).to(DEVICE)


def collect(model, dataset):
    ws_true, ws_pred = [], []
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, boat_id in loader:
            pred, _, _ = model(batch_x.to(DEVICE), boat_id.to(DEVICE))
            ws_true.append(batch_y[..., 0].numpy())
            ws_pred.append(pred.squeeze(-1).cpu().numpy())
    return np.concatenate(ws_true), np.concatenate(ws_pred)


def finite_sample_quantiles(residuals: np.ndarray) -> tuple[np.ndarray, int]:
    ordered = np.sort(residuals, axis=0)
    rank_one_based = min(int(np.ceil((ordered.shape[0] + 1) * (1.0 - ALPHA))), ordered.shape[0])
    return ordered[rank_one_based - 1], rank_one_based


def intervals(pred: np.ndarray, quantiles: np.ndarray):
    lower = np.maximum(pred - quantiles.reshape(1, -1), 0.0)
    upper = pred + quantiles.reshape(1, -1)
    return lower, upper


def metrics(true: np.ndarray, lower: np.ndarray, upper: np.ndarray, mask=None):
    if mask is None:
        mask = np.ones_like(true, dtype=bool)
    true_m = true[mask]
    lower_m = lower[mask]
    upper_m = upper[mask]
    if true_m.size == 0:
        return {"samples": 0, "picp": np.nan, "mpiw": np.nan, "winkler": np.nan}
    score = upper_m - lower_m
    below = true_m < lower_m
    above = true_m > upper_m
    score[below] += 2.0 / ALPHA * (lower_m[below] - true_m[below])
    score[above] += 2.0 / ALPHA * (true_m[above] - upper_m[above])
    return {
        "samples": int(true_m.size),
        "picp": float(100.0 * np.mean((true_m >= lower_m) & (true_m <= upper_m))),
        "mpiw": float(np.mean(upper_m - lower_m)),
        "winkler": float(np.mean(score)),
    }


def selected_segment(target_file: str, scaler) -> pd.DataFrame:
    probe = USVDataset(
        [target_file],
        global_boat_ids=boat_ids_for([target_file]),
        flag="test",
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        scaler=scaler,
    )
    return probe._load_continuous_segments(target_file)[0]


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    prediction_root = OUTPUT_ROOT / "predictions"
    prediction_root.mkdir(exist_ok=True)
    checkpoint_root = locate_checkpoint_root()
    scaler, thresholds = source_scaler_and_thresholds()
    metric_rows = []
    split_rows = []
    checkpoint_rows = []

    for target_file in TARGET_BOAT_FILES:
        target = boat_tag(target_file)
        target_id = boat_ids_for([target_file])[0]
        frame = selected_segment(target_file, scaler)
        train_end = int(len(frame) * TRAIN_RATIO)
        val_end = train_end + int(len(frame) * VAL_RATIO)
        original_test = frame.iloc[val_end:].reset_index(drop=True)
        interval_split = len(original_test) // 2
        calibration_frame = original_test.iloc[:interval_split].reset_index(drop=True)
        final_test_frame = original_test.iloc[interval_split:].reset_index(drop=True)
        calibration = ChronologicalBlockDataset(calibration_frame, scaler, target_id)
        final_test = ChronologicalBlockDataset(final_test_frame, scaler, target_id)

        checkpoint_path = checkpoint_root / f"transfer_{target}_best.pth"
        model = build_model(scaler)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        calib_true, calib_pred = collect(model, calibration)
        test_true, test_pred = collect(model, final_test)
        quantiles, rank = finite_sample_quantiles(np.abs(calib_pred - calib_true))
        lower, upper = intervals(test_pred, quantiles)

        overall = metrics(test_true, lower, upper)
        metric_rows.append({"target": target, "threshold_definition": "not_applicable", "regime": "overall", **overall})
        for threshold_name, threshold in thresholds.items():
            upper_mask = test_true > threshold
            metric_rows.append(
                {
                    "target": target,
                    "threshold_definition": threshold_name,
                    "threshold_mps": threshold,
                    "regime": "upper_tail",
                    **metrics(test_true, lower, upper, upper_mask),
                }
            )
            metric_rows.append(
                {
                    "target": target,
                    "threshold_definition": threshold_name,
                    "threshold_mps": threshold,
                    "regime": "non_upper_tail",
                    **metrics(test_true, lower, upper, ~upper_mask),
                }
            )

        np.savez_compressed(
            prediction_root / f"{target}_clean_split_conformal.npz",
            calibration_true=calib_true,
            calibration_pred=calib_pred,
            final_test_true=test_true,
            final_test_pred=test_pred,
            final_test_lower=lower,
            final_test_upper=upper,
            quantiles=quantiles,
        )
        split_rows.append(
            {
                "target": target,
                "total_raw_rows": len(frame),
                "train_raw_rows": train_end,
                "validation_raw_rows": val_end - train_end,
                "original_test_raw_rows": len(original_test),
                "calibration_raw_rows": len(calibration_frame),
                "final_interval_test_raw_rows": len(final_test_frame),
                "calibration_windows": len(calibration),
                "final_interval_test_windows": len(final_test),
                "calibration_target_start": str(calibration_frame.loc[SEQ_LEN, "time"]),
                "calibration_target_end": str(calibration_frame.loc[len(calibration_frame) - 1, "time"]),
                "final_test_target_start": str(final_test_frame.loc[SEQ_LEN, "time"]),
                "final_test_target_end": str(final_test_frame.loc[len(final_test_frame) - 1, "time"]),
                "target_time_overlap": False,
                "train_ratio": train_end / len(frame),
                "validation_ratio": (val_end - train_end) / len(frame),
                "calibration_ratio": len(calibration_frame) / len(frame),
                "final_interval_test_ratio": len(final_test_frame) / len(frame),
                "conformal_rank_one_based": rank,
                "calibration_residual_windows": len(calibration),
            }
        )
        checkpoint_rows.append(
            {
                "target": target,
                "checkpoint": str(checkpoint_path),
                "sha256": sha256(checkpoint_path),
                "backend": "legacy",
                "best_score": checkpoint.get("best_score"),
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    means = []
    for (threshold_definition, regime), group in metrics_df.groupby(["threshold_definition", "regime"], dropna=False):
        row = {
            "target": "target_mean",
            "threshold_definition": threshold_definition,
            "threshold_mps": group["threshold_mps"].dropna().iloc[0] if group["threshold_mps"].notna().any() else np.nan,
            "regime": regime,
            "samples": int(group["samples"].sum()),
        }
        for name in ("picp", "mpiw", "winkler"):
            row[name] = float(group[name].mean())
        means.append(row)
    metrics_df = pd.concat([metrics_df, pd.DataFrame(means)], ignore_index=True)
    metrics_df.to_csv(OUTPUT_ROOT / "clean_interval_metrics.csv", index=False)
    pd.DataFrame(split_rows).to_csv(OUTPUT_ROOT / "chronological_split_metadata.csv", index=False)
    pd.DataFrame(checkpoint_rows).to_csv(OUTPUT_ROOT / "checkpoint_manifest.csv", index=False)
    pd.DataFrame(
        [{"threshold_definition": name, "exact_threshold_mps": value} for name, value in thresholds.items()]
    ).to_csv(OUTPUT_ROOT / "training_only_thresholds.csv", index=False)
    (OUTPUT_ROOT / "run_metadata.json").write_text(
        json.dumps(
            {
                "alpha": ALPHA,
                "nominal_coverage_percent": 100 * (1 - ALPHA),
                "sequence_length": SEQ_LEN,
                "prediction_length": PRED_LEN,
                "checkpoint_root": str(checkpoint_root),
                "device": str(DEVICE),
                "margin_inflation": False,
                "lower_bound_clipped_at_zero": True,
                "split_rule": "raw original-test block split 50/50 chronologically before window generation",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
