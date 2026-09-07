"""Shared, isolated utilities for Applied Soft Computing revision experiments.

This module imports the submitted implementation without modifying it. Every
write performed here is directed below major_revision_round1.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PROJECT_ROOT = SCRIPT_DIR
ROUND_ROOT = Path(
    os.getenv("WIND_MAMBA_REVISION_OUTPUT", str(REPO_ROOT / "outputs" / "revision"))
).resolve()

import sys

sys.path.insert(0, str(PROJECT_ROOT))

from data_provider import TRAIN_RATIO, VAL_RATIO, USVDataset  # noqa: E402
from experiment_config import (  # noqa: E402
    ALL_BOAT_FILES,
    SOURCE_BOAT_FILES,
    TARGET_BOAT_FILES,
    boat_ids_for,
    boat_tag,
)
from model import EdgeWind_Mamba_Model, build_loss  # noqa: E402


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
SEEDS = (42, 43, 44, 45, 46)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_ROOT = Path(
    os.getenv("WIND_MAMBA_CHECKPOINT_ROOT", str(REPO_ROOT))
).resolve()
BACKUP_ROOTS = (CHECKPOINT_ROOT,)


def absolute_files(files: Iterable[str]) -> List[str]:
    resolved = []
    for path in files:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        resolved.append(str(candidate.resolve()))
    return resolved


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


def feature_stats(scaler, u_idx: int = 5, v_idx: int = 6) -> Dict[str, float]:
    return {
        "u_mean": float(scaler.mean_[u_idx]),
        "u_scale": float(scaler.scale_[u_idx]),
        "v_mean": float(scaler.mean_[v_idx]),
        "v_scale": float(scaler.scale_[v_idx]),
    }


def standard_source_datasets():
    source_ids = boat_ids_for(SOURCE_BOAT_FILES)
    source_train = USVDataset(
        absolute_files(SOURCE_BOAT_FILES),
        global_boat_ids=source_ids,
        flag="train",
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
    )
    source_val = USVDataset(
        absolute_files(SOURCE_BOAT_FILES),
        global_boat_ids=source_ids,
        flag="val",
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        scaler=source_train.scaler,
    )
    return source_train, source_val


def standard_target_datasets(target_file: str, scaler):
    target_id = boat_ids_for([target_file])
    path = absolute_files([target_file])
    return {
        flag: USVDataset(
            path,
            global_boat_ids=target_id,
            flag=flag,
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN,
            scaler=scaler,
        )
        for flag in ("train", "val", "test")
    }


class MagnitudeOnlyFrequencyExtractor(nn.Module):
    """Parameter-matched magnitude-only, zero-phase frequency branch."""

    def __init__(self, in_dim: int, hidden_dim: int, seq_len: int):
        super().__init__()
        self.freq_proj = nn.Linear(in_dim, hidden_dim)
        self.real_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.imag_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.real_norm = nn.LayerNorm(hidden_dim)
        self.imag_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        _, seq_len, _ = x.shape
        x_proj = self.freq_proj(x)
        magnitude = torch.abs(torch.fft.rfft(x_proj, dim=1, norm="ortho"))
        branch_a = self.real_conv(magnitude.transpose(1, 2)).transpose(1, 2)
        branch_b = self.imag_conv(magnitude.transpose(1, 2)).transpose(1, 2)
        magnitude_features = F.softplus(
            (self.real_norm(branch_a) + self.imag_norm(branch_b)) / math.sqrt(2.0)
        )
        zero_phase = torch.complex(magnitude_features, torch.zeros_like(magnitude_features))
        h_time = torch.fft.irfft(zero_phase, dim=1, norm="ortho", n=seq_len)
        return self.out_proj(h_time) + x_proj


class MagnitudeOnlyModel(EdgeWind_Mamba_Model):
    def __init__(self, *args, **kwargs):
        in_dim = int(kwargs.get("in_dim", 10))
        hidden_dim = int(kwargs.get("hidden_dim", HIDDEN_DIM))
        seq_len = int(kwargs.get("seq_len", SEQ_LEN))
        super().__init__(*args, **kwargs)
        self.turbulence_extractor = MagnitudeOnlyFrequencyExtractor(in_dim, hidden_dim, seq_len)
        self.turbulence_extractor.apply(self._init_weights)


class NoFFTModel(EdgeWind_Mamba_Model):
    def forward(self, x, boat_id):
        batch_size = x.size(0)
        base_u, base_v = self._last_wind_vector(x)
        h_input = self.input_proj(x) + self.boat_embedding(boat_id).unsqueeze(1)
        h_m = self.mamba_backbone(h_input)
        h_context = self.pool(h_m)
        horizon_ids = torch.arange(self.pred_len, device=x.device)
        horizon_tokens = self.horizon_embedding(horizon_ids).unsqueeze(0).expand(batch_size, -1, -1)
        decoder_input = self.decoder_input_proj(
            torch.cat([h_context.unsqueeze(1).expand(-1, self.pred_len, -1), horizon_tokens], dim=-1)
        )
        h_state, _ = self.horizon_decoder(decoder_input, h_context.unsqueeze(0).contiguous())
        u_delta = self.max_uv_delta * torch.tanh(self.u_delta_head(h_state))
        v_delta = self.max_uv_delta * torch.tanh(self.v_delta_head(h_state))
        pred_u = base_u.unsqueeze(1).unsqueeze(-1) + u_delta
        pred_v = base_v.unsqueeze(1).unsqueeze(-1) + v_delta
        ws_pred, pred_angle = self._vector_to_ws_angle(pred_u.squeeze(-1), pred_v.squeeze(-1))
        ws_pred = torch.clamp(ws_pred.unsqueeze(-1), min=0.0)
        pred_angle = pred_angle.unsqueeze(-1)
        return ws_pred, torch.sin(pred_angle), torch.cos(pred_angle)


class ResidualBoundModel(EdgeWind_Mamba_Model):
    def __init__(self, *args, bound_mode: str, bound_u: float = 4.0, bound_v: float = 4.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.bound_mode = bound_mode
        self.bound_u_value = float(bound_u)
        self.bound_v_value = float(bound_v)
        if bound_mode == "learnable":
            self.raw_bound_u = nn.Parameter(torch.tensor(_inverse_softplus(self.bound_u_value), dtype=torch.float32))
            self.raw_bound_v = nn.Parameter(torch.tensor(_inverse_softplus(self.bound_v_value), dtype=torch.float32))

    def current_bounds(self) -> Tuple[float, float]:
        if self.bound_mode == "learnable":
            return (
                float(F.softplus(self.raw_bound_u).detach().cpu()),
                float(F.softplus(self.raw_bound_v).detach().cpu()),
            )
        if self.bound_mode == "unbounded":
            return float("inf"), float("inf")
        return self.bound_u_value, self.bound_v_value

    def forward(self, x, boat_id):
        batch_size = x.size(0)
        base_u, base_v = self._last_wind_vector(x)
        h_input = self.input_proj(x) + self.boat_embedding(boat_id).unsqueeze(1)
        h_m = self.mamba_backbone(h_input)
        h_f = self.turbulence_extractor(x)
        gate = 0.5 + torch.sigmoid(self.gate_network(torch.cat([h_m, h_f], dim=-1)))
        h_combined = torch.cat([h_m, gate * h_f], dim=-1)
        h_fused_seq = h_m + self.fusion_proj(h_combined)
        h_context = self.pool(h_fused_seq)
        horizon_ids = torch.arange(self.pred_len, device=x.device)
        horizon_tokens = self.horizon_embedding(horizon_ids).unsqueeze(0).expand(batch_size, -1, -1)
        decoder_input = self.decoder_input_proj(
            torch.cat([h_context.unsqueeze(1).expand(-1, self.pred_len, -1), horizon_tokens], dim=-1)
        )
        h_state, _ = self.horizon_decoder(decoder_input, h_context.unsqueeze(0).contiguous())
        raw_u = self.u_delta_head(h_state)
        raw_v = self.v_delta_head(h_state)
        if self.bound_mode == "unbounded":
            u_delta, v_delta = raw_u, raw_v
        elif self.bound_mode == "learnable":
            u_delta = F.softplus(self.raw_bound_u) * torch.tanh(raw_u)
            v_delta = F.softplus(self.raw_bound_v) * torch.tanh(raw_v)
        else:
            u_delta = self.bound_u_value * torch.tanh(raw_u)
            v_delta = self.bound_v_value * torch.tanh(raw_v)
        pred_u = base_u.unsqueeze(1).unsqueeze(-1) + u_delta
        pred_v = base_v.unsqueeze(1).unsqueeze(-1) + v_delta
        ws_pred, pred_angle = self._vector_to_ws_angle(pred_u.squeeze(-1), pred_v.squeeze(-1))
        ws_pred = torch.clamp(ws_pred.unsqueeze(-1), min=0.0)
        pred_angle = pred_angle.unsqueeze(-1)
        return ws_pred, torch.sin(pred_angle), torch.cos(pred_angle)


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(float(value)))


def build_model(kind: str, scaler, bound_u: float = 4.0, bound_v: float = 4.0, in_dim: int = 10):
    stats = feature_stats(scaler, 5, 6)
    common = dict(
        in_dim=in_dim,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        hidden_dim=HIDDEN_DIM,
        d_state=D_STATE,
        n_mamba_layers=MAMBA_LAYERS,
        num_boats=len(ALL_BOAT_FILES),
        feature_stats=stats,
        mamba_backend="custom",
    )
    if kind == "full":
        model = EdgeWind_Mamba_Model(**common)
    elif kind == "magnitude_only":
        model = MagnitudeOnlyModel(**common)
    elif kind == "no_fft":
        model = NoFFTModel(**common)
    elif kind == "unbounded":
        model = ResidualBoundModel(**common, bound_mode="unbounded")
    elif kind == "fixed6":
        model = ResidualBoundModel(**common, bound_mode="fixed", bound_u=6.0, bound_v=6.0)
    elif kind == "training_derived":
        model = ResidualBoundModel(**common, bound_mode="fixed", bound_u=bound_u, bound_v=bound_v)
    elif kind == "learnable":
        model = ResidualBoundModel(**common, bound_mode="learnable", bound_u=bound_u, bound_v=bound_v)
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    return model.to(DEVICE)


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _batch_stats(ws_pred, sin_pred, cos_pred, y):
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
        x = x.to(DEVICE).float()
        y = y.to(DEVICE).float()
        boat_id = boat_id.to(DEVICE).long()
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            ws, sin_pred, cos_pred = model(x, boat_id)
            loss, _, _ = criterion(ws, sin_pred, cos_pred, y[..., 0:1], y[..., 1:2], y[..., 2:3])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss encountered.")
            if training:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_MAX_NORM)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("Non-finite gradient norm encountered.")
                optimizer.step()
        stats = _batch_stats(ws.detach(), sin_pred.detach(), cos_pred.detach(), y)
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


def validation_score(metrics: Dict[str, float]) -> float:
    return 0.7 * metrics["ws_rmse"] + 0.3 * metrics["wd_mae"] / 100.0


def fit_stage(
    model,
    stage_name: str,
    train_data,
    val_data,
    criterion,
    epochs: int,
    learning_rate: float,
    checkpoint_path: Path,
    log_path: Path,
    metadata: Dict,
):
    completion_path = checkpoint_path.with_suffix(".complete.json")
    if checkpoint_path.exists() and completion_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Reusing completed revision checkpoint: {checkpoint_path}", flush=True)
        return json.loads(completion_path.read_text(encoding="utf-8"))

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_score = float("inf")
    stagnant = 0
    started = time.perf_counter()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    epochs_run = 0
    with log_path.open("a", encoding="utf-8") as log_handle:
        for epoch in range(1, epochs + 1):
            train_metrics = run_epoch(model, make_loader(train_data, True), criterion, optimizer)
            val_metrics = run_epoch(model, make_loader(val_data, False), criterion)
            scheduler.step()
            current = validation_score(val_metrics)
            epochs_run = epoch
            record = {
                "run_id": run_id,
                "stage": stage_name,
                "epoch": epoch,
                "score": current,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            log_handle.write(json.dumps(record) + "\n")
            log_handle.flush()
            print(
                f"{stage_name} epoch={epoch:02d} val_ws={val_metrics['ws_rmse']:.4f} "
                f"val_wd={val_metrics['wd_mae']:.2f} score={current:.4f}",
                flush=True,
            )
            if current < best_score:
                best_score = current
                stagnant = 0
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "best_score": best_score,
                    "val_metrics": val_metrics,
                    **metadata,
                }
                torch.save(checkpoint, checkpoint_path)
            else:
                stagnant += 1
                if stagnant >= PATIENCE:
                    break
    if not checkpoint_path.exists():
        raise RuntimeError(f"{stage_name} did not produce a valid checkpoint.")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    completion = {
        "stage": stage_name,
        "epochs_run": epochs_run,
        "best_score": float(checkpoint["best_score"]),
        "runtime_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint_path),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    completion_path.write_text(json.dumps(completion, indent=2), encoding="utf-8")
    return completion


def collect_predictions(model, dataset):
    model.eval()
    xs, ys, pred_ws, pred_sin, pred_cos, boats = [], [], [], [], [], []
    with torch.no_grad():
        for x, y, boat_id in make_loader(dataset, False):
            ws, sin_value, cos_value = model(x.to(DEVICE).float(), boat_id.to(DEVICE).long())
            xs.append(x.numpy())
            ys.append(y.numpy())
            boats.append(boat_id.numpy())
            pred_ws.append(ws.squeeze(-1).cpu().numpy())
            pred_sin.append(sin_value.squeeze(-1).cpu().numpy())
            pred_cos.append(cos_value.squeeze(-1).cpu().numpy())
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    ws_true = y[..., 0]
    wd_true = np.mod(np.rad2deg(np.arctan2(y[..., 1], y[..., 2])), 360.0)
    ws_pred = np.concatenate(pred_ws)
    sin_pred = np.concatenate(pred_sin)
    cos_pred = np.concatenate(pred_cos)
    wd_pred = np.mod(np.rad2deg(np.arctan2(sin_pred, cos_pred)), 360.0)
    return {
        "x_scaled": x,
        "y_encoded": y,
        "boat_id": np.concatenate(boats),
        "ws_true": ws_true,
        "ws_pred": ws_pred,
        "wd_true": wd_true,
        "wd_pred": wd_pred,
    }


def circular_error_deg(pred, true):
    return np.abs((pred - true + 180.0) % 360.0 - 180.0)


def metrics_from_predictions(payload, mask=None):
    if mask is None:
        mask = np.ones_like(payload["ws_true"], dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    count = int(mask.sum())
    if count == 0:
        return {"count": 0, "ws_rmse": np.nan, "ws_mae": np.nan, "wd_mae": np.nan, "r2": np.nan}
    true = payload["ws_true"][mask]
    pred = payload["ws_pred"][mask]
    sq = np.sum((pred - true) ** 2)
    variance = np.sum((true - np.mean(true)) ** 2)
    return {
        "count": count,
        "ws_rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
        "ws_mae": float(np.mean(np.abs(pred - true))),
        "wd_mae": float(np.mean(circular_error_deg(payload["wd_pred"][mask], payload["wd_true"][mask]))),
        "r2": float(1.0 - sq / variance) if variance > 1e-12 else np.nan,
    }


def train_transfer_variant(
    variant: str,
    model_kind: str,
    seed: int,
    output_root: Path,
    criterion,
    bound_u: float = 4.0,
    bound_v: float = 4.0,
):
    seed_everything(seed)
    source_train, source_val = standard_source_datasets()
    scaler = source_train.scaler
    run_root = output_root / "runs" / variant / f"seed_{seed}"
    checkpoint_root = run_root / "checkpoints"
    log_path = run_root / "training_log.jsonl"
    model_factory = lambda: build_model(model_kind, scaler, bound_u=bound_u, bound_v=bound_v)
    pretrain_model = model_factory()
    parameter_count = count_parameters(pretrain_model)
    pretrain_path = checkpoint_root / "transfer_pretrain_source8_best.pth"
    stages = []
    stages.append(
        fit_stage(
            pretrain_model,
            f"{variant}-pretrain-seed{seed}",
            source_train,
            source_val,
            criterion,
            PRETRAIN_EPOCHS,
            PRETRAIN_LR,
            pretrain_path,
            log_path,
            {"variant": variant, "seed": seed, "stage": "pretrain", "source_boats": SOURCE_BOAT_FILES},
        )
    )
    rows = []
    for target_file in TARGET_BOAT_FILES:
        target = boat_tag(target_file)
        datasets = standard_target_datasets(target_file, scaler)
        model = model_factory()
        model.load_state_dict(torch.load(pretrain_path, map_location=DEVICE)["model_state_dict"])
        target_path = checkpoint_root / f"transfer_{target}_best.pth"
        stages.append(
            fit_stage(
                model,
                f"{variant}-finetune-{target}-seed{seed}",
                datasets["train"],
                datasets["val"],
                criterion,
                FINETUNE_EPOCHS,
                FINETUNE_LR,
                target_path,
                log_path,
                {
                    "variant": variant,
                    "seed": seed,
                    "stage": "finetune",
                    "target": target,
                    "source_checkpoint": str(pretrain_path),
                },
            )
        )
        payload = collect_predictions(model, datasets["test"])
        prediction_path = run_root / f"predictions_{target}.npz"
        np.savez_compressed(prediction_path, **payload)
        metrics = metrics_from_predictions(payload)
        bound_values = model.current_bounds() if isinstance(model, ResidualBoundModel) else (4.0, 4.0)
        rows.append(
            {
                "variant": variant,
                "seed": seed,
                "target": target,
                "checkpoint": str(target_path),
                "prediction_path": str(prediction_path),
                "parameter_count": parameter_count,
                "learned_bound_u": bound_values[0],
                "learned_bound_v": bound_values[1],
                **metrics,
            }
        )
    runtime = float(sum(stage["runtime_seconds"] for stage in stages))
    if model_kind == "unbounded":
        configured_bound_u = None
        configured_bound_v = None
        bound_mechanism = "unbounded linear residual output"
    elif model_kind == "fixed6":
        configured_bound_u = 6.0
        configured_bound_v = 6.0
        bound_mechanism = "fixed component-wise tanh bound"
    elif model_kind == "training_derived":
        configured_bound_u = float(bound_u)
        configured_bound_v = float(bound_v)
        bound_mechanism = "source-training-derived fixed component-wise tanh bound"
    elif model_kind == "learnable":
        configured_bound_u = float(bound_u)
        configured_bound_v = float(bound_v)
        bound_mechanism = "positive learnable component-wise tanh bound (values are initial bounds)"
    else:
        configured_bound_u = 4.0
        configured_bound_v = 4.0
        bound_mechanism = "submitted fixed component-wise tanh bound"

    config = base_configuration()
    config.update(
        {
            "variant": variant,
            "model_kind": model_kind,
            "seed": seed,
            "bound_mechanism": bound_mechanism,
            "bound_u": configured_bound_u,
            "bound_v": configured_bound_v,
            "parameter_count": parameter_count,
            "runtime_seconds": runtime,
            "python_executable": sys.executable,
            "exact_command": " ".join([f'"{sys.executable}"', *sys.argv]),
            "checkpoint_root": str(checkpoint_root),
        }
    )
    (run_root / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(run_root / "target_metrics.csv", index=False)
    return rows


def base_configuration():
    return {
        "input_dimensions": 10,
        "input_angle_representation": "COG/HDG degrees with source-training-fitted z-score",
        "sequence_length": SEQ_LEN,
        "prediction_length": PRED_LEN,
        "hidden_dim": HIDDEN_DIM,
        "state_dimension": D_STATE,
        "mamba_layers": MAMBA_LAYERS,
        "lambda_d": 1.0,
        "batch_size": BATCH_SIZE,
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "finetune_epochs": FINETUNE_EPOCHS,
        "pretrain_learning_rate": PRETRAIN_LR,
        "finetune_learning_rate": FINETUNE_LR,
        "optimizer": "AdamW",
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR eta_min=1e-6",
        "patience": PATIENCE,
        "gradient_clip_norm": GRAD_MAX_NORM,
        "main_split": "60/20/20 chronological",
        "source_boats": SOURCE_BOAT_FILES,
        "target_boats": TARGET_BOAT_FILES,
        "mamba_backend": "custom",
        "device": str(DEVICE),
        "torch_version": torch.__version__,
    }


def locate_checkpoint(relative_path: Path) -> Path:
    for root in BACKUP_ROOTS:
        candidate = root / relative_path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Checkpoint not found in backup roots: {relative_path}")


def main_checkpoint(seed: int, target: str) -> Path:
    return locate_checkpoint(
        Path("weights") / "transfer_two_targets_v1" / "edgewind" / f"rev4_seed_{seed}" / f"transfer_{target}_best.pth"
    )


def no_fft_checkpoint(seed: int, target: str) -> Path:
    return locate_checkpoint(
        Path("weights")
        / "transfer_two_targets_v1"
        / "ablation"
        / f"rev6_fresh_seed_{seed}"
        / "wo_fft"
        / f"transfer_{target}_best.pth"
    )


def loss_checkpoint(seed: int, loss_key: str, target: str) -> Path:
    return locate_checkpoint(
        Path("weights")
        / "transfer_two_targets_v1"
        / "loss_ablation"
        / f"rev4_seed_{seed}"
        / loss_key
        / f"transfer_{target}_best.pth"
    )


def load_model_checkpoint(model, path: Path):
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def evaluate_existing_checkpoint(model_kind: str, checkpoint: Path, target_file: str, scaler):
    model = build_model(model_kind, scaler)
    load_model_checkpoint(model, checkpoint)
    dataset = standard_target_datasets(target_file, scaler)["test"]
    payload = collect_predictions(model, dataset)
    return model, dataset, payload, metrics_from_predictions(payload)


def inverse_inputs(payload, scaler):
    shape = payload["x_scaled"].shape
    raw = scaler.inverse_transform(payload["x_scaled"].reshape(-1, shape[-1]))
    return raw.reshape(shape)


def _frequency_ratio(windows_uv):
    spectrum = np.fft.rfft(windows_uv, axis=1, norm="ortho")
    energy = np.abs(spectrum) ** 2
    high = energy[:, 10:19, :].sum(axis=(1, 2))
    nonzero = energy[:, 1:19, :].sum(axis=(1, 2))
    return np.divide(high, nonzero, out=np.zeros_like(high), where=nonzero > 1e-12)


def source_training_statistics(source_train=None):
    if source_train is None:
        source_train, _ = standard_source_datasets()
    scaler = source_train.scaler
    ws_points = np.concatenate([values[:, 0] for values in source_train.y_data])
    gust_points = []
    delta_u_values, delta_v_values, wd_change_values, hf_values = [], [], [], []
    for x_scaled, y in zip(source_train.x_data, source_train.y_data):
        raw_x = scaler.inverse_transform(x_scaled)
        gust_points.append(raw_x[:, 7])
        count = len(raw_x) - SEQ_LEN - PRED_LEN + 1
        windows_uv = np.stack([raw_x[index : index + SEQ_LEN, 5:7] for index in range(count)])
        hf_values.append(_frequency_ratio(windows_uv))
        base_u = raw_x[np.arange(count) + SEQ_LEN - 1, 5][:, None]
        base_v = raw_x[np.arange(count) + SEQ_LEN - 1, 6][:, None]
        future = np.stack([y[index + SEQ_LEN : index + SEQ_LEN + PRED_LEN] for index in range(count)])
        future_ws = future[..., 0]
        future_wd = np.mod(np.rad2deg(np.arctan2(future[..., 1], future[..., 2])), 360.0)
        future_rad = np.deg2rad(future_wd)
        future_u = -future_ws * np.sin(future_rad)
        future_v = -future_ws * np.cos(future_rad)
        base_wd = np.mod(np.rad2deg(np.arctan2(base_u, base_v)) + 180.0, 360.0)
        delta_u_values.append(np.abs(future_u - base_u).ravel())
        delta_v_values.append(np.abs(future_v - base_v).ravel())
        wd_change_values.append(circular_error_deg(future_wd, base_wd).ravel())
    hf = np.concatenate(hf_values)
    result = {
        "source_training_ws_p95": float(np.percentile(ws_points, 95)),
        "source_training_gust_p95": float(np.percentile(np.concatenate(gust_points), 95)),
        "source_training_abs_delta_u_p99": float(np.percentile(np.concatenate(delta_u_values), 99)),
        "source_training_abs_delta_v_p99": float(np.percentile(np.concatenate(delta_v_values), 99)),
        "source_training_circular_wd_change_p95": float(np.percentile(np.concatenate(wd_change_values), 95)),
        "source_training_hf_q25": float(np.percentile(hf, 25)),
        "source_training_hf_q50": float(np.percentile(hf, 50)),
        "source_training_hf_q75": float(np.percentile(hf, 75)),
        "source_training_ws_point_count": int(ws_points.size),
        "source_training_gust_point_count": int(sum(values.size for values in gust_points)),
        "source_training_forecast_point_count": int(sum(values.size for values in delta_u_values)),
        "source_training_window_count": int(hf.size),
    }
    return result


def regime_masks(payload, scaler, thresholds):
    raw_x = inverse_inputs(payload, scaler)
    hf = _frequency_ratio(raw_x[..., 5:7])
    hf_edges = [thresholds["source_training_hf_q25"], thresholds["source_training_hf_q50"], thresholds["source_training_hf_q75"]]
    hf_bin = np.digitize(hf, hf_edges, right=True) + 1
    gust_sample = np.max(raw_x[..., 7], axis=1) > thresholds["source_training_gust_p95"]
    ws_true = payload["ws_true"]
    wd_true = payload["wd_true"]
    base_u = raw_x[:, -1, 5][:, None]
    base_v = raw_x[:, -1, 6][:, None]
    radians = np.deg2rad(wd_true)
    future_u = -ws_true * np.sin(radians)
    future_v = -ws_true * np.cos(radians)
    base_wd = np.mod(np.rad2deg(np.arctan2(base_u, base_v)) + 180.0, 360.0)
    return {
        "upper_tail": ws_true > thresholds["source_training_ws_p95"],
        "non_upper_tail": ws_true <= thresholds["source_training_ws_p95"],
        "rapid_component_change": (np.abs(future_u - base_u) > 4.0) | (np.abs(future_v - base_v) > 4.0),
        "strong_gust": np.broadcast_to(gust_sample[:, None], ws_true.shape),
        "non_strong_gust": np.broadcast_to((~gust_sample)[:, None], ws_true.shape),
        "rapid_wd_change": circular_error_deg(wd_true, base_wd) > thresholds["source_training_circular_wd_change_p95"],
        "hf_q1": np.broadcast_to((hf_bin == 1)[:, None], ws_true.shape),
        "hf_q2": np.broadcast_to((hf_bin == 2)[:, None], ws_true.shape),
        "hf_q3": np.broadcast_to((hf_bin == 3)[:, None], ws_true.shape),
        "hf_q4": np.broadcast_to((hf_bin == 4)[:, None], ws_true.shape),
        "hf_ratio": hf,
        "gust_sample": gust_sample,
    }


def add_target_means(frame: pd.DataFrame, group_columns: List[str], metric_columns: List[str]):
    target_rows = frame[frame["target"].isin(["sd1042", "sd1091"])]
    means = target_rows.groupby(group_columns, as_index=False)[metric_columns].mean()
    means["target"] = "target_mean"
    return pd.concat([frame, means], ignore_index=True)


def five_seed_summary(frame: pd.DataFrame, group_columns: List[str], metric_columns: List[str]):
    grouped = frame.groupby(group_columns)[metric_columns]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).add_suffix("_sample_std")
    return mean.join(std).reset_index()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_code_snapshot(output_path: Path, extra_files: Optional[Iterable[Path]] = None):
    files = [PROJECT_ROOT / "model.py", PROJECT_ROOT / "data_provider.py", PROJECT_ROOT / "experiment_config.py", Path(__file__)]
    files.extend(list(extra_files or []))
    payload = {str(path.resolve()): file_sha256(path) for path in files if path.exists()}
    payload["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
