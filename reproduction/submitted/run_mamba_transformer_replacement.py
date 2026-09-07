import math
import os
import random
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from baselines import LearnablePositionalEncoding
from data_provider import USVDataset
from experiment_config import (
    ALL_BOAT_FILES,
    EXPERIMENT_VERSION,
    SOURCE_BOAT_FILES,
    TARGET_BOAT_FILES,
    boat_ids_for,
    boat_tag,
)
from model import DEFAULT_HIGH_WIND_THRESHOLD, EdgeWind_Mamba_Model, build_loss


warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_CPU = torch.device("cpu")
DEVICE_GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = int(os.getenv("EDGEWIND_SEED", "42"))
OUTPUT_SUFFIX = os.getenv("EDGEWIND_OUTPUT_SUFFIX", "").strip()
BATCH_SIZE = 32
SEQ_LEN = int(os.getenv("EDGEWIND_SEQ_LEN", "36"))
PRED_LEN = int(os.getenv("EDGEWIND_PRED_LEN", "6"))
HIDDEN_DIM = int(os.getenv("EDGEWIND_HIDDEN_DIM", "96"))
LOSS_MODE = os.getenv("EDGEWIND_LOSS_MODE", "smoothl1_dircos").strip().lower()
EXTREME_WEIGHT = float(os.getenv("EDGEWIND_EXTREME_WEIGHT", "2.0"))
WD_WEIGHT = float(os.getenv("EDGEWIND_WD_WEIGHT", "1.0"))

EPOCHS_TRANSFER_PRETRAIN = 60
EPOCHS_TRANSFER_FINETUNE = 20
LR_TRANSFER_PRETRAIN = 2e-4
LR_TRANSFER_FINETUNE = 5e-5
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 8
GRAD_MAX_NORM = 1.0
EXTREME_WS_THRESHOLD = float(os.getenv("EDGEWIND_EXTREME_WS_THRESHOLD", f"{DEFAULT_HIGH_WIND_THRESHOLD}"))

LATENCY_BATCH_SIZE = 16
LATENCY_WARMUP_STEPS = 5
LATENCY_MEASURE_STEPS = 20

SHOW_PROGRESS = sys.stderr.isatty()

WEIGHTS_ROOT = Path("weights") / EXPERIMENT_VERSION / "mamba_transformer_replacement"
OUTPUT_ROOT = Path("comparison_outputs") / EXPERIMENT_VERSION / "mamba_transformer_replacement"
if OUTPUT_SUFFIX:
    WEIGHTS_ROOT = WEIGHTS_ROOT / OUTPUT_SUFFIX
    OUTPUT_ROOT = OUTPUT_ROOT / OUTPUT_SUFFIX

SUMMARY_CSV = OUTPUT_ROOT / "mamba_transformer_replacement_summary.csv"
MEAN_CSV = OUTPUT_ROOT / "mamba_transformer_replacement_mean.csv"
MEAN_PNG = OUTPUT_ROOT / "mamba_transformer_replacement_mean.png"
EFFICIENCY_CSV = OUTPUT_ROOT / "mamba_transformer_replacement_efficiency.csv"
EFFICIENCY_PNG = OUTPUT_ROOT / "mamba_transformer_replacement_efficiency.png"

WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


class TemporalTransformerBackbone(nn.Module):
    def __init__(self, hidden_dim, seq_len, num_layers=3, dropout=0.2):
        super().__init__()
        self.pos_encoding = LearnablePositionalEncoding(seq_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.encoder(self.pos_encoding(x))


class EdgeWindTransformerSwapModel(EdgeWind_Mamba_Model):
    def __init__(self, *args, n_transformer_layers=3, **kwargs):
        seq_len = kwargs.get("seq_len", SEQ_LEN)
        hidden_dim = kwargs.get("hidden_dim", HIDDEN_DIM)
        super().__init__(*args, **kwargs)
        self.temporal_backbone = TemporalTransformerBackbone(
            hidden_dim=hidden_dim,
            seq_len=seq_len,
            num_layers=n_transformer_layers,
            dropout=0.2,
        )

    def forward(self, x, boat_id):
        batch_size = x.size(0)
        base_u, base_v = self._last_wind_vector(x)

        h_input = self.input_proj(x) + self.boat_embedding(boat_id).unsqueeze(1)
        h_m = self.temporal_backbone(h_input)
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

        u_delta = self.max_uv_delta * torch.tanh(self.u_delta_head(h_state))
        v_delta = self.max_uv_delta * torch.tanh(self.v_delta_head(h_state))

        pred_u = base_u.unsqueeze(1).unsqueeze(-1) + u_delta
        pred_v = base_v.unsqueeze(1).unsqueeze(-1) + v_delta
        ws_pred, pred_angle = self._vector_to_ws_angle(pred_u.squeeze(-1), pred_v.squeeze(-1))

        ws_pred = torch.clamp(ws_pred.unsqueeze(-1), min=0.0)
        pred_angle = pred_angle.unsqueeze(-1)
        return ws_pred, torch.sin(pred_angle), torch.cos(pred_angle)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_feature_stats(scaler):
    return {
        "u_mean": scaler.mean_[5],
        "u_scale": scaler.scale_[5],
        "v_mean": scaler.mean_[6],
        "v_scale": scaler.scale_[6],
    }


def build_mamba_model(feature_stats):
    return EdgeWind_Mamba_Model(
        in_dim=10,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        hidden_dim=HIDDEN_DIM,
        num_boats=len(ALL_BOAT_FILES),
        feature_stats=feature_stats,
    ).to(DEVICE)


def build_transformer_swap_model(feature_stats):
    return EdgeWindTransformerSwapModel(
        in_dim=10,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        hidden_dim=HIDDEN_DIM,
        num_boats=len(ALL_BOAT_FILES),
        feature_stats=feature_stats,
    ).to(DEVICE)


def make_loader(dataset, shuffle):
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


def calculate_score(ws_rmse, wd_mae):
    return 0.7 * ws_rmse + 0.3 * (wd_mae / 100.0)


def compute_batch_statistics(ws_pred, wd_sin_pred, wd_cos_pred, ws_true, wd_sin_true, wd_cos_true):
    ws_sq_error = torch.sum((ws_pred - ws_true) ** 2).item()
    ws_abs_error = torch.sum(torch.abs(ws_pred - ws_true)).item()
    ws_true_sum = torch.sum(ws_true).item()
    ws_true_sq_sum = torch.sum(ws_true ** 2).item()
    target_count = ws_true.numel()

    wd_pred_deg = (torch.atan2(wd_sin_pred, wd_cos_pred) * 180.0 / math.pi + 360.0) % 360.0
    wd_true_deg = (torch.atan2(wd_sin_true, wd_cos_true) * 180.0 / math.pi + 360.0) % 360.0
    wd_diff = torch.abs(wd_pred_deg - wd_true_deg)
    wd_err = torch.min(wd_diff, 360.0 - wd_diff)
    wd_mae_sum = torch.sum(wd_err).item()

    extreme_mask = ws_true > EXTREME_WS_THRESHOLD
    if torch.any(extreme_mask):
        extreme_sq_error = torch.sum((ws_pred[extreme_mask] - ws_true[extreme_mask]) ** 2).item()
        extreme_count = int(extreme_mask.sum().item())
    else:
        extreme_sq_error = 0.0
        extreme_count = 0

    return {
        "ws_sq_error": ws_sq_error,
        "ws_abs_error": ws_abs_error,
        "ws_true_sum": ws_true_sum,
        "ws_true_sq_sum": ws_true_sq_sum,
        "wd_mae_sum": wd_mae_sum,
        "target_count": target_count,
        "extreme_sq_error": extreme_sq_error,
        "extreme_count": extreme_count,
    }


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_count = 0
    total_ws_sq_error = 0.0
    total_ws_abs_error = 0.0
    total_ws_true_sum = 0.0
    total_ws_true_sq_sum = 0.0
    total_wd_mae_sum = 0.0
    total_extreme_sq_error = 0.0
    total_extreme_count = 0

    loop = tqdm(loader, leave=False, disable=not SHOW_PROGRESS)
    for batch_x, batch_y, boat_id in loop:
        batch_x = batch_x.to(DEVICE).float()
        batch_y = batch_y.to(DEVICE).float()
        boat_id = boat_id.to(DEVICE).long()

        ws_true = batch_y[..., 0:1]
        wd_sin_true = batch_y[..., 1:2]
        wd_cos_true = batch_y[..., 2:3]

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            ws_pred, wd_sin_pred, wd_cos_pred = model(batch_x, boat_id)
            loss, _, _ = criterion(ws_pred, wd_sin_pred, wd_cos_pred, ws_true, wd_sin_true, wd_cos_true)
            if not torch.isfinite(loss):
                continue
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_MAX_NORM)
                optimizer.step()

        stats = compute_batch_statistics(
            ws_pred.detach(),
            wd_sin_pred.detach(),
            wd_cos_pred.detach(),
            ws_true.detach(),
            wd_sin_true.detach(),
            wd_cos_true.detach(),
        )

        batch_size = batch_x.size(0)
        total_loss += loss.item() * batch_size
        total_count += stats["target_count"]
        total_ws_sq_error += stats["ws_sq_error"]
        total_ws_abs_error += stats["ws_abs_error"]
        total_ws_true_sum += stats["ws_true_sum"]
        total_ws_true_sq_sum += stats["ws_true_sq_sum"]
        total_wd_mae_sum += stats["wd_mae_sum"]
        total_extreme_sq_error += stats["extreme_sq_error"]
        total_extreme_count += stats["extreme_count"]

        if SHOW_PROGRESS:
            loop.set_postfix(loss=f"{loss.item():.4f}")

    if total_count == 0:
        return {
            "loss": float("inf"),
            "ws_rmse": float("inf"),
            "ws_mae": float("inf"),
            "wd_mae": float("inf"),
            "extreme_ws_rmse": float("inf"),
            "r2": float("nan"),
        }

    ws_mean = total_ws_true_sum / total_count
    ws_total_var = total_ws_true_sq_sum - total_count * (ws_mean ** 2)
    r2 = 1.0 - (total_ws_sq_error / ws_total_var) if ws_total_var > 1e-12 else float("nan")

    return {
        "loss": total_loss / max(total_count / PRED_LEN, 1),
        "ws_rmse": math.sqrt(total_ws_sq_error / total_count),
        "ws_mae": total_ws_abs_error / total_count,
        "wd_mae": total_wd_mae_sum / total_count,
        "extreme_ws_rmse": math.sqrt(total_extreme_sq_error / total_extreme_count) if total_extreme_count > 0 else 0.0,
        "r2": r2,
    }


def load_checkpoint_into_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def fit_stage(model, stage_name, train_loader, val_loader, optimizer, scheduler, criterion, epochs, checkpoint_path, metadata):
    best_score = float("inf")
    best_state = None
    epochs_without_improve = 0

    for epoch in range(1, epochs + 1):
        print(f"\n[{stage_name}] Epoch {epoch}/{epochs}")
        train_metrics = run_epoch(model, train_loader, criterion, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer=None)

        if scheduler is not None:
            scheduler.step()

        current_score = calculate_score(val_metrics["ws_rmse"], val_metrics["wd_mae"])
        print(
            f"train loss={train_metrics['loss']:.4f}, "
            f"val ws_rmse={val_metrics['ws_rmse']:.4f}, "
            f"val wd_mae={val_metrics['wd_mae']:.2f}, "
            f"val extreme_rmse={val_metrics['extreme_ws_rmse']:.4f}, "
            f"val r2={val_metrics['r2']:.4f}, "
            f"val score={current_score:.4f}"
        )

        if current_score < best_score:
            best_score = current_score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            checkpoint = {
                "model_state_dict": best_state,
                "best_score": best_score,
                "val_metrics": val_metrics,
            }
            checkpoint.update(metadata)
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved improved checkpoint to {checkpoint_path}")
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= EARLY_STOPPING_PATIENCE:
                print(f"[{stage_name}] Early stopping triggered.")
                break

    if best_state is None:
        raise RuntimeError(f"{stage_name} failed to produce a valid checkpoint.")
    model.load_state_dict(best_state)


def maybe_train_stage(model, checkpoint_path, stage_name, train_loader, val_loader, criterion, epochs, lr, metadata):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        try:
            print(f"Reusing existing checkpoint: {checkpoint_path}")
            load_checkpoint_into_model(model, checkpoint_path)
            return
        except RuntimeError as exc:
            backup_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_incompatible{checkpoint_path.suffix}")
            if backup_path.exists():
                backup_path.unlink()
            checkpoint_path.replace(backup_path)
            print(
                f"Incompatible checkpoint detected for {stage_name}: {checkpoint_path}\n"
                f"Moved old checkpoint to: {backup_path}\n"
                f"Retrying this stage from scratch.\n"
                f"Load error: {exc}"
            )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    fit_stage(
        model=model,
        stage_name=stage_name,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        epochs=epochs,
        checkpoint_path=checkpoint_path,
        metadata=metadata,
    )


def build_target_datasets(target_file, scaler):
    target_id = boat_ids_for([target_file])[0]
    return {
        "train": USVDataset(
            [target_file],
            global_boat_ids=[target_id],
            flag="train",
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN,
            scaler=scaler,
        ),
        "val": USVDataset(
            [target_file],
            global_boat_ids=[target_id],
            flag="val",
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN,
            scaler=scaler,
        ),
        "test": USVDataset(
            [target_file],
            global_boat_ids=[target_id],
            flag="test",
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN,
            scaler=scaler,
        ),
    }


def save_table_image(df, output_path, title):
    display_df = df.copy()
    numeric_cols = [col for col in display_df.columns if col not in {"Variant", "Model", "Target"}]
    for col in numeric_cols:
        display_df[col] = display_df[col].map(lambda value: f"{value:.4f}" if pd.notna(value) else "N/A")

    fig_width = max(10, 1.15 * len(display_df.columns))
    fig_height = max(3.4, 0.55 * len(display_df) + 1.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.5)

    for col_idx in range(len(display_df.columns)):
        table[(0, col_idx)].set_text_props(weight="bold")
        table[(0, col_idx)].set_facecolor("#f2f2f2")

    ax.set_title(title, fontsize=15, fontweight="bold", pad=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def train_mamba_reference(source_train, source_val, source_feature_stats, source_scaler, criterion):
    rows = []
    variant_dir = WEIGHTS_ROOT / "mamba_reference"
    variant_dir.mkdir(parents=True, exist_ok=True)

    pretrain_model = build_mamba_model(source_feature_stats)
    pretrain_ckpt = variant_dir / "transfer_pretrain_source8_best.pth"
    maybe_train_stage(
        model=pretrain_model,
        checkpoint_path=pretrain_ckpt,
        stage_name="mamba-reference-pretrain",
        train_loader=make_loader(source_train, shuffle=True),
        val_loader=make_loader(source_val, shuffle=False),
        criterion=criterion,
        epochs=EPOCHS_TRANSFER_PRETRAIN,
        lr=LR_TRANSFER_PRETRAIN,
        metadata={
            "variant_key": "mamba",
            "experiment": "transfer",
            "stage": "pretrain",
        },
    )

    for target_file in TARGET_BOAT_FILES:
        tag = boat_tag(target_file)
        datasets = build_target_datasets(target_file, source_scaler)
        model = build_mamba_model(source_feature_stats)
        load_checkpoint_into_model(model, pretrain_ckpt)

        finetune_ckpt = variant_dir / f"transfer_{tag}_best.pth"
        maybe_train_stage(
            model=model,
            checkpoint_path=finetune_ckpt,
            stage_name=f"mamba-reference-finetune-{tag}",
            train_loader=make_loader(datasets["train"], shuffle=True),
            val_loader=make_loader(datasets["val"], shuffle=False),
            criterion=criterion,
            epochs=EPOCHS_TRANSFER_FINETUNE,
            lr=LR_TRANSFER_FINETUNE,
            metadata={
                "variant_key": "mamba",
                "experiment": "transfer",
                "stage": "finetune",
                "target": tag,
            },
        )

        val_metrics = run_epoch(model, make_loader(datasets["val"], shuffle=False), criterion, optimizer=None)
        test_metrics = run_epoch(model, make_loader(datasets["test"], shuffle=False), criterion, optimizer=None)

        rows.append(
            {
                "seed": SEED,
                "variant_key": "mamba",
                "Variant": "EdgeWind-Mamba",
                "target": tag,
                "val_ws_rmse": val_metrics["ws_rmse"],
                "val_ws_mae": val_metrics["ws_mae"],
                "val_wd_mae": val_metrics["wd_mae"],
                "val_extreme_ws_rmse": val_metrics["extreme_ws_rmse"],
                "val_r2": val_metrics["r2"],
                "test_ws_rmse": test_metrics["ws_rmse"],
                "test_ws_mae": test_metrics["ws_mae"],
                "test_wd_mae": test_metrics["wd_mae"],
                "test_extreme_ws_rmse": test_metrics["extreme_ws_rmse"],
                "test_r2": test_metrics["r2"],
                "checkpoint": str(finetune_ckpt),
            }
        )
    return rows


def train_transformer_swap(source_train, source_val, source_feature_stats, source_scaler, criterion):
    rows = []
    variant_dir = WEIGHTS_ROOT / "transformer_swap"
    variant_dir.mkdir(parents=True, exist_ok=True)

    pretrain_model = build_transformer_swap_model(source_feature_stats)
    pretrain_ckpt = variant_dir / "transfer_pretrain_source8_best.pth"
    maybe_train_stage(
        model=pretrain_model,
        checkpoint_path=pretrain_ckpt,
        stage_name="transformer-swap-pretrain",
        train_loader=make_loader(source_train, shuffle=True),
        val_loader=make_loader(source_val, shuffle=False),
        criterion=criterion,
        epochs=EPOCHS_TRANSFER_PRETRAIN,
        lr=LR_TRANSFER_PRETRAIN,
        metadata={
            "variant_key": "transformer_swap",
            "experiment": "transfer",
            "stage": "pretrain",
        },
    )

    for target_file in TARGET_BOAT_FILES:
        tag = boat_tag(target_file)
        datasets = build_target_datasets(target_file, source_scaler)
        model = build_transformer_swap_model(source_feature_stats)
        load_checkpoint_into_model(model, pretrain_ckpt)

        finetune_ckpt = variant_dir / f"transfer_{tag}_best.pth"
        maybe_train_stage(
            model=model,
            checkpoint_path=finetune_ckpt,
            stage_name=f"transformer-swap-finetune-{tag}",
            train_loader=make_loader(datasets["train"], shuffle=True),
            val_loader=make_loader(datasets["val"], shuffle=False),
            criterion=criterion,
            epochs=EPOCHS_TRANSFER_FINETUNE,
            lr=LR_TRANSFER_FINETUNE,
            metadata={
                "variant_key": "transformer_swap",
                "experiment": "transfer",
                "stage": "finetune",
                "target": tag,
            },
        )

        val_metrics = run_epoch(model, make_loader(datasets["val"], shuffle=False), criterion, optimizer=None)
        test_metrics = run_epoch(model, make_loader(datasets["test"], shuffle=False), criterion, optimizer=None)

        rows.append(
            {
                "seed": SEED,
                "variant_key": "transformer_swap",
                "Variant": "EdgeWind-TransformerSwap",
                "target": tag,
                "val_ws_rmse": val_metrics["ws_rmse"],
                "val_ws_mae": val_metrics["ws_mae"],
                "val_wd_mae": val_metrics["wd_mae"],
                "val_extreme_ws_rmse": val_metrics["extreme_ws_rmse"],
                "val_r2": val_metrics["r2"],
                "test_ws_rmse": test_metrics["ws_rmse"],
                "test_ws_mae": test_metrics["ws_mae"],
                "test_wd_mae": test_metrics["wd_mae"],
                "test_extreme_ws_rmse": test_metrics["extreme_ws_rmse"],
                "test_r2": test_metrics["r2"],
                "checkpoint": str(finetune_ckpt),
            }
        )
    return rows


def count_trainable_params(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad) / 1e6


def make_dummy_batch(seq_len, device):
    batch_x = torch.randn(LATENCY_BATCH_SIZE, seq_len, 10, device=device)
    boat_id = torch.zeros(LATENCY_BATCH_SIZE, dtype=torch.long, device=device)
    return batch_x, boat_id


def measure_forward_latency_ms(model, device):
    model = model.to(device)
    model.eval()
    batch_x, boat_id = make_dummy_batch(SEQ_LEN, device)

    with torch.inference_mode():
        for _ in range(LATENCY_WARMUP_STEPS):
            model(batch_x, boat_id)

        if device.type == "cuda":
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(LATENCY_MEASURE_STEPS):
                model(batch_x, boat_id)
            end_event.record()
            torch.cuda.synchronize()
            latency = start_event.elapsed_time(end_event) / LATENCY_MEASURE_STEPS
            del batch_x, boat_id, model
            torch.cuda.empty_cache()
            return latency

        start_time = time.perf_counter()
        for _ in range(LATENCY_MEASURE_STEPS):
            model(batch_x, boat_id)
        elapsed = time.perf_counter() - start_time
        return elapsed * 1000.0 / LATENCY_MEASURE_STEPS


def build_efficiency_snapshot(source_feature_stats):
    rows = []
    specs = {
        "EdgeWind-Mamba": lambda: build_mamba_model(source_feature_stats),
        "EdgeWind-TransformerSwap": lambda: build_transformer_swap_model(source_feature_stats),
    }

    for model_name, factory in specs.items():
        model_for_params = factory()
        params_m = count_trainable_params(model_for_params)
        del model_for_params

        cpu_ms = measure_forward_latency_ms(factory(), DEVICE_CPU)
        gpu_ms = float("nan")
        if DEVICE_GPU.type == "cuda":
            gpu_ms = measure_forward_latency_ms(factory(), DEVICE_GPU)

        rows.append(
            {
                "seed": SEED,
                "Model": model_name,
                "params_m": params_m,
                "cpu_inference_ms": cpu_ms,
                "gpu_inference_ms": gpu_ms,
            }
        )
    return pd.DataFrame(rows)


def build_mean_table(summary_df):
    rows = []
    for variant in ["EdgeWind-Mamba", "EdgeWind-TransformerSwap"]:
        variant_df = summary_df[summary_df["Variant"] == variant]
        rows.append(
            {
                "Variant": variant,
                "Mean Test WS-RMSE": variant_df["test_ws_rmse"].mean(),
                "Mean Test WS-MAE": variant_df["test_ws_mae"].mean(),
                "Mean Test WD-MAE": variant_df["test_wd_mae"].mean(),
            }
        )
    return pd.DataFrame(rows)


def main():
    seed_everything(SEED)
    print(f"Device: {DEVICE}")
    print(f"Experiment version: {EXPERIMENT_VERSION}")
    print(f"Seed: {SEED}")
    if OUTPUT_SUFFIX:
        print(f"Output suffix: {OUTPUT_SUFFIX}")
    print(
        f"Config | seq_len={SEQ_LEN} pred_len={PRED_LEN} hidden_dim={HIDDEN_DIM} "
        f"loss_mode={LOSS_MODE} high_wind_threshold={EXTREME_WS_THRESHOLD:.2f}"
    )
    print("Running targeted replacement ablation: Mamba time-domain branch vs Transformer replacement branch")

    criterion = build_loss(
        loss_mode=LOSS_MODE,
        extreme_ws_threshold=EXTREME_WS_THRESHOLD,
        extreme_weight=EXTREME_WEIGHT,
        wd_weight=WD_WEIGHT,
    ).to(DEVICE)

    source_ids = boat_ids_for(SOURCE_BOAT_FILES)
    source_train = USVDataset(
        SOURCE_BOAT_FILES,
        global_boat_ids=source_ids,
        flag="train",
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
    )
    source_scaler = source_train.scaler
    source_val = USVDataset(
        SOURCE_BOAT_FILES,
        global_boat_ids=source_ids,
        flag="val",
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        scaler=source_scaler,
    )
    source_feature_stats = extract_feature_stats(source_scaler)

    all_rows = []
    all_rows.extend(train_mamba_reference(source_train, source_val, source_feature_stats, source_scaler, criterion))
    all_rows.extend(train_transformer_swap(source_train, source_val, source_feature_stats, source_scaler, criterion))

    summary_df = pd.DataFrame(all_rows).sort_values(["Variant", "target"]).reset_index(drop=True)
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    mean_df = build_mean_table(summary_df)
    mean_df.to_csv(MEAN_CSV, index=False, encoding="utf-8-sig")
    save_table_image(mean_df, MEAN_PNG, "Mamba vs Transformer Replacement Mean Metrics")

    efficiency_df = build_efficiency_snapshot(source_feature_stats)
    efficiency_df.to_csv(EFFICIENCY_CSV, index=False, encoding="utf-8-sig")
    save_table_image(efficiency_df, EFFICIENCY_PNG, "Mamba vs Transformer Replacement Local Efficiency Snapshot")

    print("\nSaved outputs:")
    print(f"  Summary: {SUMMARY_CSV.resolve()}")
    print(f"  Mean: {MEAN_CSV.resolve()}")
    print(f"  Efficiency: {EFFICIENCY_CSV.resolve()}")


if __name__ == "__main__":
    main()
