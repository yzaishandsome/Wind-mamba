import math
import os
import random
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

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

SHOW_PROGRESS = sys.stderr.isatty()
VARIANT_FILTER = {
    item.strip().lower()
    for item in os.getenv("EDGEWIND_ABLATION_VARIANTS", "").split(",")
    if item.strip()
}

WEIGHTS_ROOT = Path("weights") / EXPERIMENT_VERSION / "ablation"
OUTPUT_ROOT = Path("comparison_outputs") / EXPERIMENT_VERSION / "ablation"
if OUTPUT_SUFFIX:
    WEIGHTS_ROOT = WEIGHTS_ROOT / OUTPUT_SUFFIX
    OUTPUT_ROOT = OUTPUT_ROOT / OUTPUT_SUFFIX
SUMMARY_PATH = OUTPUT_ROOT / "ablation_summary.csv"
PAPER_TABLE_CSV = OUTPUT_ROOT / "ablation_table.csv"
PAPER_TABLE_PNG = OUTPUT_ROOT / "ablation_table.png"
MEAN_TABLE_CSV = OUTPUT_ROOT / "ablation_mean_table.csv"
MEAN_TABLE_PNG = OUTPUT_ROOT / "ablation_mean_table.png"
TRANSFER_REF_CSV = OUTPUT_ROOT / "transfer_reference_table.csv"
TRANSFER_REF_PNG = OUTPUT_ROOT / "transfer_reference_table.png"

WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def seed_everything(seed=42):
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


class EdgeWindNoFFTModel(EdgeWind_Mamba_Model):
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


class EdgeWindNoPersistenceModel(EdgeWind_Mamba_Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_uv_value = 15.0

    def forward(self, x, boat_id):
        batch_size = x.size(0)

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

        pred_u = self.max_uv_value * torch.tanh(self.u_delta_head(h_state))
        pred_v = self.max_uv_value * torch.tanh(self.v_delta_head(h_state))
        ws_pred, pred_angle = self._vector_to_ws_angle(pred_u.squeeze(-1), pred_v.squeeze(-1))

        ws_pred = torch.clamp(ws_pred.unsqueeze(-1), min=0.0)
        pred_angle = pred_angle.unsqueeze(-1)
        return ws_pred, torch.sin(pred_angle), torch.cos(pred_angle)


class EdgeWindNoBoatEmbeddingModel(EdgeWind_Mamba_Model):
    def forward(self, x, boat_id):
        del boat_id
        batch_size = x.size(0)
        base_u, base_v = self._last_wind_vector(x)

        h_input = self.input_proj(x)
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

        u_delta = self.max_uv_delta * torch.tanh(self.u_delta_head(h_state))
        v_delta = self.max_uv_delta * torch.tanh(self.v_delta_head(h_state))

        pred_u = base_u.unsqueeze(1).unsqueeze(-1) + u_delta
        pred_v = base_v.unsqueeze(1).unsqueeze(-1) + v_delta
        ws_pred, pred_angle = self._vector_to_ws_angle(pred_u.squeeze(-1), pred_v.squeeze(-1))

        ws_pred = torch.clamp(ws_pred.unsqueeze(-1), min=0.0)
        pred_angle = pred_angle.unsqueeze(-1)
        return ws_pred, torch.sin(pred_angle), torch.cos(pred_angle)


class EdgeWindNoGatedFusionModel(EdgeWind_Mamba_Model):
    def forward(self, x, boat_id):
        batch_size = x.size(0)
        base_u, base_v = self._last_wind_vector(x)

        h_input = self.input_proj(x) + self.boat_embedding(boat_id).unsqueeze(1)
        h_m = self.mamba_backbone(h_input)
        h_f = self.turbulence_extractor(x)

        h_combined = torch.cat([h_m, h_f], dim=-1)
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


class EdgeWindMLPDecoderModel(EdgeWind_Mamba_Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp_decoder = torch.nn.Sequential(
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
        )

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
        h_state = self.mlp_decoder(decoder_input)

        u_delta = self.max_uv_delta * torch.tanh(self.u_delta_head(h_state))
        v_delta = self.max_uv_delta * torch.tanh(self.v_delta_head(h_state))

        pred_u = base_u.unsqueeze(1).unsqueeze(-1) + u_delta
        pred_v = base_v.unsqueeze(1).unsqueeze(-1) + v_delta
        ws_pred, pred_angle = self._vector_to_ws_angle(pred_u.squeeze(-1), pred_v.squeeze(-1))

        ws_pred = torch.clamp(ws_pred.unsqueeze(-1), min=0.0)
        pred_angle = pred_angle.unsqueeze(-1)
        return ws_pred, torch.sin(pred_angle), torch.cos(pred_angle)


def build_full_model(feature_stats):
    return EdgeWind_Mamba_Model(
        in_dim=10,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        hidden_dim=HIDDEN_DIM,
        num_boats=len(ALL_BOAT_FILES),
        feature_stats=feature_stats,
    ).to(DEVICE)


def build_no_fft_model(feature_stats):
    return EdgeWindNoFFTModel(
        in_dim=10,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        hidden_dim=HIDDEN_DIM,
        num_boats=len(ALL_BOAT_FILES),
        feature_stats=feature_stats,
    ).to(DEVICE)


def build_no_persistence_model(feature_stats):
    return EdgeWindNoPersistenceModel(
        in_dim=10,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        hidden_dim=HIDDEN_DIM,
        num_boats=len(ALL_BOAT_FILES),
        feature_stats=feature_stats,
    ).to(DEVICE)


def build_no_boat_embedding_model(feature_stats):
    return EdgeWindNoBoatEmbeddingModel(
        in_dim=10,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        hidden_dim=HIDDEN_DIM,
        num_boats=len(ALL_BOAT_FILES),
        feature_stats=feature_stats,
    ).to(DEVICE)


def build_no_gated_fusion_model(feature_stats):
    return EdgeWindNoGatedFusionModel(
        in_dim=10,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        hidden_dim=HIDDEN_DIM,
        num_boats=len(ALL_BOAT_FILES),
        feature_stats=feature_stats,
    ).to(DEVICE)


def build_mlp_decoder_model(feature_stats):
    return EdgeWindMLPDecoderModel(
        in_dim=10,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        hidden_dim=HIDDEN_DIM,
        num_boats=len(ALL_BOAT_FILES),
        feature_stats=feature_stats,
    ).to(DEVICE)


VARIANT_CONFIGS = [
    {
        "key": "full",
        "display": "Full EdgeWind-Mamba",
        "model_factory": build_full_model,
    },
    {
        "key": "wo_fft",
        "display": "w/o FFT Branch",
        "model_factory": build_no_fft_model,
    },
    {
        "key": "wo_persistence",
        "display": "w/o Persistence Prior",
        "model_factory": build_no_persistence_model,
    },
    {
        "key": "wo_boat_embedding",
        "display": "w/o Vessel Embedding",
        "model_factory": build_no_boat_embedding_model,
    },
    {
        "key": "wo_gated_fusion",
        "display": "w/o Gated Fusion",
        "model_factory": build_no_gated_fusion_model,
    },
    {
        "key": "mlp_decoder",
        "display": "MLP Decoder",
        "model_factory": build_mlp_decoder_model,
    },
]


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
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_checkpoint_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            torch.save(checkpoint, tmp_checkpoint_path)
            tmp_checkpoint_path.replace(checkpoint_path)
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
        print(f"Reusing existing checkpoint: {checkpoint_path}")
        load_checkpoint_into_model(model, checkpoint_path)
        return

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


def build_long_summary(all_rows):
    summary_df = pd.DataFrame(all_rows)
    target_order = [boat_tag(file_path) for file_path in TARGET_BOAT_FILES]
    summary_df["variant_rank"] = summary_df["variant_key"].map({cfg["key"]: idx for idx, cfg in enumerate(VARIANT_CONFIGS)})
    summary_df["target_rank"] = summary_df["target"].map({tag: idx for idx, tag in enumerate(target_order)})
    summary_df = summary_df.sort_values(["variant_rank", "target_rank"]).drop(columns=["variant_rank", "target_rank"])
    return summary_df.reset_index(drop=True)


def build_paper_table(summary_df):
    rows = []
    target_order = [boat_tag(file_path) for file_path in TARGET_BOAT_FILES]
    for config in VARIANT_CONFIGS:
        variant_df = summary_df[summary_df["variant_key"] == config["key"]]
        row = {"Variant": config["display"]}
        for target in target_order:
            target_df = variant_df[variant_df["target"] == target]
            if target_df.empty:
                continue
            record = target_df.iloc[0]
            target_label = target.upper()
            row[f"{target_label} WS-RMSE"] = record["test_ws_rmse"]
            row[f"{target_label} WS-MAE"] = record["test_ws_mae"]
            row[f"{target_label} WD-MAE"] = record["test_wd_mae"]
            row[f"{target_label} Extreme WS-RMSE"] = record["test_extreme_ws_rmse"]
            row[f"{target_label} R^2"] = record["test_r2"]

        row["Mean WS-RMSE"] = variant_df["test_ws_rmse"].mean()
        row["Mean WS-MAE"] = variant_df["test_ws_mae"].mean()
        row["Mean WD-MAE"] = variant_df["test_wd_mae"].mean()
        row["Mean Extreme WS-RMSE"] = variant_df["test_extreme_ws_rmse"].mean()
        row["Mean R^2"] = variant_df["test_r2"].mean()
        rows.append(row)

    return pd.DataFrame(rows)


def save_table_image(df, output_path, title):
    display_df = df.copy()
    numeric_cols = [col for col in df.columns if col not in {"Variant", "Experiment", "Target"}]
    for col in numeric_cols:
        display_df[col] = display_df[col].map(lambda value: f"{value:.4f}")

    fig_width = max(14, 1.25 * len(display_df.columns))
    fig_height = max(3.8, 0.55 * len(display_df) + 1.8)
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

    for col in numeric_cols:
        col_idx = display_df.columns.get_loc(col)
        if "R^2" in col:
            best_row = df[col].astype(float).idxmax()
        else:
            best_row = df[col].astype(float).idxmin()
        table[(best_row + 1, col_idx)].set_text_props(weight="bold", color="#111111")

    full_rows = display_df.index[display_df.get("Variant", pd.Series(dtype=str)).astype(str).str.contains("Full", regex=False)]
    for row_idx in full_rows:
        for col_idx in range(len(display_df.columns)):
            table[(row_idx + 1, col_idx)].set_facecolor("#fff7f2")

    ax.set_title(title, fontsize=15, fontweight="bold", pad=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_transfer_reference():
    edgewind_root = Path("weights") / EXPERIMENT_VERSION / "edgewind"
    if OUTPUT_SUFFIX:
        edgewind_root = edgewind_root / OUTPUT_SUFFIX
    edgewind_summary_path = edgewind_root / "edgewind_summary.csv"
    if not edgewind_summary_path.exists():
        return

    reference_df = pd.read_csv(edgewind_summary_path)
    reference_df["Variant"] = reference_df["experiment"].map(
        {
            "transfer": "Transfer (reported separately)",
            "no_transfer": "w/o Transfer (reported separately)",
        }
    )
    reference_df["Target"] = reference_df["target"].str.upper()
    reference_df = reference_df.rename(
        columns={
            "test_ws_rmse": "WS-RMSE",
            "test_ws_mae": "WS-MAE",
            "test_wd_mae": "WD-MAE",
        }
    )
    reference_df = reference_df[["Variant", "Target", "WS-RMSE", "WS-MAE", "WD-MAE"]].sort_values(
        ["Target", "Variant"]
    )
    reference_df.to_csv(TRANSFER_REF_CSV, index=False, encoding="utf-8-sig")
    save_table_image(reference_df, TRANSFER_REF_PNG, "Transfer Validation Reference Table")


def train_single_variant(config, source_train, source_val, source_feature_stats, source_scaler):
    variant_key = config["key"]
    variant_name = config["display"]
    criterion = build_loss(
        loss_mode=LOSS_MODE,
        extreme_ws_threshold=EXTREME_WS_THRESHOLD,
        extreme_weight=EXTREME_WEIGHT,
        wd_weight=WD_WEIGHT,
    ).to(DEVICE)
    variant_dir = WEIGHTS_ROOT / variant_key
    variant_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"Ablation Variant | {variant_name}")
    print("=" * 80)

    source_train_loader = make_loader(source_train, shuffle=True)
    source_val_loader = make_loader(source_val, shuffle=False)

    pretrain_model = config["model_factory"](source_feature_stats)
    pretrain_ckpt = variant_dir / "transfer_pretrain_source8_best.pth"
    maybe_train_stage(
        model=pretrain_model,
        checkpoint_path=pretrain_ckpt,
        stage_name=f"{variant_key}-pretrain",
        train_loader=source_train_loader,
        val_loader=source_val_loader,
        criterion=criterion,
        epochs=EPOCHS_TRANSFER_PRETRAIN,
        lr=LR_TRANSFER_PRETRAIN,
        metadata={
            "variant_key": variant_key,
            "variant_name": variant_name,
            "experiment": "transfer",
            "stage": "pretrain",
        },
    )

    rows = []
    for target_file in TARGET_BOAT_FILES:
        tag = boat_tag(target_file)
        target_datasets = build_target_datasets(target_file, source_scaler)
        model = config["model_factory"](source_feature_stats)
        load_checkpoint_into_model(model, pretrain_ckpt)

        finetune_ckpt = variant_dir / f"transfer_{tag}_best.pth"
        maybe_train_stage(
            model=model,
            checkpoint_path=finetune_ckpt,
            stage_name=f"{variant_key}-finetune-{tag}",
            train_loader=make_loader(target_datasets["train"], shuffle=True),
            val_loader=make_loader(target_datasets["val"], shuffle=False),
            criterion=criterion,
            epochs=EPOCHS_TRANSFER_FINETUNE,
            lr=LR_TRANSFER_FINETUNE,
            metadata={
                "variant_key": variant_key,
                "variant_name": variant_name,
                "experiment": "transfer",
                "stage": "finetune",
                "target": tag,
            },
        )

        test_metrics = run_epoch(model, make_loader(target_datasets["test"], shuffle=False), criterion, optimizer=None)
        print(
            f"  {tag.upper()} | WS-RMSE={test_metrics['ws_rmse']:.4f} | "
            f"WS-MAE={test_metrics['ws_mae']:.4f} | "
            f"WD-MAE={test_metrics['wd_mae']:.2f} | "
            f"Extreme-RMSE={test_metrics['extreme_ws_rmse']:.4f} | "
            f"R^2={test_metrics['r2']:.4f}"
        )
        rows.append(
            {
                "seed": SEED,
                "variant_key": variant_key,
                "variant_name": variant_name,
                "target": tag,
                "checkpoint": str(finetune_ckpt),
                "test_ws_rmse": test_metrics["ws_rmse"],
                "test_ws_mae": test_metrics["ws_mae"],
                "test_wd_mae": test_metrics["wd_mae"],
                "test_extreme_ws_rmse": test_metrics["extreme_ws_rmse"],
                "test_r2": test_metrics["r2"],
            }
        )

    return rows


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
    print("Ablation is run under the same transfer protocol as main.py.")
    print("Transfer vs no-transfer is exported separately as a reference table.")

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
    for config in VARIANT_CONFIGS:
        if VARIANT_FILTER and config["key"].lower() not in VARIANT_FILTER and config["display"].lower() not in VARIANT_FILTER:
            continue
        all_rows.extend(train_single_variant(config, source_train, source_val, source_feature_stats, source_scaler))

    summary_df = build_long_summary(all_rows)
    summary_df.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    paper_table = build_paper_table(summary_df)
    paper_table.to_csv(PAPER_TABLE_CSV, index=False, encoding="utf-8-sig")
    save_table_image(paper_table, PAPER_TABLE_PNG, "EdgeWind-Mamba Ablation Table")

    mean_table = paper_table[
        ["Variant", "Mean WS-RMSE", "Mean WS-MAE", "Mean WD-MAE", "Mean Extreme WS-RMSE", "Mean R^2"]
    ].copy()
    mean_table = mean_table.sort_values("Mean WS-RMSE").reset_index(drop=True)
    mean_table.to_csv(MEAN_TABLE_CSV, index=False, encoding="utf-8-sig")
    save_table_image(mean_table, MEAN_TABLE_PNG, "EdgeWind-Mamba Ablation Mean Metrics Table")

    save_transfer_reference()

    print("\nSaved ablation outputs:")
    print(f"  Weights: {WEIGHTS_ROOT.resolve()}")
    print(f"  Summary: {SUMMARY_PATH.resolve()}")
    print(f"  Table CSV: {PAPER_TABLE_CSV.resolve()}")
    print(f"  Table PNG: {PAPER_TABLE_PNG.resolve()}")
    print(f"  Mean CSV: {MEAN_TABLE_CSV.resolve()}")
    print(f"  Mean PNG: {MEAN_TABLE_PNG.resolve()}")
    if TRANSFER_REF_CSV.exists():
        print(f"  Transfer reference: {TRANSFER_REF_CSV.resolve()}")


if __name__ == "__main__":
    main()
