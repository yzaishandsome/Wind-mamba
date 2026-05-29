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
EPOCHS_NO_TRANSFER = 60

LR_TRANSFER_PRETRAIN = 2e-4
LR_TRANSFER_FINETUNE = 5e-5
LR_NO_TRANSFER = 2e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 8
GRAD_MAX_NORM = 1.0

EXTREME_WS_THRESHOLD = float(os.getenv("EDGEWIND_EXTREME_WS_THRESHOLD", f"{DEFAULT_HIGH_WIND_THRESHOLD}"))
PLOT_SAMPLES = 200

OUTPUT_ROOT = Path("weights") / EXPERIMENT_VERSION / "edgewind"
FIGURE_ROOT = Path("figures") / EXPERIMENT_VERSION / "edgewind"
if OUTPUT_SUFFIX:
    OUTPUT_ROOT = OUTPUT_ROOT / OUTPUT_SUFFIX
    FIGURE_ROOT = FIGURE_ROOT / OUTPUT_SUFFIX
SUMMARY_PATH = OUTPUT_ROOT / "edgewind_summary.csv"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
SHOW_PROGRESS = sys.stderr.isatty()
RUN_NO_TRANSFER = os.getenv("EDGEWIND_RUN_NO_TRANSFER", "1") != "0"


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


def calculate_score(ws_rmse, wd_mae):
    return 0.7 * ws_rmse + 0.3 * (wd_mae / 100.0)


def compute_batch_statistics(ws_pred, wd_sin_pred, wd_cos_pred, ws_true, wd_sin_true, wd_cos_true):
    ws_sq_error = torch.sum((ws_pred - ws_true) ** 2).item()
    ws_abs_error = torch.sum(torch.abs(ws_pred - ws_true)).item()
    ws_true_sum = torch.sum(ws_true).item()
    ws_true_sq_sum = torch.sum(ws_true ** 2).item()
    target_count = ws_true.numel()

    wd_pred_rad = torch.atan2(wd_sin_pred, wd_cos_pred)
    wd_pred_deg = (wd_pred_rad * 180.0 / math.pi + 360.0) % 360.0
    wd_true_rad = torch.atan2(wd_sin_true, wd_cos_true)
    wd_true_deg = (wd_true_rad * 180.0 / math.pi + 360.0) % 360.0
    wd_diff = torch.abs(wd_pred_deg - wd_true_deg)
    wd_mae_sum = torch.sum(torch.min(wd_diff, 360.0 - wd_diff)).item()

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


def build_model(feature_stats):
    return EdgeWind_Mamba_Model(
        in_dim=10,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        hidden_dim=HIDDEN_DIM,
        num_boats=len(ALL_BOAT_FILES),
        feature_stats=feature_stats,
    ).to(DEVICE)


def make_loader(dataset, shuffle):
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


def load_checkpoint_into_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_ws_loss = 0.0
    total_wd_loss = 0.0
    total_sample_count = 0
    total_target_count = 0
    total_ws_sq_error = 0.0
    total_ws_abs_error = 0.0
    total_ws_true_sum = 0.0
    total_ws_true_sq_sum = 0.0
    total_wd_mae_sum = 0.0
    total_extreme_sq_error = 0.0
    total_extreme_count = 0
    total_grad_norm = 0.0

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

        grad_norm = 0.0
        with torch.set_grad_enabled(is_train):
            ws_pred, wd_sin_pred, wd_cos_pred = model(batch_x, boat_id)
            loss, ws_loss, wd_loss = criterion(
                ws_pred,
                wd_sin_pred,
                wd_cos_pred,
                ws_true,
                wd_sin_true,
                wd_cos_true,
            )

            if not torch.isfinite(loss):
                continue

            if is_train:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_MAX_NORM).item()
                if not math.isfinite(grad_norm):
                    continue
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
        total_sample_count += batch_size
        total_target_count += stats["target_count"]
        total_loss += loss.item() * batch_size
        total_ws_loss += ws_loss.item() * batch_size
        total_wd_loss += wd_loss.item() * batch_size
        total_ws_sq_error += stats["ws_sq_error"]
        total_ws_abs_error += stats["ws_abs_error"]
        total_ws_true_sum += stats["ws_true_sum"]
        total_ws_true_sq_sum += stats["ws_true_sq_sum"]
        total_wd_mae_sum += stats["wd_mae_sum"]
        total_extreme_sq_error += stats["extreme_sq_error"]
        total_extreme_count += stats["extreme_count"]
        total_grad_norm += grad_norm * batch_size

        batch_rmse = math.sqrt(stats["ws_sq_error"] / max(stats["target_count"], 1))
        loop.set_postfix(loss=f"{loss.item():.4f}", ws_rmse=f"{batch_rmse:.3f}")

    if total_target_count == 0 or total_sample_count == 0:
        return {
            "loss": float("inf"),
            "ws_loss": float("inf"),
            "wd_loss": float("inf"),
            "ws_rmse": float("inf"),
            "ws_mae": float("inf"),
            "wd_mae": float("inf"),
            "extreme_ws_rmse": float("inf"),
            "r2": float("nan"),
            "grad_norm": float("inf"),
        }

    ws_mean = total_ws_true_sum / total_target_count
    ws_total_var = total_ws_true_sq_sum - total_target_count * (ws_mean ** 2)
    r2 = 1.0 - (total_ws_sq_error / ws_total_var) if ws_total_var > 1e-12 else float("nan")

    return {
        "loss": total_loss / total_sample_count,
        "ws_loss": total_ws_loss / total_sample_count,
        "wd_loss": total_wd_loss / total_sample_count,
        "ws_rmse": math.sqrt(total_ws_sq_error / total_target_count),
        "ws_mae": total_ws_abs_error / total_target_count,
        "wd_mae": total_wd_mae_sum / total_target_count,
        "extreme_ws_rmse": math.sqrt(total_extreme_sq_error / total_extreme_count) if total_extreme_count > 0 else 0.0,
        "r2": r2,
        "grad_norm": total_grad_norm / total_sample_count,
    }


def fit_stage(model, stage_name, train_loader, val_loader, optimizer, scheduler, criterion, epochs, checkpoint_path, metadata):
    best_score = float("inf")
    epochs_without_improve = 0

    for epoch in range(1, epochs + 1):
        print(f"\n[{stage_name}] Epoch {epoch}/{epochs}")
        train_metrics = run_epoch(model, train_loader, criterion, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer=None)

        if scheduler is not None:
            scheduler.step()

        current_score = calculate_score(val_metrics["ws_rmse"], val_metrics["wd_mae"])
        print(
            f"train loss={train_metrics['loss']:.4f}, val loss={val_metrics['loss']:.4f}, "
            f"val ws_rmse={val_metrics['ws_rmse']:.4f}, val ws_mae={val_metrics['ws_mae']:.4f}, "
            f"val wd_mae={val_metrics['wd_mae']:.2f}, val ext_rmse={val_metrics['extreme_ws_rmse']:.4f}, "
            f"val score={current_score:.4f}"
        )

        if current_score < best_score:
            best_score = current_score
            epochs_without_improve = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "best_score": best_score,
                "val_metrics": val_metrics,
            }
            checkpoint.update(metadata)
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved improved checkpoint to {checkpoint_path}")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= EARLY_STOPPING_PATIENCE:
                print(f"[{stage_name}] Early stopping triggered after {EARLY_STOPPING_PATIENCE} stagnant epochs.")
                break

    load_checkpoint_into_model(model, checkpoint_path)


def evaluate_split(model, dataset, criterion):
    loader = make_loader(dataset, shuffle=False)
    return run_epoch(model, loader, criterion, optimizer=None)


def collect_prediction_segment(model, dataset, num_samples=200):
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    ws_true, ws_pred, wd_true, wd_pred, wd_err = [], [], [], [], []

    model.eval()
    with torch.no_grad():
        for idx, (batch_x, batch_y, boat_id) in enumerate(loader):
            if idx >= num_samples:
                break

            batch_x = batch_x.to(DEVICE).float()
            batch_y = batch_y.to(DEVICE).float()
            boat_id = boat_id.to(DEVICE).long()

            pred_ws, pred_sin, pred_cos = model(batch_x, boat_id)

            true_ws = batch_y[0, 0, 0].item()
            pred_ws_value = pred_ws[0, 0, 0].item()

            true_deg = (torch.atan2(batch_y[0, 0, 1], batch_y[0, 0, 2]) * 180.0 / math.pi + 360.0) % 360.0
            pred_deg = (torch.atan2(pred_sin[0, 0, 0], pred_cos[0, 0, 0]) * 180.0 / math.pi + 360.0) % 360.0
            deg_diff = abs(pred_deg.item() - true_deg.item())
            deg_diff = min(deg_diff, 360.0 - deg_diff)

            ws_true.append(true_ws)
            ws_pred.append(pred_ws_value)
            wd_true.append(true_deg.item())
            wd_pred.append(pred_deg.item())
            wd_err.append(deg_diff)

    return {
        "ws_true": np.array(ws_true),
        "ws_pred": np.array(ws_pred),
        "wd_true": np.array(wd_true),
        "wd_pred": np.array(wd_pred),
        "wd_err": np.array(wd_err),
    }


def plot_results(prediction_bundle, val_metrics, test_metrics, save_path, title):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(prediction_bundle["ws_true"], label="Ground Truth", color="black", linewidth=1.5)
    ax.plot(prediction_bundle["ws_pred"], label="Prediction", color="#d62728", linewidth=1.4)
    ax.set_title("t+1 Wind Speed on Validation Samples")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Wind Speed (m/s)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(prediction_bundle["wd_err"], color="#1f77b4", linewidth=1.2, label="Absolute Direction Error")
    ax.axhline(prediction_bundle["wd_err"].mean(), color="#d62728", linestyle="--", linewidth=1.2, label="Mean Error")
    ax.set_title("t+1 Wind Direction Error")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Degrees")
    ax.grid(True, alpha=0.3)
    ax.legend()

    summary = (
        f"Val WS-RMSE: {val_metrics['ws_rmse']:.3f}\n"
        f"Val WS-MAE: {val_metrics['ws_mae']:.3f}\n"
        f"Val WD-MAE: {val_metrics['wd_mae']:.2f}\n"
        f"Val Extreme RMSE: {val_metrics['extreme_ws_rmse']:.3f}\n"
        f"Test WS-RMSE: {test_metrics['ws_rmse']:.3f}\n"
        f"Test WS-MAE: {test_metrics['ws_mae']:.3f}\n"
        f"Test WD-MAE: {test_metrics['wd_mae']:.2f}"
    )
    fig.text(0.77, 0.24, summary, fontsize=10, bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9})

    plt.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_target_datasets(target_file, scaler):
    target_id = boat_ids_for([target_file])[0]
    datasets = {
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
    return target_id, datasets


def maybe_train_stage(model, checkpoint_path, stage_name, train_loader, val_loader, criterion, epochs, lr, metadata):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        print(f"Reusing existing checkpoint: {checkpoint_path}")
        load_checkpoint_into_model(model, checkpoint_path)
        return

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    fit_stage(
        model,
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


def save_summary(rows):
    if not rows:
        return
    summary_df = pd.DataFrame(rows).sort_values(["experiment", "target"]).reset_index(drop=True)
    summary_df.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved summary to {SUMMARY_PATH}")


def train_transfer_group(criterion, summary_rows):
    print("\n" + "=" * 80)
    print("Transfer experiment: pretrain on 8 source boats, then finetune on SD1042 and SD1091")
    print("=" * 80)

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

    pretrain_model = build_model(source_feature_stats)
    pretrain_ckpt = OUTPUT_ROOT / "transfer_pretrain_source8_best.pth"
    maybe_train_stage(
        pretrain_model,
        pretrain_ckpt,
        stage_name="transfer-pretrain",
        train_loader=make_loader(source_train, shuffle=True),
        val_loader=make_loader(source_val, shuffle=False),
        criterion=criterion,
        epochs=EPOCHS_TRANSFER_PRETRAIN,
        lr=LR_TRANSFER_PRETRAIN,
        metadata={
            "experiment": "transfer",
            "stage": "pretrain",
            "source_boats": SOURCE_BOAT_FILES,
        },
    )

    for target_file in TARGET_BOAT_FILES:
        tag = boat_tag(target_file)
        model = build_model(source_feature_stats)
        load_checkpoint_into_model(model, pretrain_ckpt)

        target_id, target_datasets = build_target_datasets(target_file, source_scaler)
        finetune_ckpt = OUTPUT_ROOT / f"transfer_{tag}_best.pth"
        maybe_train_stage(
            model,
            finetune_ckpt,
            stage_name=f"transfer-finetune-{tag}",
            train_loader=make_loader(target_datasets["train"], shuffle=True),
            val_loader=make_loader(target_datasets["val"], shuffle=False),
            criterion=criterion,
            epochs=EPOCHS_TRANSFER_FINETUNE,
            lr=LR_TRANSFER_FINETUNE,
            metadata={
                "experiment": "transfer",
                "stage": "finetune",
                "target_boat": target_file,
                "target_boat_id": target_id,
                "source_pretrain_checkpoint": str(pretrain_ckpt),
            },
        )

        val_metrics = evaluate_split(model, target_datasets["val"], criterion)
        test_metrics = evaluate_split(model, target_datasets["test"], criterion)
        prediction_bundle = collect_prediction_segment(model, target_datasets["val"], num_samples=PLOT_SAMPLES)
        figure_path = FIGURE_ROOT / f"transfer_{tag}.png"
        plot_results(
            prediction_bundle,
            val_metrics,
            test_metrics,
            save_path=figure_path,
            title=f"EdgeWind Transfer Result on {tag.upper()}",
        )

        summary_rows.append(
            {
                "seed": SEED,
                "experiment": "transfer",
                "target": tag,
                "checkpoint": str(finetune_ckpt),
                "val_ws_rmse": val_metrics["ws_rmse"],
                "val_ws_mae": val_metrics["ws_mae"],
                "val_wd_mae": val_metrics["wd_mae"],
                "test_ws_rmse": test_metrics["ws_rmse"],
                "test_ws_mae": test_metrics["ws_mae"],
                "test_wd_mae": test_metrics["wd_mae"],
            }
        )
        print(f"Finished transfer training for {tag.upper()}.")


def train_no_transfer_group(criterion, summary_rows):
    print("\n" + "=" * 80)
    print("No-transfer experiment: train from scratch on each target boat only")
    print("=" * 80)

    for target_file in TARGET_BOAT_FILES:
        tag = boat_tag(target_file)
        target_id = boat_ids_for([target_file])[0]
        scratch_train = USVDataset(
            [target_file],
            global_boat_ids=[target_id],
            flag="train",
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN,
        )
        target_scaler = scratch_train.scaler
        target_feature_stats = extract_feature_stats(target_scaler)
        target_datasets = {
            "train": scratch_train,
            "val": USVDataset(
                [target_file],
                global_boat_ids=[target_id],
                flag="val",
                seq_len=SEQ_LEN,
                pred_len=PRED_LEN,
                scaler=target_scaler,
            ),
            "test": USVDataset(
                [target_file],
                global_boat_ids=[target_id],
                flag="test",
                seq_len=SEQ_LEN,
                pred_len=PRED_LEN,
                scaler=target_scaler,
            ),
        }
        model = build_model(target_feature_stats)
        scratch_ckpt = OUTPUT_ROOT / f"no_transfer_{tag}_best.pth"
        maybe_train_stage(
            model,
            scratch_ckpt,
            stage_name=f"no-transfer-{tag}",
            train_loader=make_loader(target_datasets["train"], shuffle=True),
            val_loader=make_loader(target_datasets["val"], shuffle=False),
            criterion=criterion,
            epochs=EPOCHS_NO_TRANSFER,
            lr=LR_NO_TRANSFER,
            metadata={
                "experiment": "no_transfer",
                "stage": "scratch",
                "target_boat": target_file,
                "target_boat_id": target_id,
            },
        )

        val_metrics = evaluate_split(model, target_datasets["val"], criterion)
        test_metrics = evaluate_split(model, target_datasets["test"], criterion)
        prediction_bundle = collect_prediction_segment(model, target_datasets["val"], num_samples=PLOT_SAMPLES)
        figure_path = FIGURE_ROOT / f"no_transfer_{tag}.png"
        plot_results(
            prediction_bundle,
            val_metrics,
            test_metrics,
            save_path=figure_path,
            title=f"EdgeWind No-Transfer Result on {tag.upper()}",
        )

        summary_rows.append(
            {
                "seed": SEED,
                "experiment": "no_transfer",
                "target": tag,
                "checkpoint": str(scratch_ckpt),
                "val_ws_rmse": val_metrics["ws_rmse"],
                "val_ws_mae": val_metrics["ws_mae"],
                "val_wd_mae": val_metrics["wd_mae"],
                "test_ws_rmse": test_metrics["ws_rmse"],
                "test_ws_mae": test_metrics["ws_mae"],
                "test_wd_mae": test_metrics["wd_mae"],
            }
        )
        print(f"Finished no-transfer training for {tag.upper()}.")


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
    print(f"Run no-transfer: {RUN_NO_TRANSFER}")
    print("Source boats:")
    for file_path in SOURCE_BOAT_FILES:
        print(f"  - {boat_tag(file_path).upper()}")
    print("Target boats:")
    for file_path in TARGET_BOAT_FILES:
        print(f"  - {boat_tag(file_path).upper()}")

    criterion = build_loss(
        loss_mode=LOSS_MODE,
        extreme_ws_threshold=EXTREME_WS_THRESHOLD,
        extreme_weight=EXTREME_WEIGHT,
        wd_weight=WD_WEIGHT,
    ).to(DEVICE)

    summary_rows = []
    train_transfer_group(criterion, summary_rows)
    if RUN_NO_TRANSFER:
        train_no_transfer_group(criterion, summary_rows)
    save_summary(summary_rows)

    print("\nEdgeWind training finished for all requested experiments.")


if __name__ == "__main__":
    main()
