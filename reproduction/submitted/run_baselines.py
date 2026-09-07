import math
import os
import random
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

from baselines import (
    Baseline_BP,
    Baseline_CNNLSTM,
    Baseline_Informer,
    Baseline_Mamba,
    Baseline_TCNLSTM,
    Baseline_Autoformer,
    Baseline_TimesNet,
    Baseline_Transformer,
)
from data_provider import USVDataset
from experiment_config import (
    ALL_BOAT_FILES,
    EXPERIMENT_VERSION,
    SOURCE_BOAT_FILES,
    TARGET_BOAT_FILES,
    boat_ids_for,
    boat_tag,
)
from model import DEFAULT_HIGH_WIND_THRESHOLD, build_loss


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = int(os.getenv("EDGEWIND_SEED", "42"))
OUTPUT_SUFFIX = os.getenv("EDGEWIND_OUTPUT_SUFFIX", "").strip()
BATCH_SIZE = 32
SEQ_LEN = int(os.getenv("EDGEWIND_SEQ_LEN", "36"))
PRED_LEN = int(os.getenv("EDGEWIND_PRED_LEN", "6"))
BASELINE_HIDDEN_DIM = int(os.getenv("EDGEWIND_HIDDEN_DIM", "96"))
LOSS_MODE = os.getenv("EDGEWIND_LOSS_MODE", "smoothl1_dircos").strip().lower()
EXTREME_WS_THRESHOLD = float(os.getenv("EDGEWIND_EXTREME_WS_THRESHOLD", f"{DEFAULT_HIGH_WIND_THRESHOLD}"))
EXTREME_WEIGHT = float(os.getenv("EDGEWIND_EXTREME_WEIGHT", "2.0"))
WD_WEIGHT = float(os.getenv("EDGEWIND_WD_WEIGHT", "1.0"))

EPOCHS_TRANSFER_PRETRAIN = 60
EPOCHS_TRANSFER_FINETUNE = 20
EPOCHS_NO_TRANSFER = 60

LR_TRANSFER_PRETRAIN = 3e-4
LR_TRANSFER_FINETUNE = 1.5e-4
LR_NO_TRANSFER = 3e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 8

WEIGHTS_DIR = Path("weights") / EXPERIMENT_VERSION / "baselines_v2"
if OUTPUT_SUFFIX:
    WEIGHTS_DIR = WEIGHTS_DIR / OUTPUT_SUFFIX
SUMMARY_PATH = WEIGHTS_DIR / "baseline_training_summary.csv"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
SHOW_PROGRESS = sys.stderr.isatty()
MODEL_FILTER = {
    item.strip().lower()
    for item in os.getenv("EDGEWIND_BASELINE_MODELS", "").split(",")
    if item.strip()
}
RUN_NO_TRANSFER = os.getenv("EDGEWIND_RUN_NO_TRANSFER", "1") != "0"

MODEL_SPECS = {
    "bp": (
        "BP",
        lambda: Baseline_BP(seq_len=SEQ_LEN, pred_len=PRED_LEN, hidden_dim=256).to(DEVICE),
    ),
    "cnnlstm": (
        "CNN-LSTM",
        lambda: Baseline_CNNLSTM(seq_len=SEQ_LEN, pred_len=PRED_LEN, hidden_dim=BASELINE_HIDDEN_DIM).to(DEVICE),
    ),
    "tcnlstm": (
        "TCN-LSTM (SOTA)",
        lambda: Baseline_TCNLSTM(seq_len=SEQ_LEN, pred_len=PRED_LEN, hidden_dim=BASELINE_HIDDEN_DIM).to(DEVICE),
    ),
    "informer": (
        "Informer",
        lambda: Baseline_Informer(seq_len=SEQ_LEN, pred_len=PRED_LEN, hidden_dim=BASELINE_HIDDEN_DIM).to(DEVICE),
    ),
    "transformer": (
        "Standard Transformer",
        lambda: Baseline_Transformer(seq_len=SEQ_LEN, pred_len=PRED_LEN, hidden_dim=BASELINE_HIDDEN_DIM).to(DEVICE),
    ),
    "autoformer": (
        "Autoformer",
        lambda: Baseline_Autoformer(seq_len=SEQ_LEN, pred_len=PRED_LEN, hidden_dim=BASELINE_HIDDEN_DIM).to(DEVICE),
    ),
    "timesnet": (
        "TimesNet",
        lambda: Baseline_TimesNet(seq_len=SEQ_LEN, pred_len=PRED_LEN, hidden_dim=BASELINE_HIDDEN_DIM).to(DEVICE),
    ),
    "stdmamba": (
        "Standard Mamba",
        lambda: Baseline_Mamba(seq_len=SEQ_LEN, pred_len=PRED_LEN, hidden_dim=96).to(DEVICE),
    ),
}


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_fair_score(ws_rmse, wd_mae):
    return 0.7 * ws_rmse + 0.3 * (wd_mae / 100.0)


def calculate_fair_metrics(ws_pred, wd_sin_pred, wd_cos_pred, ws_true, wd_sin_true, wd_cos_true):
    ws_sq_error = torch.sum((ws_pred - ws_true) ** 2).item()
    ws_abs_error = torch.sum(torch.abs(ws_pred - ws_true)).item()
    target_count = ws_true.numel()

    wd_pred_deg = (torch.atan2(wd_sin_pred, wd_cos_pred) * 180.0 / math.pi + 360.0) % 360.0
    wd_true_deg = (torch.atan2(wd_sin_true, wd_cos_true) * 180.0 / math.pi + 360.0) % 360.0
    wd_diff = torch.abs(wd_pred_deg - wd_true_deg)
    wd_mae_sum = torch.sum(torch.min(wd_diff, 360.0 - wd_diff)).item()
    return ws_sq_error, ws_abs_error, wd_mae_sum, target_count


def make_loader(dataset, shuffle):
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def run_stage(model, stage_name, loader_train, loader_val, epochs, lr, criterion, save_path, metadata):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_score = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loop = tqdm(
            loader_train,
            leave=False,
            desc=f"{stage_name} Epoch {epoch}/{epochs}",
            disable=not SHOW_PROGRESS,
        )
        for batch_x, batch_y, boat_id in train_loop:
            batch_x = batch_x.to(DEVICE).float()
            batch_y = batch_y.to(DEVICE).float()
            boat_id = boat_id.to(DEVICE).long()

            optimizer.zero_grad(set_to_none=True)
            ws_pred, wd_sin_pred, wd_cos_pred = model(batch_x, boat_id)
            loss, _, _ = criterion(
                ws_pred,
                wd_sin_pred,
                wd_cos_pred,
                batch_y[..., 0:1],
                batch_y[..., 1:2],
                batch_y[..., 2:3],
            )
            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loop.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        model.eval()
        total_ws_sq_error = 0.0
        total_ws_abs_error = 0.0
        total_wd_mae_sum = 0.0
        total_target_count = 0
        with torch.no_grad():
            for batch_x, batch_y, boat_id in loader_val:
                batch_x = batch_x.to(DEVICE).float()
                batch_y = batch_y.to(DEVICE).float()
                boat_id = boat_id.to(DEVICE).long()

                ws_pred, wd_sin_pred, wd_cos_pred = model(batch_x, boat_id)
                ws_sq_error, ws_abs_error, wd_mae_sum, target_count = calculate_fair_metrics(
                    ws_pred,
                    wd_sin_pred,
                    wd_cos_pred,
                    batch_y[..., 0:1],
                    batch_y[..., 1:2],
                    batch_y[..., 2:3],
                )
                total_ws_sq_error += ws_sq_error
                total_ws_abs_error += ws_abs_error
                total_wd_mae_sum += wd_mae_sum
                total_target_count += target_count

        mean_ws_rmse = math.sqrt(total_ws_sq_error / max(total_target_count, 1))
        mean_ws_mae = total_ws_abs_error / max(total_target_count, 1)
        mean_wd_mae = total_wd_mae_sum / max(total_target_count, 1)
        current_score = calculate_fair_score(mean_ws_rmse, mean_wd_mae)

        print(
            f"{stage_name} | epoch {epoch} | "
            f"val ws_rmse={mean_ws_rmse:.4f} | val ws_mae={mean_ws_mae:.4f} | "
            f"val wd_mae={mean_wd_mae:.2f} | fair score={current_score:.4f}"
        )

        if current_score < best_score:
            best_score = current_score
            patience_counter = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "ws_rmse": mean_ws_rmse,
                "ws_mae": mean_ws_mae,
                "wd_mae": mean_wd_mae,
                "fair_score": current_score,
                "experiment_version": EXPERIMENT_VERSION,
            }
            checkpoint.update(metadata)
            torch.save(checkpoint, save_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"{stage_name} | early stopping triggered.")
                break

    load_checkpoint(model, save_path)


def maybe_train_stage(model, checkpoint_path, stage_name, loader_train, loader_val, epochs, lr, criterion, metadata):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        try:
            print(f"Reusing existing checkpoint: {checkpoint_path}")
            load_checkpoint(model, checkpoint_path)
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

    run_stage(
        model=model,
        stage_name=stage_name,
        loader_train=loader_train,
        loader_val=loader_val,
        epochs=epochs,
        lr=lr,
        criterion=criterion,
        save_path=checkpoint_path,
        metadata=metadata,
    )


def evaluate_checkpoint(model, dataset):
    loader = make_loader(dataset, shuffle=False)
    model.eval()
    total_ws_sq_error = 0.0
    total_ws_abs_error = 0.0
    total_wd_mae_sum = 0.0
    total_target_count = 0
    total_ws_sum = 0.0
    total_ws_sq_sum = 0.0
    with torch.no_grad():
        for batch_x, batch_y, boat_id in loader:
            batch_x = batch_x.to(DEVICE).float()
            batch_y = batch_y.to(DEVICE).float()
            boat_id = boat_id.to(DEVICE).long()

            ws_pred, wd_sin_pred, wd_cos_pred = model(batch_x, boat_id)
            ws_sq_error, ws_abs_error, wd_mae_sum, target_count = calculate_fair_metrics(
                ws_pred,
                wd_sin_pred,
                wd_cos_pred,
                batch_y[..., 0:1],
                batch_y[..., 1:2],
                batch_y[..., 2:3],
            )
            total_ws_sq_error += ws_sq_error
            total_ws_abs_error += ws_abs_error
            total_wd_mae_sum += wd_mae_sum
            total_target_count += target_count
            ws_true = batch_y[..., 0:1]
            total_ws_sum += torch.sum(ws_true).item()
            total_ws_sq_sum += torch.sum(ws_true**2).item()

    ws_rmse = math.sqrt(total_ws_sq_error / max(total_target_count, 1))
    ws_mae = total_ws_abs_error / max(total_target_count, 1)
    wd_mae = total_wd_mae_sum / max(total_target_count, 1)
    ws_mean = total_ws_sum / max(total_target_count, 1)
    ws_sst = total_ws_sq_sum - total_target_count * (ws_mean**2)
    ws_r2 = 1.0 - total_ws_sq_error / max(ws_sst, 1e-12)
    return {
        "ws_rmse": ws_rmse,
        "ws_mae": ws_mae,
        "wd_mae": wd_mae,
        "ws_r2": ws_r2,
        "fair_score": calculate_fair_score(ws_rmse, wd_mae),
    }


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


def save_summary(rows):
    if not rows:
        return
    summary_df = pd.DataFrame(rows).sort_values(["model", "experiment", "target"]).reset_index(drop=True)
    summary_df.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved summary to {SUMMARY_PATH}")


def train_transfer_for_model(model_key, model_name, model_factory, criterion, summary_rows):
    model_dir = WEIGHTS_DIR / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

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

    pretrain_ckpt = model_dir / "transfer_pretrain_source8_best.pth"
    pretrain_model = model_factory()
    maybe_train_stage(
        pretrain_model,
        checkpoint_path=pretrain_ckpt,
        stage_name=f"{model_name}-transfer-pretrain",
        loader_train=make_loader(source_train, shuffle=True),
        loader_val=make_loader(source_val, shuffle=False),
        epochs=EPOCHS_TRANSFER_PRETRAIN,
        lr=LR_TRANSFER_PRETRAIN,
        criterion=criterion,
        metadata={
            "model_name": model_name,
            "experiment": "transfer",
            "stage": "pretrain",
            "source_boats": SOURCE_BOAT_FILES,
        },
    )

    for target_file in TARGET_BOAT_FILES:
        tag = boat_tag(target_file)
        target_id, target_datasets = build_target_datasets(target_file, source_scaler)

        model = model_factory()
        load_checkpoint(model, pretrain_ckpt)
        target_ckpt = model_dir / f"transfer_{tag}_best.pth"
        maybe_train_stage(
            model,
            checkpoint_path=target_ckpt,
            stage_name=f"{model_name}-transfer-finetune-{tag}",
            loader_train=make_loader(target_datasets["train"], shuffle=True),
            loader_val=make_loader(target_datasets["val"], shuffle=False),
            epochs=EPOCHS_TRANSFER_FINETUNE,
            lr=LR_TRANSFER_FINETUNE,
            criterion=criterion,
            metadata={
                "model_name": model_name,
                "experiment": "transfer",
                "stage": "finetune",
                "target_boat": target_file,
                "target_boat_id": target_id,
            },
        )

        val_metrics = evaluate_checkpoint(model, target_datasets["val"])
        test_metrics = evaluate_checkpoint(model, target_datasets["test"])
        summary_rows.append(
            {
                "seed": SEED,
                "model": model_name,
                "experiment": "transfer",
                "target": tag,
                "checkpoint": str(target_ckpt),
                "val_ws_rmse": val_metrics["ws_rmse"],
                "val_ws_mae": val_metrics["ws_mae"],
                "val_wd_mae": val_metrics["wd_mae"],
                "val_ws_r2": val_metrics["ws_r2"],
                "val_fair_score": val_metrics["fair_score"],
                "test_ws_rmse": test_metrics["ws_rmse"],
                "test_ws_mae": test_metrics["ws_mae"],
                "test_wd_mae": test_metrics["wd_mae"],
                "test_ws_r2": test_metrics["ws_r2"],
                "test_fair_score": test_metrics["fair_score"],
            }
        )


def train_no_transfer_for_model(model_key, model_name, model_factory, criterion, summary_rows):
    model_dir = WEIGHTS_DIR / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

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

        model = model_factory()
        target_ckpt = model_dir / f"no_transfer_{tag}_best.pth"
        maybe_train_stage(
            model,
            checkpoint_path=target_ckpt,
            stage_name=f"{model_name}-no-transfer-{tag}",
            loader_train=make_loader(target_datasets["train"], shuffle=True),
            loader_val=make_loader(target_datasets["val"], shuffle=False),
            epochs=EPOCHS_NO_TRANSFER,
            lr=LR_NO_TRANSFER,
            criterion=criterion,
            metadata={
                "model_name": model_name,
                "experiment": "no_transfer",
                "stage": "scratch",
                "target_boat": target_file,
                "target_boat_id": target_id,
            },
        )

        val_metrics = evaluate_checkpoint(model, target_datasets["val"])
        test_metrics = evaluate_checkpoint(model, target_datasets["test"])
        summary_rows.append(
            {
                "seed": SEED,
                "model": model_name,
                "experiment": "no_transfer",
                "target": tag,
                "checkpoint": str(target_ckpt),
                "val_ws_rmse": val_metrics["ws_rmse"],
                "val_ws_mae": val_metrics["ws_mae"],
                "val_wd_mae": val_metrics["wd_mae"],
                "val_ws_r2": val_metrics["ws_r2"],
                "val_fair_score": val_metrics["fair_score"],
                "test_ws_rmse": test_metrics["ws_rmse"],
                "test_ws_mae": test_metrics["ws_mae"],
                "test_wd_mae": test_metrics["wd_mae"],
                "test_ws_r2": test_metrics["ws_r2"],
                "test_fair_score": test_metrics["fair_score"],
            }
        )


def main():
    seed_everything(SEED)
    print(f"Device: {DEVICE}")
    print(f"Experiment version: {EXPERIMENT_VERSION}")
    print(f"Seed: {SEED}")
    if OUTPUT_SUFFIX:
        print(f"Output suffix: {OUTPUT_SUFFIX}")
    print(
        f"Config | seq_len={SEQ_LEN} pred_len={PRED_LEN} hidden_dim={BASELINE_HIDDEN_DIM} "
        f"loss_mode={LOSS_MODE} high_wind_threshold={EXTREME_WS_THRESHOLD:.2f}"
    )
    print(f"Run no-transfer baselines: {RUN_NO_TRANSFER}")
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

    for model_key, (model_name, model_factory) in MODEL_SPECS.items():
        if MODEL_FILTER and model_key.lower() not in MODEL_FILTER and model_name.lower() not in MODEL_FILTER:
            continue
        print("\n" + "=" * 80)
        print(f"Training baseline: {model_name}")
        print("=" * 80)
        train_transfer_for_model(model_key, model_name, model_factory, criterion, summary_rows)
        if RUN_NO_TRANSFER:
            train_no_transfer_for_model(model_key, model_name, model_factory, criterion, summary_rows)

    save_summary(summary_rows)
    print("\nBaseline training finished for all requested experiments.")


if __name__ == "__main__":
    main()
