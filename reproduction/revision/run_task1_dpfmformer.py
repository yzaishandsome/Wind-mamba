"""Run the isolated closest-prior-art DPFMformer comparison."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from dpfmformer_model import DPFMformerKFLDirectionLoss, DPFMformerMarine
from experiment_config import SOURCE_BOAT_FILES, TARGET_BOAT_FILES, boat_tag
from revision_core import (
    BATCH_SIZE,
    DEVICE,
    FINETUNE_EPOCHS,
    FINETUNE_LR,
    GRAD_MAX_NORM,
    PATIENCE,
    PRED_LEN,
    PRETRAIN_EPOCHS,
    PRETRAIN_LR,
    ROUND_ROOT,
    SEEDS,
    SEQ_LEN,
    WEIGHT_DECAY,
    add_target_means,
    base_configuration,
    build_loss,
    collect_predictions,
    count_parameters,
    file_sha256,
    fit_stage,
    five_seed_summary,
    make_loader,
    metrics_from_predictions,
    seed_everything,
    standard_source_datasets,
    standard_target_datasets,
    write_code_snapshot,
)


OUTPUT_ROOT = ROUND_ROOT / "closest_prior_art"
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PDF = Path(
    os.getenv("DPFMFORMER_PAPER_PATH", str(REPO_ROOT / "references" / "1-s2.0-S0360544225028671-main.pdf"))
).resolve()
SOURCE_PDF_SHA256 = "034A0E9CD86F58FD88916FFB7D5B6C39ECA59466EE600BE0774EBC790229F4B8"
SPEC_PATH = REPO_ROOT / "docs" / "dpfmformer_reimplementation_spec.md"
VARIANTS = ("paper_kfl_dircos", "common_smoothl1_dircos")


def build_dpfmformer() -> DPFMformerMarine:
    return DPFMformerMarine(
        input_dim=10,
        sequence_length=SEQ_LEN,
        prediction_length=PRED_LEN,
        model_dim=16,
        state_dim=16,
        convolution_kernel=4,
        attention_heads=4,
        dropout=0.1,
        moving_average_kernel=25,
        scale_lengths=(36, 18, 9),
    ).to(DEVICE)


def build_criterion(variant: str):
    if variant == "paper_kfl_dircos":
        return DPFMformerKFLDirectionLoss(alpha=0.9, direction_weight=1.0).to(DEVICE)
    if variant == "common_smoothl1_dircos":
        return build_loss("smoothl1_dircos", wd_weight=1.0).to(DEVICE)
    raise ValueError(f"Unknown DPFMformer variant: {variant}")


def verify_primary_source() -> None:
    if not SOURCE_PDF.exists():
        raise FileNotFoundError(SOURCE_PDF)
    observed = file_sha256(SOURCE_PDF).upper()
    if observed != SOURCE_PDF_SHA256:
        raise RuntimeError(f"Primary-source PDF hash changed: {observed}")


def smoke_test() -> None:
    """Check shapes and gradients without constructing any target test loader."""
    verify_primary_source()
    seed_everything(42)
    source_train, source_val = standard_source_datasets()
    train_batch = next(iter(make_loader(source_train, True)))
    val_batch = next(iter(make_loader(source_val, False)))
    records = []
    for variant in VARIANTS:
        seed_everything(42)
        model = build_dpfmformer()
        criterion = build_criterion(variant)
        optimizer = optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=WEIGHT_DECAY)
        model.train()
        x, y, boat_id = train_batch
        x = x.to(DEVICE).float()
        y = y.to(DEVICE).float()
        boat_id = boat_id.to(DEVICE).long()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(x, boat_id)
        loss, wind_speed_loss, direction_loss = criterion(
            *outputs,
            y[..., 0:1],
            y[..., 1:2],
            y[..., 2:3],
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_MAX_NORM)
        if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"Non-finite seed-42 smoke result for {variant}")
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_x, val_y, val_boat = val_batch
            val_outputs = model(val_x.to(DEVICE).float(), val_boat.to(DEVICE).long())
            val_loss, _, _ = criterion(
                *val_outputs,
                val_y.to(DEVICE).float()[..., 0:1],
                val_y.to(DEVICE).float()[..., 1:2],
                val_y.to(DEVICE).float()[..., 2:3],
            )
        expected_shape = (len(x), PRED_LEN, 1)
        shapes = [tuple(output.shape) for output in outputs]
        if any(shape != expected_shape for shape in shapes):
            raise RuntimeError(f"Unexpected output shapes for {variant}: {shapes}")
        direction_norm = torch.sqrt(outputs[1].square() + outputs[2].square())
        if not torch.allclose(direction_norm, torch.ones_like(direction_norm), atol=1e-5):
            raise RuntimeError(f"Non-unit direction output for {variant}")
        records.append(
            {
                "variant": variant,
                "seed": 42,
                "source_train_windows": len(source_train),
                "source_validation_windows": len(source_val),
                "target_test_loader_constructed": False,
                "output_shapes": [list(shape) for shape in shapes],
                "parameter_count": count_parameters(model),
                "train_smoke_total_loss": float(loss.item()),
                "train_smoke_ws_objective": float(wind_speed_loss.item()),
                "train_smoke_direction_loss": float(direction_loss.item()),
                "validation_smoke_total_loss": float(val_loss.item()),
                "gradient_norm_before_clip": float(gradient_norm.item()),
                "minimum_predicted_ws": float(outputs[0].min().item()),
                "maximum_direction_norm_error": float((direction_norm - 1.0).abs().max().item()),
            }
        )
    payload = {
        "status": "PASS",
        "purpose": "shape, finite-gradient, and split-use smoke test only",
        "architecture_frozen_before_smoke": True,
        "target_test_metrics_examined": False,
        "primary_source_pdf": str(SOURCE_PDF),
        "primary_source_sha256": SOURCE_PDF_SHA256,
        "specification_sha256": file_sha256(SPEC_PATH),
        "records": records,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (OUTPUT_ROOT / "smoke_test_seed42.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


def train_variant_seed(variant: str, seed: int) -> list[dict]:
    verify_primary_source()
    seed_everything(seed)
    source_train, source_val = standard_source_datasets()
    criterion = build_criterion(variant)
    run_root = OUTPUT_ROOT / "runs" / variant / f"seed_{seed}"
    checkpoint_root = run_root / "checkpoints"
    log_path = run_root / "training_log.jsonl"
    model = build_dpfmformer()
    parameter_count = count_parameters(model)
    pretrain_path = checkpoint_root / "transfer_pretrain_source8_best.pth"
    stages = [
        fit_stage(
            model,
            f"dpfmformer-{variant}-pretrain-seed{seed}",
            source_train,
            source_val,
            criterion,
            PRETRAIN_EPOCHS,
            PRETRAIN_LR,
            pretrain_path,
            log_path,
            {
                "architecture": "DPFMformerMarine",
                "variant": variant,
                "seed": seed,
                "stage": "pretrain",
                "source_boats": SOURCE_BOAT_FILES,
                "primary_source_sha256": SOURCE_PDF_SHA256,
                "specification_sha256": file_sha256(SPEC_PATH),
            },
        )
    ]

    rows = []
    for target_file in TARGET_BOAT_FILES:
        target = boat_tag(target_file)
        datasets = standard_target_datasets(target_file, source_train.scaler)
        target_model = build_dpfmformer()
        target_model.load_state_dict(torch.load(pretrain_path, map_location=DEVICE)["model_state_dict"])
        target_path = checkpoint_root / f"transfer_{target}_best.pth"
        stages.append(
            fit_stage(
                target_model,
                f"dpfmformer-{variant}-finetune-{target}-seed{seed}",
                datasets["train"],
                datasets["val"],
                criterion,
                FINETUNE_EPOCHS,
                FINETUNE_LR,
                target_path,
                log_path,
                {
                    "architecture": "DPFMformerMarine",
                    "variant": variant,
                    "seed": seed,
                    "stage": "finetune",
                    "target": target,
                    "source_checkpoint": str(pretrain_path),
                    "primary_source_sha256": SOURCE_PDF_SHA256,
                    "specification_sha256": file_sha256(SPEC_PATH),
                },
            )
        )
        payload = collect_predictions(target_model, datasets["test"])
        prediction_path = run_root / f"predictions_{target}.npz"
        np.savez_compressed(prediction_path, **payload)
        rows.append(
            {
                "architecture": "DPFMformerMarine",
                "loss_variant": variant,
                "seed": seed,
                "target": target,
                "checkpoint": str(target_path),
                "prediction_path": str(prediction_path),
                "parameter_count": parameter_count,
                **metrics_from_predictions(payload),
            }
        )

    runtime_seconds = float(sum(stage["runtime_seconds"] for stage in stages))
    configuration = base_configuration()
    configuration.update(
        {
            "architecture": "paper-based DPFMformer minimal marine adaptation",
            "loss_variant": variant,
            "seed": seed,
            "model_dim": 16,
            "state_dimension": 16,
            "mamba_convolution_kernel": 4,
            "attention_heads": 4,
            "dropout": 0.1,
            "moving_average_kernel": 25,
            "scale_lengths": [36, 18, 9],
            "paper_kfl_alpha": 0.9 if variant == "paper_kfl_dircos" else None,
            "paper_original_optimizer": "Adam",
            "paper_original_learning_rate": 1e-4,
            "paper_original_epochs": 10,
            "benchmark_optimizer_used": "AdamW",
            "parameter_count": parameter_count,
            "runtime_seconds": runtime_seconds,
            "python_executable": sys.executable,
            "exact_command": " ".join([f'"{sys.executable}"', *sys.argv]),
            "primary_source_pdf": str(SOURCE_PDF),
            "primary_source_sha256": SOURCE_PDF_SHA256,
            "specification_sha256": file_sha256(SPEC_PATH),
            "test_used_for_hyperparameter_or_architecture_selection": False,
        }
    )
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "run_config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(run_root / "target_metrics.csv", index=False)
    return rows


def aggregate() -> None:
    rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            path = OUTPUT_ROOT / "runs" / variant / f"seed_{seed}" / "target_metrics.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing formal result: {path}")
            rows.extend(pd.read_csv(path).to_dict(orient="records"))
    metric_columns = ["ws_rmse", "ws_mae", "wd_mae", "r2", "count"]
    frame = add_target_means(pd.DataFrame(rows), ["architecture", "loss_variant", "seed"], metric_columns)
    frame.to_csv(OUTPUT_ROOT / "closest_prior_art_five_seed_metrics.csv", index=False)
    summary = five_seed_summary(
        frame,
        ["architecture", "loss_variant", "target"],
        ["ws_rmse", "ws_mae", "wd_mae", "r2", "parameter_count"],
    )
    summary.to_csv(OUTPUT_ROOT / "closest_prior_art_five_seed_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "train", "aggregate", "all"), default="all")
    parser.add_argument("--variant", choices=VARIANTS, action="append")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_code_snapshot(
        OUTPUT_ROOT / "code_snapshot.json",
        [Path(__file__), Path(__file__).with_name("dpfmformer_model.py"), SPEC_PATH],
    )
    if args.mode in ("smoke", "all"):
        smoke_test()
    if args.mode in ("train", "all"):
        variants = tuple(args.variant or VARIANTS)
        for variant in variants:
            for seed in args.seeds:
                if seed not in SEEDS:
                    raise ValueError(f"Seed {seed} is outside the pre-specified set {SEEDS}")
                train_variant_seed(variant, seed)
    if args.mode in ("aggregate", "all"):
        aggregate()


if __name__ == "__main__":
    main()
