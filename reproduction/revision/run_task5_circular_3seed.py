"""Extend the existing seed-42 circular-input sensitivity to seeds 43 and 44."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from revision_core import (
    DEVICE,
    PROJECT_ROOT,
    ROUND_ROOT,
    TARGET_BOAT_FILES,
    add_target_means,
    build_loss,
    build_model,
    collect_predictions,
    count_parameters,
    five_seed_summary,
    load_model_checkpoint,
    main_checkpoint,
    metrics_from_predictions,
    seed_everything,
    standard_source_datasets,
    standard_target_datasets,
    write_code_snapshot,
)
from experiment_config import SOURCE_BOAT_FILES, boat_ids_for, boat_tag


OUTPUT_ROOT = ROUND_ROOT / "circular_encoding_3seed"
NEW_SEEDS = (43, 44)
ALL_SEEDS = (42, 43, 44)
OLD_SCRIPT = PROJECT_ROOT / "circular" / "run_circular_sensitivity.py"


def load_old_module():
    spec = importlib.util.spec_from_file_location("seed42_circular_reference", OLD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def train_circular_seed(module, seed):
    seed_everything(seed)
    module.SEED = seed
    run_root = OUTPUT_ROOT / "runs" / f"seed_{seed}"
    checkpoint_root = run_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    source_ids = boat_ids_for(SOURCE_BOAT_FILES)
    source_train = module.CircularInputDataset(SOURCE_BOAT_FILES, source_ids, "train")
    scaler = source_train.scaler
    source_val = module.CircularInputDataset(SOURCE_BOAT_FILES, source_ids, "val", scaler=scaler)
    criterion = build_loss("smoothl1_dircos", wd_weight=1.0).to(DEVICE)
    logs = []
    started = time.perf_counter()
    pretrain_path = checkpoint_root / f"circular12d_transfer_pretrain_source8_seed{seed}.pth"
    pretrain_model = module.circular_model(scaler)
    module.fit_stage(
        pretrain_model,
        f"circular-pretrain-seed{seed}",
        source_train,
        source_val,
        criterion,
        module.PRETRAIN_EPOCHS,
        module.PRETRAIN_LR,
        pretrain_path,
        logs,
    )
    rows = []
    for target_file in TARGET_BOAT_FILES:
        target = boat_tag(target_file)
        target_id = boat_ids_for([target_file])
        train_data = module.CircularInputDataset([target_file], target_id, "train", scaler=scaler)
        val_data = module.CircularInputDataset([target_file], target_id, "val", scaler=scaler)
        test_data = module.CircularInputDataset([target_file], target_id, "test", scaler=scaler)
        model = module.circular_model(scaler)
        model.load_state_dict(torch.load(pretrain_path, map_location=DEVICE)["model_state_dict"])
        target_path = checkpoint_root / f"circular12d_transfer_{target}_seed{seed}.pth"
        module.fit_stage(
            model,
            f"circular-finetune-{target}-seed{seed}",
            train_data,
            val_data,
            criterion,
            module.FINETUNE_EPOCHS,
            module.FINETUNE_LR,
            target_path,
            logs,
        )
        metrics = module.run_epoch(model, module.loader(test_data, False), criterion)
        rows.append(
            {
                "variant": "B_circular_12d",
                "seed": seed,
                "target": target,
                "checkpoint": str(target_path),
                "parameter_count": count_parameters(model),
                **metrics,
            }
        )
    pd.DataFrame(logs).to_csv(run_root / "training_log.csv", index=False)
    (run_root / "run_config.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "variant": "12D sin/cos COG/HDG",
                "single_changed_factor": "input representation and resulting input dimension",
                "source_reference_script": str(OLD_SCRIPT),
                "exact_command": " ".join(sys.argv),
                "runtime_seconds": time.perf_counter() - started,
                "parameter_count": count_parameters(pretrain_model),
                "device": str(DEVICE),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return rows


def evaluate_degree_seed(seed, scaler):
    rows = []
    for target_file in TARGET_BOAT_FILES:
        target = boat_tag(target_file)
        checkpoint = main_checkpoint(seed, target)
        model = build_model("full", scaler)
        load_model_checkpoint(model, checkpoint)
        payload = collect_predictions(model, standard_target_datasets(target_file, scaler)["test"])
        rows.append(
            {
                "variant": "A_degree_10d",
                "seed": seed,
                "target": target,
                "checkpoint": str(checkpoint),
                "parameter_count": count_parameters(model),
                **metrics_from_predictions(payload),
            }
        )
    return rows


def seed42_circular_rows(module):
    old_root = ROUND_ROOT / "circular_encoding"
    frame = pd.read_csv(old_root / "circular_sensitivity_target_metrics.csv")
    frame = frame[frame["variant"] == "B_circular_12d"].copy()
    source_ids = boat_ids_for(SOURCE_BOAT_FILES)
    source_train = module.CircularInputDataset(SOURCE_BOAT_FILES, source_ids, "train")
    frame["parameter_count"] = count_parameters(module.circular_model(source_train.scaler))
    return frame.to_dict("records")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_code_snapshot(OUTPUT_ROOT / "code_snapshot.json", [Path(__file__), OLD_SCRIPT])
    module = load_old_module()
    source_train, _ = standard_source_datasets()
    rows = []
    for seed in ALL_SEEDS:
        rows.extend(evaluate_degree_seed(seed, source_train.scaler))
    rows.extend(seed42_circular_rows(module))
    for seed in NEW_SEEDS:
        rows.extend(train_circular_seed(module, seed))

    metrics = ["ws_rmse", "ws_mae", "wd_mae", "r2"]
    frame = add_target_means(pd.DataFrame(rows), ["variant", "seed"], metrics)
    frame.to_csv(OUTPUT_ROOT / "circular_encoding_3seed_metrics.csv", index=False)
    five_seed_summary(frame, ["variant", "target"], metrics).to_csv(
        OUTPUT_ROOT / "circular_encoding_3seed_summary.csv", index=False
    )
    print("TASK 5 complete", flush=True)


if __name__ == "__main__":
    main()
