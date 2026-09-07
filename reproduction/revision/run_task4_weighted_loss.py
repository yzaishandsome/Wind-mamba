"""Re-run only the weighted Dir-Cos loss with a training-only threshold."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from revision_core import (
    DEVICE,
    ROUND_ROOT,
    SEEDS,
    TARGET_BOAT_FILES,
    add_target_means,
    build_loss,
    build_model,
    collect_predictions,
    count_parameters,
    five_seed_summary,
    load_model_checkpoint,
    loss_checkpoint,
    metrics_from_predictions,
    source_training_statistics,
    standard_source_datasets,
    standard_target_datasets,
    train_transfer_variant,
    write_code_snapshot,
)
from experiment_config import boat_tag


OUTPUT_ROOT = ROUND_ROOT / "weighted_loss_training_threshold"
WEIGHTED_VARIANT = "upper_tail_weighted_dir_cos_training_p95"
REUSED_VARIANTS = (
    ("dir_l1_unweighted", "smoothl1_dir_l1"),
    ("dir_cos_unweighted", "smoothl1_dir_cos"),
)


def evaluated_row(variant, seed, target_file, scaler, threshold):
    target = boat_tag(target_file)
    loss_key = dict(REUSED_VARIANTS)[variant]
    checkpoint = loss_checkpoint(seed, loss_key, target)
    model = build_model("full", scaler)
    load_model_checkpoint(model, checkpoint)
    dataset = standard_target_datasets(target_file, scaler)["test"]
    payload = collect_predictions(model, dataset)
    prediction_root = OUTPUT_ROOT / "reused_predictions" / variant / f"seed_{seed}"
    prediction_root.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_root / f"predictions_{target}.npz"
    np.savez_compressed(prediction_path, **payload)
    overall = metrics_from_predictions(payload)
    upper = metrics_from_predictions(payload, payload["ws_true"] > threshold)
    return {
        "variant": variant,
        "seed": seed,
        "target": target,
        "checkpoint": str(checkpoint),
        "prediction_path": str(prediction_path),
        "parameter_count": count_parameters(model),
        "threshold": threshold,
        "upper_tail_count": upper["count"],
        "upper_tail_ratio": upper["count"] / overall["count"],
        "upper_tail_ws_rmse": upper["ws_rmse"],
        "overall_count": overall["count"],
        "ws_rmse": overall["ws_rmse"],
        "ws_mae": overall["ws_mae"],
        "wd_mae": overall["wd_mae"],
        "r2": overall["r2"],
    }


def weighted_row(seed, target, threshold):
    run_root = OUTPUT_ROOT / "runs" / WEIGHTED_VARIANT / f"seed_{seed}"
    row = pd.read_csv(run_root / "target_metrics.csv")
    row = row[row["target"] == target].iloc[0].to_dict()
    payload = dict(np.load(row["prediction_path"]))
    upper = metrics_from_predictions(payload, payload["ws_true"] > threshold)
    row.update(
        {
            "threshold": threshold,
            "upper_tail_count": upper["count"],
            "upper_tail_ratio": upper["count"] / row["count"],
            "upper_tail_ws_rmse": upper["ws_rmse"],
            "overall_count": row["count"],
        }
    )
    return row


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_code_snapshot(OUTPUT_ROOT / "code_snapshot.json", [Path(__file__)])
    started = time.perf_counter()
    source_train, _ = standard_source_datasets()
    scaler = source_train.scaler
    statistics = source_training_statistics(source_train)
    threshold = statistics["source_training_ws_p95"]
    if not np.isclose(threshold, 10.580439745930407, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"Unexpected source-training p95: {threshold!r}")
    (OUTPUT_ROOT / "training_only_threshold.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")

    criterion = build_loss(
        "smoothl1_extreme_dircos",
        extreme_ws_threshold=threshold,
        extreme_weight=2.0,
        wd_weight=1.0,
    ).to(DEVICE)
    for seed in SEEDS:
        train_transfer_variant(
            variant=WEIGHTED_VARIANT,
            model_kind="full",
            seed=seed,
            output_root=OUTPUT_ROOT,
            criterion=criterion,
        )

    rows = []
    for seed in SEEDS:
        for target_file in TARGET_BOAT_FILES:
            for variant, _ in REUSED_VARIANTS:
                rows.append(evaluated_row(variant, seed, target_file, scaler, threshold))
            rows.append(weighted_row(seed, boat_tag(target_file), threshold))

    metrics = ["ws_rmse", "ws_mae", "wd_mae", "r2", "upper_tail_ws_rmse", "upper_tail_count", "upper_tail_ratio"]
    frame = add_target_means(pd.DataFrame(rows), ["variant", "seed", "threshold"], metrics)
    frame.to_csv(OUTPUT_ROOT / "weighted_loss_revision_metrics.csv", index=False)
    five_seed_summary(frame, ["variant", "target"], metrics).to_csv(
        OUTPUT_ROOT / "weighted_loss_revision_summary.csv", index=False
    )
    (OUTPUT_ROOT / "task4_runtime.json").write_text(
        json.dumps({"runtime_seconds": time.perf_counter() - started, "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2),
        encoding="utf-8",
    )
    print("TASK 4 complete", flush=True)


if __name__ == "__main__":
    main()
