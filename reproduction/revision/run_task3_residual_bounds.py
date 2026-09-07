"""Run pre-specified residual-bound sensitivity experiments."""

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
    main_checkpoint,
    metrics_from_predictions,
    regime_masks,
    source_training_statistics,
    standard_source_datasets,
    standard_target_datasets,
    train_transfer_variant,
    write_code_snapshot,
)
from experiment_config import boat_tag


OUTPUT_ROOT = ROUND_ROOT / "residual_bound"
NEW_VARIANTS = (
    ("unbounded", "unbounded"),
    ("fixed6", "fixed6"),
    ("training_derived_component", "training_derived"),
    ("learnable_global_component", "learnable"),
)


def trained_payload(variant, seed, target):
    run_root = OUTPUT_ROOT / "runs" / variant / f"seed_{seed}"
    row = pd.read_csv(run_root / "target_metrics.csv")
    row = row[row["target"] == target].iloc[0].to_dict()
    return row, dict(np.load(row["prediction_path"]))


def fixed4_payload(seed, target_file, scaler):
    target = boat_tag(target_file)
    checkpoint = main_checkpoint(seed, target)
    model = build_model("full", scaler)
    load_model_checkpoint(model, checkpoint)
    dataset = standard_target_datasets(target_file, scaler)["test"]
    payload = collect_predictions(model, dataset)
    prediction_root = OUTPUT_ROOT / "reused_predictions" / "fixed4" / f"seed_{seed}"
    prediction_root.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_root / f"predictions_{target}.npz"
    np.savez_compressed(prediction_path, **payload)
    return {
        "variant": "fixed4_submitted",
        "seed": seed,
        "target": target,
        "checkpoint": str(checkpoint),
        "prediction_path": str(prediction_path),
        "parameter_count": count_parameters(model),
        "learned_bound_u": 4.0,
        "learned_bound_v": 4.0,
        **metrics_from_predictions(payload),
    }, payload


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_code_snapshot(OUTPUT_ROOT / "code_snapshot.json", [Path(__file__)])
    started = time.perf_counter()
    source_train, _ = standard_source_datasets()
    scaler = source_train.scaler
    thresholds = source_training_statistics(source_train)
    pd.DataFrame([thresholds]).to_csv(OUTPUT_ROOT / "training_residual_statistics.csv", index=False)

    criterion = build_loss("smoothl1_dircos", wd_weight=1.0).to(DEVICE)
    bound_u = thresholds["source_training_abs_delta_u_p99"]
    bound_v = thresholds["source_training_abs_delta_v_p99"]
    for variant, model_kind in NEW_VARIANTS:
        for seed in SEEDS:
            train_transfer_variant(
                variant=variant,
                model_kind=model_kind,
                seed=seed,
                output_root=OUTPUT_ROOT,
                criterion=criterion,
                bound_u=bound_u,
                bound_v=bound_v,
            )

    metric_rows = []
    regime_rows = []
    bound_rows = []
    for seed in SEEDS:
        for target_file in TARGET_BOAT_FILES:
            target = boat_tag(target_file)
            for variant, _ in (("fixed4_submitted", "full"),) + NEW_VARIANTS:
                if variant == "fixed4_submitted":
                    row, payload = fixed4_payload(seed, target_file, scaler)
                else:
                    row, payload = trained_payload(variant, seed, target)
                metric_rows.append(row)
                bound_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "target": target,
                        "delta_u": row.get("learned_bound_u", np.nan),
                        "delta_v": row.get("learned_bound_v", np.nan),
                    }
                )
                masks = regime_masks(payload, scaler, thresholds)
                for regime in ("upper_tail", "rapid_component_change", "strong_gust", "rapid_wd_change"):
                    regime_rows.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "target": target,
                            "regime": regime,
                            **metrics_from_predictions(payload, masks[regime]),
                        }
                    )

    numeric = ["count", "ws_rmse", "ws_mae", "wd_mae", "r2"]
    metrics = add_target_means(pd.DataFrame(metric_rows), ["variant", "seed"], numeric)
    metrics.to_csv(OUTPUT_ROOT / "residual_bound_five_seed_metrics.csv", index=False)
    five_seed_summary(metrics, ["variant", "target"], ["ws_rmse", "ws_mae", "wd_mae", "r2"]).to_csv(
        OUTPUT_ROOT / "residual_bound_five_seed_summary.csv", index=False
    )

    regimes = add_target_means(pd.DataFrame(regime_rows), ["variant", "seed", "regime"], numeric)
    regimes.to_csv(OUTPUT_ROOT / "residual_bound_regime_metrics.csv", index=False)
    five_seed_summary(regimes, ["variant", "target", "regime"], numeric).to_csv(
        OUTPUT_ROOT / "residual_bound_regime_summary.csv", index=False
    )
    pd.DataFrame(bound_rows).to_csv(OUTPUT_ROOT / "learned_bound_values.csv", index=False)
    (OUTPUT_ROOT / "task3_runtime.json").write_text(
        json.dumps({"runtime_seconds": time.perf_counter() - started, "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2),
        encoding="utf-8",
    )
    print("TASK 3 complete", flush=True)


if __name__ == "__main__":
    main()
