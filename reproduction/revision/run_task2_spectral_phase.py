"""Run the pre-specified spectral-representation sensitivity and diagnostics."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from revision_core import (
    DEVICE,
    ROUND_ROOT,
    SEEDS,
    TARGET_BOAT_FILES,
    _frequency_ratio,
    add_target_means,
    build_loss,
    build_model,
    collect_predictions,
    count_parameters,
    five_seed_summary,
    load_model_checkpoint,
    main_checkpoint,
    make_loader,
    metrics_from_predictions,
    no_fft_checkpoint,
    regime_masks,
    source_training_statistics,
    standard_source_datasets,
    standard_target_datasets,
    train_transfer_variant,
    write_code_snapshot,
)
from experiment_config import boat_tag


OUTPUT_ROOT = ROUND_ROOT / "spectral_phase"
TRAINED_VARIANT = "magnitude_only_strict_zero_phase"


def feature_response(model, dataset):
    captured = []

    def hook(_, __, output):
        captured.append(output.detach().cpu().numpy())

    handle = model.turbulence_extractor.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for x, _, boat_id in make_loader(dataset, False):
            model(x.to(DEVICE).float(), boat_id.to(DEVICE).long())
    handle.remove()
    return _frequency_ratio(np.concatenate(captured, axis=0))


def evaluate_reused_variant(variant, model_kind, seed, target_file, scaler):
    target = boat_tag(target_file)
    checkpoint = main_checkpoint(seed, target) if variant == "full_real_imag" else no_fft_checkpoint(seed, target)
    dataset = standard_target_datasets(target_file, scaler)["test"]
    model = build_model(model_kind, scaler)
    load_model_checkpoint(model, checkpoint)
    payload = collect_predictions(model, dataset)
    prediction_root = OUTPUT_ROOT / "reused_predictions" / variant / f"seed_{seed}"
    prediction_root.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_root / f"predictions_{target}.npz"
    np.savez_compressed(prediction_path, **payload)
    return model, dataset, payload, {
        "variant": variant,
        "seed": seed,
        "target": target,
        "checkpoint": str(checkpoint),
        "prediction_path": str(prediction_path),
        "parameter_count": count_parameters(model),
        **metrics_from_predictions(payload),
    }


def locate_trained_payload(seed, target):
    run_root = OUTPUT_ROOT / "runs" / TRAINED_VARIANT / f"seed_{seed}"
    metrics = pd.read_csv(run_root / "target_metrics.csv")
    row = metrics[metrics["target"] == target].iloc[0].to_dict()
    payload = dict(np.load(row["prediction_path"]))
    return row, payload


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_code_snapshot(OUTPUT_ROOT / "code_snapshot.json", [Path(__file__)])
    started = time.perf_counter()
    source_train, _ = standard_source_datasets()
    scaler = source_train.scaler
    thresholds = source_training_statistics(source_train)
    (OUTPUT_ROOT / "training_only_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    criterion = build_loss("smoothl1_dircos", wd_weight=1.0).to(DEVICE)
    for seed in SEEDS:
        train_transfer_variant(
            variant=TRAINED_VARIANT,
            model_kind="magnitude_only",
            seed=seed,
            output_root=OUTPUT_ROOT,
            criterion=criterion,
        )

    metric_rows = []
    regime_rows = []
    gust_rows = []
    diagnostic_rows = []
    for seed in SEEDS:
        for target_file in TARGET_BOAT_FILES:
            target = boat_tag(target_file)
            datasets = standard_target_datasets(target_file, scaler)
            for variant, model_kind in (
                ("full_real_imag", "full"),
                (TRAINED_VARIANT, "magnitude_only"),
                ("no_fft", "no_fft"),
            ):
                if variant == TRAINED_VARIANT:
                    row, payload = locate_trained_payload(seed, target)
                    checkpoint = Path(row["checkpoint"])
                    model = build_model("magnitude_only", scaler)
                    load_model_checkpoint(model, checkpoint)
                    metric_rows.append(row)
                else:
                    model, _, payload, row = evaluate_reused_variant(variant, model_kind, seed, target_file, scaler)
                    metric_rows.append(row)

                masks = regime_masks(payload, scaler, thresholds)
                for regime in ("hf_q1", "hf_q2", "hf_q3", "hf_q4"):
                    regime_rows.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "target": target,
                            "regime": regime,
                            **metrics_from_predictions(payload, masks[regime]),
                        }
                    )
                for regime in ("strong_gust", "non_strong_gust"):
                    gust_rows.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "target": target,
                            "regime": regime,
                            "gust_threshold": thresholds["source_training_gust_p95"],
                            **metrics_from_predictions(payload, masks[regime]),
                        }
                    )

                output_hf = feature_response(model, datasets["test"]) if variant != "no_fft" else np.full(len(masks["hf_ratio"]), np.nan)
                for sample_index, (input_ratio, output_ratio) in enumerate(zip(masks["hf_ratio"], output_hf)):
                    diagnostic_rows.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "target": target,
                            "sample_index": sample_index,
                            "input_high_frequency_energy_ratio": input_ratio,
                            "spectral_output_high_frequency_energy_ratio": output_ratio,
                            "frequency_quartile": int(
                                np.digitize(
                                    input_ratio,
                                    [
                                        thresholds["source_training_hf_q25"],
                                        thresholds["source_training_hf_q50"],
                                        thresholds["source_training_hf_q75"],
                                    ],
                                    right=True,
                                )
                                + 1
                            ),
                            "gust_rich": bool(masks["gust_sample"][sample_index]),
                        }
                    )

    metrics = pd.DataFrame(metric_rows)
    numeric = ["ws_rmse", "ws_mae", "wd_mae", "r2", "count"]
    metrics = add_target_means(metrics, ["variant", "seed"], numeric)
    metrics.to_csv(OUTPUT_ROOT / "spectral_phase_five_seed_metrics.csv", index=False)
    five_seed_summary(metrics, ["variant", "target"], ["ws_rmse", "ws_mae", "wd_mae", "r2"]).to_csv(
        OUTPUT_ROOT / "spectral_phase_five_seed_summary.csv", index=False
    )

    regimes = pd.DataFrame(regime_rows)
    regimes = add_target_means(regimes, ["variant", "seed", "regime"], numeric)
    regimes.to_csv(OUTPUT_ROOT / "spectral_frequency_regime_metrics.csv", index=False)
    five_seed_summary(regimes, ["variant", "target", "regime"], ["ws_rmse", "ws_mae", "wd_mae", "r2", "count"]).to_csv(
        OUTPUT_ROOT / "spectral_frequency_regime_summary.csv", index=False
    )

    gust = pd.DataFrame(gust_rows)
    gust = add_target_means(gust, ["variant", "seed", "regime", "gust_threshold"], numeric)
    gust.to_csv(OUTPUT_ROOT / "gust_subset_metrics.csv", index=False)
    five_seed_summary(gust, ["variant", "target", "regime"], ["ws_rmse", "ws_mae", "wd_mae", "r2", "count"]).to_csv(
        OUTPUT_ROOT / "gust_subset_summary.csv", index=False
    )
    pd.DataFrame(diagnostic_rows).to_csv(OUTPUT_ROOT / "frequency_diagnostic_data.csv", index=False)
    (OUTPUT_ROOT / "task2_runtime.json").write_text(
        json.dumps({"runtime_seconds": time.perf_counter() - started, "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2),
        encoding="utf-8",
    )
    print("TASK 2 complete", flush=True)


if __name__ == "__main__":
    main()
