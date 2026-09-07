"""Aggregate residual-bound results after all independent workers finish."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from revision_core import (
    ROUND_ROOT,
    SEEDS,
    TARGET_BOAT_FILES,
    add_target_means,
    five_seed_summary,
    metrics_from_predictions,
    regime_masks,
    source_training_statistics,
    standard_source_datasets,
)
from run_task3_residual_bounds import NEW_VARIANTS, fixed4_payload, trained_payload
from experiment_config import boat_tag


OUTPUT_ROOT = ROUND_ROOT / "residual_bound"


def main():
    source_train, _ = standard_source_datasets()
    scaler = source_train.scaler
    thresholds = source_training_statistics(source_train)
    pd.DataFrame([thresholds]).to_csv(OUTPUT_ROOT / "training_residual_statistics.csv", index=False)
    missing = []
    for variant, _ in NEW_VARIANTS:
        for seed in SEEDS:
            expected = OUTPUT_ROOT / "runs" / variant / f"seed_{seed}" / "target_metrics.csv"
            if not expected.exists():
                missing.append(str(expected))
    if missing:
        raise RuntimeError("Missing completed Task 3 results:\n" + "\n".join(missing))

    metric_rows, regime_rows, bound_rows = [], [], []
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
    (OUTPUT_ROOT / "task3_finalize_metadata.json").write_text(
        json.dumps({"completed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "all_variants_and_seeds_present": True}, indent=2),
        encoding="utf-8",
    )
    print("TASK 3 aggregation complete")


if __name__ == "__main__":
    main()
