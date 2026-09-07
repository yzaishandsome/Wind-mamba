"""Run an explicitly selected residual-bound variant/seed subset."""

from __future__ import annotations

import argparse
from pathlib import Path

from revision_core import (
    DEVICE,
    ROUND_ROOT,
    build_loss,
    source_training_statistics,
    standard_source_datasets,
    train_transfer_variant,
    write_code_snapshot,
)


VARIANTS = {
    "unbounded": "unbounded",
    "fixed6": "fixed6",
    "training_derived_component": "training_derived",
    "learnable_global_component": "learnable",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    output_root = ROUND_ROOT / "residual_bound"
    write_code_snapshot(output_root / f"code_snapshot_worker_{args.variant}.json", [Path(__file__)])
    source_train, _ = standard_source_datasets()
    statistics = source_training_statistics(source_train)
    criterion = build_loss("smoothl1_dircos", wd_weight=1.0).to(DEVICE)
    for seed in args.seeds:
        train_transfer_variant(
            variant=args.variant,
            model_kind=VARIANTS[args.variant],
            seed=seed,
            output_root=output_root,
            criterion=criterion,
            bound_u=statistics["source_training_abs_delta_u_p99"],
            bound_v=statistics["source_training_abs_delta_v_p99"],
        )
    print(f"Worker complete: {args.variant} seeds={args.seeds}", flush=True)


if __name__ == "__main__":
    main()
