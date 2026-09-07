import itertools
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_VERSION = "transfer_two_targets_v1"
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
OUTPUT_DIR = Path("comparison_outputs") / EXPERIMENT_VERSION / "significance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_SUFFIX_PREFIX = "rev2_seed"


def parse_seeds():
    raw_value = os.getenv("EDGEWIND_REPEAT_SEEDS", "").strip()
    if not raw_value:
        return DEFAULT_SEEDS
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def seed_suffix(seed):
    prefix = os.getenv("EDGEWIND_REPEAT_SUFFIX_PREFIX", DEFAULT_SUFFIX_PREFIX).strip()
    return f"{prefix}_{seed}"


def mean_std_text(values):
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.3f} +/- {values.std(ddof=1):.3f}" if len(values) > 1 else f"{values.mean():.3f}"


def average_rank(abs_values):
    order = sorted(range(len(abs_values)), key=lambda idx: abs_values[idx])
    ranks = [0.0] * len(abs_values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and math.isclose(abs_values[order[end]], abs_values[order[cursor]], rel_tol=0, abs_tol=1e-12):
            end += 1
        rank_value = (cursor + 1 + end) / 2.0
        for idx in order[cursor:end]:
            ranks[idx] = rank_value
        cursor = end
    return ranks


def exact_wilcoxon_less(proposed_values, baseline_values):
    """One-sided exact signed-rank test for lower proposed metric values."""
    diffs = [float(p) - float(b) for p, b in zip(proposed_values, baseline_values)]
    diffs = [diff for diff in diffs if not math.isclose(diff, 0.0, rel_tol=0, abs_tol=1e-12)]
    n = len(diffs)
    if n == 0:
        return float("nan")

    ranks = average_rank([abs(diff) for diff in diffs])
    observed_positive = sum(rank for rank, diff in zip(ranks, diffs) if diff > 0)

    possible_positive = []
    for signs in itertools.product([0, 1], repeat=n):
        possible_positive.append(sum(rank for rank, sign in zip(ranks, signs) if sign == 1))
    return sum(value <= observed_positive + 1e-12 for value in possible_positive) / len(possible_positive)


def significance_marker(p_value):
    if pd.isna(p_value):
        return ""
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def read_seed_csv(path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def collect_main_comparison(seeds):
    rows = []
    for seed in seeds:
        suffix = seed_suffix(seed)
        edgewind_path = Path("weights") / EXPERIMENT_VERSION / "edgewind" / suffix / "edgewind_summary.csv"
        edgewind_df = read_seed_csv(edgewind_path)
        if edgewind_df is not None:
            transfer_df = edgewind_df[edgewind_df["experiment"] == "transfer"]
            rows.append(
                {
                    "seed": seed,
                    "model": "EdgeWind-Mamba",
                    "ws_rmse": transfer_df["test_ws_rmse"].mean(),
                    "ws_mae": transfer_df["test_ws_mae"].mean(),
                    "wd_mae": transfer_df["test_wd_mae"].mean(),
                }
            )

        baseline_path = Path("weights") / EXPERIMENT_VERSION / "baselines_v2" / suffix / "baseline_training_summary.csv"
        baseline_df = read_seed_csv(baseline_path)
        if baseline_df is not None:
            baseline_df = baseline_df[baseline_df["experiment"] == "transfer"]
            for model_name, group_df in baseline_df.groupby("model"):
                rows.append(
                    {
                        "seed": seed,
                        "model": model_name,
                        "ws_rmse": group_df["test_ws_rmse"].mean(),
                        "ws_mae": group_df["test_ws_mae"].mean(),
                        "wd_mae": group_df["test_wd_mae"].mean(),
                    }
                )
    return pd.DataFrame(rows)


def summarize_against_edgewind(df, key_col="model", metric="ws_rmse"):
    if df.empty or "EdgeWind-Mamba" not in set(df[key_col]):
        return pd.DataFrame()

    edgewind = df[df[key_col] == "EdgeWind-Mamba"].set_index("seed")[metric]
    rows = []
    for name, group_df in df.groupby(key_col):
        metric_series = group_df.set_index("seed")[metric]
        shared_seeds = sorted(set(edgewind.index) & set(metric_series.index))
        values = [metric_series.loc[seed] for seed in shared_seeds]
        edge_values = [edgewind.loc[seed] for seed in shared_seeds]
        p_value = float("nan") if name == "EdgeWind-Mamba" else exact_wilcoxon_less(edge_values, values)
        rows.append(
            {
                key_col: name,
                "n_seeds": len(shared_seeds),
                f"{metric}_mean": np.mean(values) if values else float("nan"),
                f"{metric}_std": np.std(values, ddof=1) if len(values) > 1 else float("nan"),
                f"{metric}_mean_std": mean_std_text(values) if values else "",
                "wilcoxon_p_vs_edgewind": p_value,
                "sig": significance_marker(p_value),
            }
        )
    return pd.DataFrame(rows).sort_values(f"{metric}_mean").reset_index(drop=True)


def collect_variant_table(seeds, relative_path, key_col, metric_cols):
    rows = []
    for seed in seeds:
        suffix = seed_suffix(seed)
        csv_path = Path("comparison_outputs") / EXPERIMENT_VERSION / relative_path / suffix / f"{relative_path}_summary.csv"
        df = read_seed_csv(csv_path)
        if df is None:
            continue
        for name, group_df in df.groupby(key_col):
            row = {"seed": seed, key_col: name}
            for metric_col in metric_cols:
                row[metric_col] = group_df[metric_col].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    seeds = parse_seeds()
    main_df = collect_main_comparison(seeds)
    main_df.to_csv(OUTPUT_DIR / "repeated_main_raw.csv", index=False, encoding="utf-8-sig")
    main_summary = summarize_against_edgewind(main_df, key_col="model", metric="ws_rmse")
    main_summary.to_csv(OUTPUT_DIR / "repeated_main_wilcoxon.csv", index=False, encoding="utf-8-sig")

    ablation_df = collect_variant_table(
        seeds,
        "ablation",
        "variant_name",
        ["test_ws_rmse", "test_ws_mae", "test_wd_mae", "test_extreme_ws_rmse", "test_r2"],
    )
    ablation_df.to_csv(OUTPUT_DIR / "repeated_ablation_raw.csv", index=False, encoding="utf-8-sig")

    loss_df = collect_variant_table(
        seeds,
        "loss_ablation",
        "loss_name",
        ["test_ws_rmse", "test_ws_mae", "test_wd_mae", "test_extreme_ws_rmse", "test_r2"],
    )
    loss_df.to_csv(OUTPUT_DIR / "repeated_loss_raw.csv", index=False, encoding="utf-8-sig")

    print(f"Saved repeated-significance summaries to {OUTPUT_DIR.resolve()}")
    if main_summary.empty:
        print("Main comparison summary is empty. Run run_repeated_core_experiments.py first.")
    else:
        print(main_summary)


if __name__ == "__main__":
    main()
