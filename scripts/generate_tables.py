"""Regenerate numeric source tables used by the ASOC revision manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUBMITTED = ROOT / "results" / "submitted"
REVISION = ROOT / "results" / "revision"


def save(frame: pd.DataFrame, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{name}.csv"
    tex_path = output_dir / f"{name}.tex"
    frame.to_csv(csv_path, index=False)
    tex_path.write_text(simple_latex(frame), encoding="utf-8")
    print(csv_path)


def latex_escape(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def simple_latex(frame: pd.DataFrame) -> str:
    columns = [latex_escape(column) for column in frame.columns]
    lines = [r"\begin{tabular}{" + "l" * len(columns) + "}", " & ".join(columns) + r" \\"]
    for row in frame.itertuples(index=False, name=None):
        lines.append(" & ".join(latex_escape(value) for value in row) + r" \\")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def table4() -> pd.DataFrame:
    raw = pd.read_csv(SUBMITTED / "repeated_main_raw.csv")
    summary = raw.groupby("model", sort=False)[["ws_rmse", "ws_mae", "wd_mae"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    persistence = pd.read_csv(SUBMITTED / "persistence_baseline_summary.csv")
    deterministic = [{
        "model": "Persistence",
        "ws_rmse_mean": persistence["WS-RMSE"].mean(),
        "ws_mae_mean": persistence["WS-MAE"].mean(),
        "wd_mae_mean": persistence["WD-MAE"].mean(),
    }]
    statistical = pd.read_csv(SUBMITTED / "statistical_baseline_summary.csv")
    for model, group in statistical.groupby("model", sort=False):
        deterministic.append({
            "model": model,
            "ws_rmse_mean": group["test_ws_rmse"].mean(),
            "ws_mae_mean": group["test_ws_mae"].mean(),
            "wd_mae_mean": group["test_wd_mae"].mean(),
        })
    return pd.concat([summary, pd.DataFrame(deterministic)], ignore_index=True)


def table8() -> pd.DataFrame:
    """Recompute Table 8 from target-level seed records with an explicit row map."""
    raw = pd.read_csv(SUBMITTED / "repeated_core_ablation_raw.csv")
    metric_map = {
        "test_ws_rmse": "ws_rmse",
        "test_ws_mae": "ws_mae",
        "test_wd_mae": "wd_mae",
        "test_r2": "r2",
    }
    row_map = [
        ("full", "Full Wind-Mamba"),
        ("wo_persistence", "w/o Persistence Prior"),
        ("wo_fft", "w/o FFT Branch"),
        ("wo_boat_embedding", "w/o Vessel Embedding"),
        ("wo_gated_fusion", "w/o Gated Fusion"),
        ("mlp_decoder", "w/o GRU Decoder (MLP)"),
    ]

    target_means = raw.groupby(["variant_key", "seed"], as_index=False)[list(metric_map)].mean()
    rows = []
    for variant_key, manuscript_label in row_map:
        selected = target_means[target_means["variant_key"] == variant_key]
        if selected["seed"].tolist() != [42, 43, 44, 45, 46]:
            raise RuntimeError(f"Incomplete Table 8 seed records for {variant_key}")
        row = {"variant_key": variant_key, "manuscript_label": manuscript_label}
        for source, output in metric_map.items():
            row[f"{output}_mean"] = selected[source].mean()
            row[f"{output}_sample_std"] = selected[source].std(ddof=1)
        rows.append(row)

    frame = pd.DataFrame(rows)
    released = pd.read_csv(SUBMITTED / "repeated_core_ablation_summary.csv").set_index("variant_key")
    for row in frame.itertuples(index=False):
        expected = released.loc[row.variant_key]
        for metric in metric_map.values():
            if not np.isclose(getattr(row, f"{metric}_mean"), expected[f"{metric}_mean"], atol=1e-12):
                raise RuntimeError(f"Table 8 mean mismatch for {row.variant_key}/{metric}")
            if not np.isclose(getattr(row, f"{metric}_sample_std"), expected[f"{metric}_std"], atol=1e-12):
                raise RuntimeError(f"Table 8 sample-std mismatch for {row.variant_key}/{metric}")
    return frame


def table15() -> pd.DataFrame:
    """Assemble the three manuscript rows and recover fixed parameter counts."""
    spectral_summary = pd.read_csv(REVISION / "spectral_phase_five_seed_summary.csv")
    spectral_metrics = pd.read_csv(REVISION / "spectral_phase_five_seed_metrics.csv")
    closest_summary = pd.read_csv(REVISION / "closest_prior_art_five_seed_summary.csv")
    closest_metrics = pd.read_csv(REVISION / "closest_prior_art_five_seed_metrics.csv")

    rows = []
    full = spectral_summary[
        (spectral_summary["variant"] == "full_real_imag")
        & (spectral_summary["target"] == "target_mean")
    ].iloc[0]
    full_params = spectral_metrics.loc[
        (spectral_metrics["variant"] == "full_real_imag")
        & (spectral_metrics["target"] != "target_mean"),
        "parameter_count",
    ].dropna().unique()
    if full_params.size != 1:
        raise RuntimeError("Wind-Mamba parameter count is not uniquely defined")
    rows.append(_table15_row("Wind-Mamba", full, float(full_params[0])))

    for loss_variant, label in (
        ("paper_kfl_dircos", "DPFMformer + original FK loss"),
        ("common_smoothl1_dircos", "DPFMformer + common loss"),
    ):
        selected = closest_summary[
            (closest_summary["loss_variant"] == loss_variant)
            & (closest_summary["target"] == "target_mean")
        ].iloc[0]
        params = closest_metrics.loc[
            (closest_metrics["loss_variant"] == loss_variant)
            & (closest_metrics["target"] != "target_mean"),
            "parameter_count",
        ].dropna().unique()
        if params.size != 1:
            raise RuntimeError(f"DPFMformer parameter count is not uniquely defined for {loss_variant}")
        rows.append(_table15_row(label, selected, float(params[0])))
    return pd.DataFrame(rows)


def _table15_row(label: str, row: pd.Series, parameter_count: float) -> dict:
    return {
        "model": label,
        "ws_rmse_mean": row["ws_rmse_mean"],
        "ws_rmse_sample_std": row["ws_rmse_sample_std"],
        "ws_mae_mean": row["ws_mae_mean"],
        "ws_mae_sample_std": row["ws_mae_sample_std"],
        "wd_mae_mean": row["wd_mae_mean"],
        "wd_mae_sample_std": row["wd_mae_sample_std"],
        "r2_mean": row["r2_mean"],
        "r2_sample_std": row["r2_sample_std"],
        "parameters_millions": parameter_count / 1_000_000.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "generated_tables")
    args = parser.parse_args()

    save(table4(), args.output_dir, "table_4_main_benchmark")
    save(table8(), args.output_dir, "table_8_module_ablation")

    weighted = pd.read_csv(REVISION / "weighted_loss_revision_summary.csv")
    save(weighted[weighted["target"] == "target_mean"], args.output_dir, "table_9_loss_ablation")

    save(pd.read_csv(REVISION / "conformal_clean_final_metrics.csv"), args.output_dir, "table_11_conformal_metrics")
    save(pd.read_csv(REVISION / "chronological_split_counts.csv"), args.output_dir, "table_11_split_composition")

    horizon = pd.read_csv(REVISION / "horizon_metrics_five_seed_summary.csv")
    save(horizon[horizon["target"] == "target_mean"], args.output_dir, "table_12_horizon_metrics")

    spectral = pd.read_csv(REVISION / "spectral_phase_five_seed_summary.csv")
    save(spectral[spectral["target"] == "target_mean"], args.output_dir, "table_13_spectral_overall")
    save(pd.read_csv(REVISION / "spectral_frequency_regime_summary.csv"), args.output_dir, "table_13_frequency_regimes")
    save(pd.read_csv(REVISION / "gust_subset_summary.csv"), args.output_dir, "table_13_gust_subset")

    bounds = pd.read_csv(REVISION / "residual_bound_five_seed_summary.csv")
    save(bounds[bounds["target"] == "target_mean"], args.output_dir, "table_14_residual_bounds")
    save(pd.read_csv(REVISION / "residual_bound_regime_summary.csv"), args.output_dir, "table_14_residual_regimes")

    save(table15(), args.output_dir, "table_15_dpfmformer")


if __name__ == "__main__":
    main()
