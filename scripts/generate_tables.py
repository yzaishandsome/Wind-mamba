"""Regenerate numeric source tables used by the ASOC revision manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "generated_tables")
    args = parser.parse_args()

    save(table4(), args.output_dir, "table_4_main_benchmark")
    save(pd.read_csv(SUBMITTED / "repeated_core_ablation_summary.csv"), args.output_dir, "table_8_module_ablation")

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

    closest = pd.read_csv(REVISION / "closest_prior_art_five_seed_summary.csv")
    save(closest[closest["target"] == "target_mean"], args.output_dir, "table_15_dpfmformer")


if __name__ == "__main__":
    main()
