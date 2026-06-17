"""Chronological interval calibration for Wind-Mamba wind-speed forecasts.

This script reproduces the post-hoc interval-calibration analysis described in
the manuscript. It uses validation-calibration residuals to construct pointwise,
horizon-wise wind-speed prediction intervals and evaluates PICP, MPIW, and the
Winkler score on the chronological test split.

Important scope notes
---------------------
1. The intervals are wind-speed intervals only; wind-direction uncertainty is not
   calibrated here.
2. The intervals are pointwise and horizon-wise. They should not be interpreted
   as joint coverage guarantees for the complete six-step trajectory.
3. The target-specific margin is selected only from the chronological
   validation-calibration split. It is an empirical operating-margin factor for
   non-stationary onboard wind measurements, not a test-tuned parameter and not
   a new conformal inference method.
4. Visualization-case export is optional and is not used for calibration, model
   selection, or quantitative evaluation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from data_provider import USVDataset
from experiment_config import (
    ALL_BOAT_FILES,
    EXPERIMENT_VERSION,
    SOURCE_BOAT_FILES,
    TARGET_BOAT_FILES,
    boat_ids_for,
    boat_tag,
)
from model import DEFAULT_UPPER_TAIL_WS_THRESHOLD, WindMambaModel


U_COMPONENT_INDEX = 5
V_COMPONENT_INDEX = 6


@dataclass(frozen=True)
class IntervalConfig:
    seq_len: int = 36
    pred_len: int = 6
    hidden_dim: int = 96
    alpha: float = 0.10
    eval_batch_size: int = 128
    upper_tail_ws_threshold: float = DEFAULT_UPPER_TAIL_WS_THRESHOLD
    margin_min: float = 1.00
    margin_max: float = 2.50
    margin_step: float = 0.02
    margin_blocks: int = 4
    block_coverage_tolerance: float = 2.0
    visible_history_steps: int = 6
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_version: str = EXPERIMENT_VERSION
    checkpoint_root: str = str(Path("weights") / EXPERIMENT_VERSION / "wind_mamba")
    output_root: str = str(Path("interval_outputs") / EXPERIMENT_VERSION)
    export_cases: bool = True
    top_visualization_cases: int = 5

    @property
    def target_coverage(self) -> float:
        return 1.0 - self.alpha

    @property
    def margin_grid(self) -> np.ndarray:
        stop = self.margin_max + self.margin_step / 2.0
        return np.round(np.arange(self.margin_min, stop, self.margin_step), 2)


def parse_args() -> IntervalConfig:
    parser = argparse.ArgumentParser(
        description="Run chronological pointwise interval calibration for trained Wind-Mamba checkpoints.",
    )
    parser.add_argument("--seq-len", type=int, default=36)
    parser.add_argument("--pred-len", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--upper-tail-ws-threshold", type=float, default=DEFAULT_UPPER_TAIL_WS_THRESHOLD)
    parser.add_argument("--margin-min", type=float, default=1.00)
    parser.add_argument("--margin-max", type=float, default=2.50)
    parser.add_argument("--margin-step", type=float, default=0.02)
    parser.add_argument("--margin-blocks", type=int, default=4)
    parser.add_argument("--block-coverage-tolerance", type=float, default=2.0)
    parser.add_argument("--visible-history-steps", type=int, default=6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--checkpoint-root",
        default=str(Path("weights") / EXPERIMENT_VERSION / "wind_mamba"),
        help="Directory containing transfer_<target>_best.pth checkpoints.",
    )
    parser.add_argument(
        "--output-root",
        default=str(Path("interval_outputs") / EXPERIMENT_VERSION),
        help="Directory for interval CSV/NPZ/JSON outputs.",
    )
    parser.add_argument("--no-export-cases", action="store_true", help="Skip visualization-case export.")
    parser.add_argument("--top-visualization-cases", type=int, default=5)
    args = parser.parse_args()
    return IntervalConfig(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        hidden_dim=args.hidden_dim,
        alpha=args.alpha,
        eval_batch_size=args.eval_batch_size,
        upper_tail_ws_threshold=args.upper_tail_ws_threshold,
        margin_min=args.margin_min,
        margin_max=args.margin_max,
        margin_step=args.margin_step,
        margin_blocks=args.margin_blocks,
        block_coverage_tolerance=args.block_coverage_tolerance,
        visible_history_steps=args.visible_history_steps,
        device=args.device,
        checkpoint_root=args.checkpoint_root,
        output_root=args.output_root,
        export_cases=not args.no_export_cases,
        top_visualization_cases=args.top_visualization_cases,
    )


def extract_feature_stats(scaler: object) -> dict[str, float]:
    """Extract feature normalization statistics needed by the residual decoder."""
    required = max(U_COMPONENT_INDEX, V_COMPONENT_INDEX) + 1
    if not hasattr(scaler, "mean_") or len(scaler.mean_) < required:
        raise ValueError("The fitted scaler does not contain the expected wind-component feature statistics.")
    return {
        "u_mean": float(scaler.mean_[U_COMPONENT_INDEX]),
        "u_scale": float(scaler.scale_[U_COMPONENT_INDEX]),
        "v_mean": float(scaler.mean_[V_COMPONENT_INDEX]),
        "v_scale": float(scaler.scale_[V_COMPONENT_INDEX]),
    }


def build_model(config: IntervalConfig, feature_stats: dict[str, float]) -> WindMambaModel:
    model = WindMambaModel(
        in_dim=10,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        hidden_dim=config.hidden_dim,
        num_boats=len(ALL_BOAT_FILES),
        feature_stats=feature_stats,
    )
    return model.to(torch.device(config.device))


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run main.py first or pass --checkpoint-root to the directory containing trained checkpoints."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint


def build_source_scaler(config: IntervalConfig):
    source_ids = boat_ids_for(SOURCE_BOAT_FILES)
    return USVDataset(
        SOURCE_BOAT_FILES,
        global_boat_ids=source_ids,
        flag="train",
        seq_len=config.seq_len,
        pred_len=config.pred_len,
    ).scaler


def build_target_dataset(target_file: str, flag: str, scaler: object, config: IntervalConfig) -> USVDataset:
    target_id = boat_ids_for([target_file])[0]
    return USVDataset(
        [target_file],
        global_boat_ids=[target_id],
        flag=flag,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        scaler=scaler,
    )


def make_loader(dataset: USVDataset, config: IntervalConfig) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=0,
    )


def history_speed_from_inputs(batch_x: torch.Tensor, scaler: object) -> np.ndarray:
    x_np = batch_x.detach().cpu().numpy()
    u = x_np[..., U_COMPONENT_INDEX] * scaler.scale_[U_COMPONENT_INDEX] + scaler.mean_[U_COMPONENT_INDEX]
    v = x_np[..., V_COMPONENT_INDEX] * scaler.scale_[V_COMPONENT_INDEX] + scaler.mean_[V_COMPONENT_INDEX]
    return np.sqrt(u**2 + v**2 + 1e-8)


def collect_speed_outputs(
    model: torch.nn.Module,
    dataset: USVDataset,
    scaler: object,
    config: IntervalConfig,
) -> dict[str, np.ndarray]:
    history_ws, ws_true, ws_pred = [], [], []
    device = torch.device(config.device)
    with torch.no_grad():
        for batch_x, batch_y, boat_id in make_loader(dataset, config):
            batch_x = batch_x.to(device).float()
            boat_id = boat_id.to(device).long()
            pred_ws, _, _ = model(batch_x, boat_id)
            history_ws.append(history_speed_from_inputs(batch_x, scaler))
            ws_true.append(batch_y[..., 0].cpu().numpy())
            ws_pred.append(pred_ws.squeeze(-1).cpu().numpy())

    payload = {
        "history_ws": np.concatenate(history_ws, axis=0),
        "ws_true": np.concatenate(ws_true, axis=0),
        "ws_pred": np.concatenate(ws_pred, axis=0),
    }
    validate_physical_wind_speed(payload["ws_true"])
    return payload


def validate_physical_wind_speed(ws_true: np.ndarray) -> None:
    """Guard against accidentally evaluating normalized wind-speed targets."""
    if not np.all(np.isfinite(ws_true)):
        raise ValueError("Wind-speed target contains non-finite values.")
    if np.nanmax(ws_true) > 60.0 or np.nanmean(ws_true) <= 0.0:
        raise ValueError(
            "Wind-speed target does not look like physical m/s values. "
            "Check data_provider.py target scaling before interval calibration."
        )


def conformal_quantiles(abs_residuals: np.ndarray, alpha: float) -> np.ndarray:
    """Horizon-wise empirical conformal residual quantiles."""
    sorted_residuals = np.sort(abs_residuals, axis=0)
    rank = int(np.ceil((sorted_residuals.shape[0] + 1) * (1.0 - alpha))) - 1
    rank = max(0, min(rank, sorted_residuals.shape[0] - 1))
    return sorted_residuals[rank]


def build_intervals(ws_pred: np.ndarray, quantiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quantiles = quantiles.reshape(1, -1)
    lower = np.maximum(ws_pred - quantiles, 0.0)
    upper = ws_pred + quantiles
    return lower, upper


def winkler_score(ws_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> np.ndarray:
    score = upper - lower
    below = ws_true < lower
    above = ws_true > upper
    score[below] += (2.0 / alpha) * (lower[below] - ws_true[below])
    score[above] += (2.0 / alpha) * (ws_true[above] - upper[above])
    return score


def interval_metrics(ws_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> dict[str, float]:
    return {
        "picp": 100.0 * float(np.mean((ws_true >= lower) & (ws_true <= upper))),
        "mpiw": float(np.mean(upper - lower)),
        "winkler": float(np.mean(winkler_score(ws_true, lower, upper, alpha))),
    }


def interval_metrics_for_mask(
    ws_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
    mask: np.ndarray,
) -> dict[str, float | int]:
    if not np.any(mask):
        return {"count": 0, "picp": float("nan"), "mpiw": float("nan"), "winkler": float("nan")}
    return {"count": int(np.sum(mask)), **interval_metrics(ws_true[mask], lower[mask], upper[mask], alpha)}


def split_inner_calibration(payload: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    split_idx = max(1, payload["ws_true"].shape[0] // 2)
    return (
        {key: value[:split_idx] for key, value in payload.items()},
        {key: value[split_idx:] for key, value in payload.items()},
    )


def select_margin_from_validation(
    val_payload: dict[str, np.ndarray],
    config: IntervalConfig,
) -> tuple[float, np.ndarray, list[dict[str, float]]]:
    """Select a static operating-margin multiplier without using test samples."""
    inner_calib, inner_select = split_inner_calibration(val_payload)
    base_quantiles = conformal_quantiles(
        np.abs(inner_calib["ws_pred"] - inner_calib["ws_true"]),
        config.alpha,
    )

    sensitivity_rows = []
    best_margin = float(config.margin_grid[-1])
    for margin in config.margin_grid:
        lower, upper = build_intervals(inner_select["ws_pred"], base_quantiles * margin)
        metrics = interval_metrics(inner_select["ws_true"], lower, upper, config.alpha)

        block_picps = []
        block_indices = np.array_split(np.arange(inner_select["ws_true"].shape[0]), config.margin_blocks)
        for block_idx in block_indices:
            block_lower, block_upper = build_intervals(inner_select["ws_pred"][block_idx], base_quantiles * margin)
            block_metrics = interval_metrics(inner_select["ws_true"][block_idx], block_lower, block_upper, config.alpha)
            block_picps.append(block_metrics["picp"])

        min_block_picp = float(np.min(block_picps))
        sensitivity_rows.append(
            {
                "margin": float(margin),
                "selection_picp": metrics["picp"],
                "selection_mpiw": metrics["mpiw"],
                "selection_winkler": metrics["winkler"],
                "min_block_picp": min_block_picp,
            }
        )
        if (
            metrics["picp"] >= config.target_coverage * 100.0
            and min_block_picp >= config.target_coverage * 100.0 - config.block_coverage_tolerance
        ):
            best_margin = float(margin)
            break

    final_quantiles = conformal_quantiles(
        np.abs(val_payload["ws_pred"] - val_payload["ws_true"]),
        config.alpha,
    )
    return best_margin, final_quantiles * best_margin, sensitivity_rows


def local_case_metrics(
    ws_true: np.ndarray,
    ws_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    return {
        "future_mean_ws": float(np.mean(ws_true)),
        "local_rmse": float(np.sqrt(np.mean((ws_pred - ws_true) ** 2))),
        "local_picp": 100.0 * float(np.mean((ws_true >= lower) & (ws_true <= upper))),
        "local_mpiw": float(np.mean(upper - lower)),
    }


def select_upper_tail_cases_for_visualization(
    target_payloads: dict[str, dict[str, np.ndarray]],
    config: IntervalConfig,
) -> list[dict[str, int | str]]:
    """Select representative upper-tail cases for plots only.

    This routine uses test-set observations only to export illustrative cases.
    It is not used for calibration, model selection, or quantitative evaluation.
    """
    selected_cases = []
    for target_tag, payload in target_payloads.items():
        ws_true = payload["test_ws_true"]
        ws_pred = payload["test_ws_pred"]
        ws_lower = payload["test_ws_lower"]
        ws_upper = payload["test_ws_upper"]

        candidate_mask = np.mean(ws_true, axis=1) > config.upper_tail_ws_threshold
        scored_indices = []
        for idx in np.where(candidate_mask)[0]:
            coverage = float(np.mean((ws_true[idx] >= ws_lower[idx]) & (ws_true[idx] <= ws_upper[idx])))
            local_rmse = float(np.sqrt(np.mean((ws_pred[idx] - ws_true[idx]) ** 2)))
            width = float(np.mean(ws_upper[idx] - ws_lower[idx]))
            score = 100.0 * coverage - 20.0 * local_rmse - 5.0 * width
            scored_indices.append((score, int(idx)))

        scored_indices.sort(key=lambda item: item[0], reverse=True)
        for rank, (_, idx) in enumerate(scored_indices[: config.top_visualization_cases], start=1):
            selected_cases.append({"target": target_tag, "index": idx, "rank": rank})
    return selected_cases


def save_case_payload(
    target_tag: str,
    payload: dict[str, np.ndarray],
    case_index: int,
    rank: int,
    quantiles: np.ndarray,
    target_metrics: dict[str, float],
    config: IntervalConfig,
    output_root: Path,
) -> None:
    case_metrics = local_case_metrics(
        payload["test_ws_true"][case_index],
        payload["test_ws_pred"][case_index],
        payload["test_ws_lower"][case_index],
        payload["test_ws_upper"][case_index],
        config.alpha,
    )
    np.savez(
        output_root / f"interval_case_{target_tag}_rank{rank}.npz",
        history_full=payload["test_history"][case_index],
        future_true=payload["test_ws_true"][case_index],
        future_pred=payload["test_ws_pred"][case_index],
        future_lower=payload["test_ws_lower"][case_index],
        future_upper=payload["test_ws_upper"][case_index],
    )
    metadata = {
        "target": target_tag,
        "rank": rank,
        "case_index": int(case_index),
        "seq_len": config.seq_len,
        "pred_len": config.pred_len,
        "visible_history_steps": config.visible_history_steps,
        "alpha": config.alpha,
        "target_coverage": config.target_coverage,
        "upper_tail_ws_threshold": config.upper_tail_ws_threshold,
        "interval_scope": "pointwise_horizon_wise_wind_speed",
        "case_selection_scope": "visualization_only_not_used_for_evaluation",
        "quantiles": [float(v) for v in quantiles],
        "overall_metrics": target_metrics,
        "case_metrics": case_metrics,
    }
    (output_root / f"interval_case_{target_tag}_rank{rank}_meta.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def weighted_average(values: Iterable[float], weights: Iterable[float]) -> float:
    values_arr = np.asarray(list(values), dtype=float)
    weights_arr = np.asarray(list(weights), dtype=float)
    valid = np.isfinite(values_arr) & np.isfinite(weights_arr) & (weights_arr > 0)
    if not np.any(valid):
        return float("nan")
    return float(np.sum(values_arr[valid] * weights_arr[valid]) / np.sum(weights_arr[valid]))


def append_macro_mean(summary_df: pd.DataFrame) -> pd.DataFrame:
    mean_row = {"target": "macro_mean"}
    for col in summary_df.columns:
        if col != "target":
            mean_row[col] = float(summary_df[col].mean())
    return pd.concat([summary_df, pd.DataFrame([mean_row])], ignore_index=True)


def append_stratified_means(stratified_df: pd.DataFrame) -> pd.DataFrame:
    rows = [stratified_df]
    aggregate_rows = []
    for regime_name, regime_df in stratified_df.groupby("regime", sort=False):
        macro_row = {
            "target": "macro_mean",
            "regime": regime_name,
            "count": int(regime_df["count"].sum()),
        }
        weighted_row = {
            "target": "weighted_mean",
            "regime": regime_name,
            "count": int(regime_df["count"].sum()),
        }
        for col in ["picp", "mpiw", "winkler"]:
            macro_row[col] = float(regime_df[col].mean())
            weighted_row[col] = weighted_average(regime_df[col], regime_df["count"])
        aggregate_rows.extend([macro_row, weighted_row])
    rows.append(pd.DataFrame(aggregate_rows))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    config = parse_args()
    device = torch.device(config.device)
    checkpoint_root = Path(config.checkpoint_root)
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    source_scaler = build_source_scaler(config)
    feature_stats = extract_feature_stats(source_scaler)

    summary_rows = []
    stratified_rows = []
    margin_sensitivity_rows = []
    target_payloads = {}

    for target_file in TARGET_BOAT_FILES:
        target_tag = boat_tag(target_file)
        model = build_model(config, feature_stats)
        checkpoint_path = checkpoint_root / f"transfer_{target_tag}_best.pth"
        load_checkpoint(model, checkpoint_path, device)

        val_payload = collect_speed_outputs(
            model,
            build_target_dataset(target_file, "val", source_scaler, config),
            source_scaler,
            config,
        )
        test_payload = collect_speed_outputs(
            model,
            build_target_dataset(target_file, "test", source_scaler, config),
            source_scaler,
            config,
        )

        margin, quantiles, sensitivity_rows = select_margin_from_validation(val_payload, config)
        for row in sensitivity_rows:
            row["target"] = target_tag
            margin_sensitivity_rows.append(row)

        ws_lower, ws_upper = build_intervals(test_payload["ws_pred"], quantiles)
        overall_metrics = interval_metrics(test_payload["ws_true"], ws_lower, ws_upper, config.alpha)

        upper_tail_mask = test_payload["ws_true"] > config.upper_tail_ws_threshold
        non_upper_tail_mask = ~upper_tail_mask
        for regime_name, regime_mask in (
            ("non_upper_tail", non_upper_tail_mask),
            ("upper_tail", upper_tail_mask),
        ):
            stratified_rows.append(
                {
                    "target": target_tag,
                    "regime": regime_name,
                    **interval_metrics_for_mask(test_payload["ws_true"], ws_lower, ws_upper, config.alpha, regime_mask),
                }
            )

        summary_row = {
            "target": target_tag,
            "margin": margin,
            "picp": overall_metrics["picp"],
            "mpiw": overall_metrics["mpiw"],
            "winkler": overall_metrics["winkler"],
        }
        summary_rows.append(summary_row)
        target_payloads[target_tag] = {
            "test_history": test_payload["history_ws"],
            "test_ws_true": test_payload["ws_true"],
            "test_ws_pred": test_payload["ws_pred"],
            "test_ws_lower": ws_lower,
            "test_ws_upper": ws_upper,
            "quantiles": quantiles,
            "metrics": summary_row,
        }
        print(
            f"[{target_tag.upper()}] PICP={overall_metrics['picp']:.2f}% | "
            f"MPIW={overall_metrics['mpiw']:.3f} | margin={margin:.2f}"
        )

    summary_df = pd.DataFrame(summary_rows)
    append_macro_mean(summary_df).to_csv(output_root / "interval_metrics_summary.csv", index=False)
    pd.DataFrame(margin_sensitivity_rows).to_csv(output_root / "interval_margin_sensitivity.csv", index=False)

    stratified_df = pd.DataFrame(stratified_rows)
    append_stratified_means(stratified_df).to_csv(output_root / "interval_metrics_stratified.csv", index=False)

    config_path = output_root / "interval_config.json"
    config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    if config.export_cases:
        selected_cases = select_upper_tail_cases_for_visualization(target_payloads, config)
        for case in selected_cases:
            target = str(case["target"])
            save_case_payload(
                target,
                target_payloads[target],
                int(case["index"]),
                int(case["rank"]),
                target_payloads[target]["quantiles"],
                target_payloads[target]["metrics"],
                config,
                output_root,
            )
            print(f"Extracted visualization case: {target.upper()} rank {case['rank']}")

    print(f"Interval outputs written to: {output_root}")


if __name__ == "__main__":
    main()
