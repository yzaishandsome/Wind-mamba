"""Regenerate the locked clean-conformal Fig. 9 from checked-in plot data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "revision"
THRESHOLD = 10.580439745930407


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "generated_figures" / "nature_fig9_conformal_interval_revision.pdf")
    args = parser.parse_args()

    history = pd.read_csv(SOURCE / "fig9_recent_history.csv")
    case_npz = np.load(SOURCE / "fig9_candidate_sd1042.npz")
    case = {key: case_npz[key].astype(float) for key in case_npz.files}
    metadata = json.loads((SOURCE / "fig9_candidate_metadata.json").read_text(encoding="utf-8"))
    future_true = case["future_true"]
    if not np.all(future_true > THRESHOLD):
        raise RuntimeError("The locked chronological case no longer satisfies the source-training p95 rule.")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.0,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    x_history = history["relative_time_min"].to_numpy(dtype=int)
    recent = history["wind_speed_mps"].to_numpy(dtype=float)
    x_future = np.arange(10, 70, 10)
    x_observed = np.concatenate(([0], x_future))
    y_observed = np.concatenate(([recent[-1]], future_true))

    fig, ax = plt.subplots(figsize=(5.55, 3.05))
    ax.plot(x_history, recent, color="#202020", linewidth=1.35, label="Observed recent history", zorder=4)
    ax.plot(x_observed, y_observed, color="#202020", marker="o", markersize=3.2, linewidth=1.35,
            label="Future ground truth", zorder=5)
    ax.fill_between(x_future, case["future_lower"], case["future_upper"], color="#8fbadb", alpha=0.52,
                    edgecolor="#6fa5cd", linewidth=0.65, label="Clean 90% interval", zorder=1)
    ax.plot(x_future, case["future_pred"], color="#001f8f", marker="s", markersize=3.2, linestyle="--",
            linewidth=1.35, label="Wind-Mamba point forecast", zorder=6)
    ax.axvline(0, color="#e67e00", linestyle="-.", linewidth=1.05, zorder=3)
    ax.axhline(THRESHOLD, color="#b23a3a", linestyle="--", linewidth=0.9, alpha=0.9, zorder=2)

    ymin = min(recent.min(), future_true.min(), case["future_lower"].min(), THRESHOLD) - 0.42
    ymax = max(recent.max(), future_true.max(), case["future_upper"].max(), THRESHOLD) + 0.45
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-53, 63)
    ax.text(0.025, 0.955, "Earlier history omitted\n(30 of 36 input steps not shown)", transform=ax.transAxes,
            fontsize=6.7, color="#666666", style="italic", va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.5), zorder=8)
    ax.text(2.5, ymax - 0.12, "Forecast starts", color="#d55e00", fontsize=7.0, va="top", ha="left")
    ax.text(60, THRESHOLD + 0.07, "Source-training p95 = 10.58 m/s", ha="right", va="bottom",
            fontsize=6.5, color="#8c2d2d")
    ax.text(0.025, 0.075,
            f"SD1042 aggregate PICP: {metadata['sd1042_overall_picp_percent']:.2f}%\n"
            f"SD1042 aggregate MPIW: {metadata['sd1042_overall_mpiw_mps']:.2f} m/s",
            transform=ax.transAxes, fontsize=6.8, va="bottom", ha="left",
            bbox=dict(facecolor="#f8fbff", edgecolor="#c8d3dd", boxstyle="round,pad=0.25", alpha=0.96), zorder=8)
    ticks = np.arange(-50, 70, 10)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick:+d}" if tick != 0 else "0 (Now)" for tick in ticks], fontsize=7.0)
    ax.set_xlabel("Relative time (min)", fontsize=8.0)
    ax.set_ylabel("Wind speed (m/s)", fontsize=8.0)
    ax.tick_params(axis="y", labelsize=7.0)
    ax.grid(True, color="#dddddd", linestyle="--", linewidth=0.45, alpha=0.8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=6.7,
              frameon=False, handlelength=2.4)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.96, bottom=0.29)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
