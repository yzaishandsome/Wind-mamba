import os
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SAMPLING_INTERVAL = timedelta(minutes=10)
SAMPLING_RULE = "10min"
MIN_SEGMENT_COMPLETENESS = 0.95
MAX_INTERPOLATABLE_STEPS = 1
MIN_SEGMENT_LENGTH = 96 + 6

INPUT_FILES = [
    "sd1031.csv",
    "sd1033.csv",
    "sd1036.csv",
    "sd1040.csv",
    "sd1041.csv",
    "sd1042.csv",
    "sd1057.csv",
    "sd1069.csv",
    "sd1083.csv",
    "sd1091.csv",
]
OUTPUT_DIR = "processed_data"
SELECTION_RULE = "longest_segment_then_highest_completeness_then_fewer_interpolated_points"

MODEL_FEATURE_COLS = [
    "latitude",
    "longitude",
    "SOG",
    "COG",
    "HDG",
    "UWND_MEAN",
    "VWND_MEAN",
    "GUST_WND_MEAN",
    "TEMP_AIR_MEAN",
    "BARO_PRES_MEAN",
]

TIME_COL_CANDIDATES = ["time", "Time", "timestamp", "Timestamp", "datetime", "Datetime", "date", "Date"]
ANGLE_COLS = ["COG", "HDG", "WIND_FROM_MEAN"]
RAW_NUMERIC_COLS = [
    "latitude",
    "longitude",
    "SOG",
    "COG",
    "HDG",
    "ROLL_FILTERED_MEAN",
    "PITCH_FILTERED_MEAN",
    "WIND_FROM_MEAN",
    "WIND_SPEED_MEAN",
    "UWND_MEAN",
    "VWND_MEAN",
    "GUST_WND_MEAN",
    "TEMP_AIR_MEAN",
    "RH_MEAN",
    "BARO_PRES_MEAN",
    "WAVE_SIGNIFICANT_HEIGHT",
    "WS_TRUE",
    "WD_TRUE",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_time_column(df):
    for candidate in TIME_COL_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def circular_mean_deg(values):
    values = pd.Series(values).dropna()
    if values.empty:
        return np.nan
    radians = np.deg2rad(values.to_numpy(dtype=float))
    sin_mean = np.mean(np.sin(radians))
    cos_mean = np.mean(np.cos(radians))
    return (np.degrees(np.arctan2(sin_mean, cos_mean)) + 360.0) % 360.0


def aggregate_duplicate_timestamps(df):
    agg_map = {}
    for col in df.columns:
        if col == "time":
            continue
        if col in ANGLE_COLS:
            agg_map[col] = circular_mean_deg
        elif pd.api.types.is_numeric_dtype(df[col]):
            agg_map[col] = "mean"
        else:
            agg_map[col] = "first"
    return df.groupby("time", as_index=False).agg(agg_map)


def prepare_raw_dataframe(file_path):
    df = pd.read_csv(
        file_path,
        low_memory=False,
        skiprows=[1],
        na_values=["", " ", "NA", "N/A", "nan", "NaN", "None", "inf", "-inf", "NaN "],
    )

    time_col = find_time_column(df)
    if time_col is None:
        raise ValueError(f"No time column found in {file_path}")

    numeric_cols = [col for col in RAW_NUMERIC_COLS if col in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["time"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df = df.dropna(subset=["time"]).copy()
    df["time"] = df["time"].dt.round(SAMPLING_RULE)
    df = df.sort_values("time").reset_index(drop=True)
    df = aggregate_duplicate_timestamps(df)

    missing_required = [col for col in MODEL_FEATURE_COLS if col not in df.columns]
    if missing_required:
        raise ValueError(f"{file_path} is missing required columns: {missing_required}")

    return df


def interpolate_series(series, time_index):
    if not series.isna().any():
        return series

    filled = series.copy()
    for method in ("cubic", "time", "linear"):
        if not filled.isna().any():
            break
        try:
            if method == "time":
                time_series = pd.Series(filled.to_numpy(), index=time_index, name=series.name)
                time_series = time_series.interpolate(
                    method="time",
                    limit=MAX_INTERPOLATABLE_STEPS,
                    limit_area="inside",
                )
                filled = pd.Series(time_series.to_numpy(), index=series.index, name=series.name)
            else:
                filled = filled.interpolate(
                    method=method,
                    limit=MAX_INTERPOLATABLE_STEPS,
                    limit_area="inside",
                )
        except Exception:
            continue
    return filled


def interpolate_segment(seg_df, feature_cols):
    seg_df = seg_df.sort_values("time").reset_index(drop=True).copy()
    time_index = pd.DatetimeIndex(seg_df["time"])

    numeric_cols = [
        col
        for col in seg_df.columns
        if col not in {"time", "segment_id", "segment_completeness", "is_missing", "is_interpolated"}
        and pd.api.types.is_numeric_dtype(seg_df[col])
        and col not in {"WS_TRUE", "WD_TRUE"}
    ]

    circular_cols = [col for col in ANGLE_COLS if col in numeric_cols]
    linear_cols = [col for col in numeric_cols if col not in circular_cols]

    for col in circular_cols:
        radians = np.deg2rad(seg_df[col])
        seg_df[f"__{col}_sin"] = np.sin(radians)
        seg_df[f"__{col}_cos"] = np.cos(radians)

    interp_cols = linear_cols + [f"__{col}_sin" for col in circular_cols] + [f"__{col}_cos" for col in circular_cols]
    for col in interp_cols:
        seg_df[col] = interpolate_series(seg_df[col], time_index)

    for col in circular_cols:
        sin_col = f"__{col}_sin"
        cos_col = f"__{col}_cos"
        seg_df[col] = (np.degrees(np.arctan2(seg_df[sin_col], seg_df[cos_col])) + 360.0) % 360.0
        seg_df = seg_df.drop(columns=[sin_col, cos_col])

    seg_df = seg_df.dropna(subset=feature_cols).copy()
    seg_df["WS_TRUE"] = np.sqrt(seg_df["UWND_MEAN"] ** 2 + seg_df["VWND_MEAN"] ** 2 + 1e-8)
    seg_df["WD_TRUE"] = (np.degrees(np.arctan2(-seg_df["UWND_MEAN"], -seg_df["VWND_MEAN"])) + 360.0) % 360.0
    seg_df["is_interpolated"] = seg_df["is_interpolated"].astype(bool)
    seg_df["is_missing"] = False
    return seg_df.reset_index(drop=True)


def split_segments_by_long_gaps(merged_df):
    merged_df = merged_df.copy()
    run_id = (merged_df["is_missing"] != merged_df["is_missing"].shift(fill_value=False)).cumsum()
    merged_df["missing_run_id"] = run_id

    run_stats = merged_df.groupby("missing_run_id")["is_missing"].agg(["first", "size"])
    long_gap_run_ids = set(run_stats[(run_stats["first"]) & (run_stats["size"] > MAX_INTERPOLATABLE_STEPS)].index)
    merged_df["is_long_gap"] = merged_df["missing_run_id"].isin(long_gap_run_ids)
    merged_df["segment_break"] = merged_df["is_long_gap"] & ~merged_df["is_long_gap"].shift(fill_value=False)

    usable_df = merged_df.loc[~merged_df["is_long_gap"]].copy()
    if usable_df.empty:
        return []

    usable_df["segment_id"] = merged_df["segment_break"].cumsum()[~merged_df["is_long_gap"]].to_numpy()

    segments = []
    for _, seg_df in usable_df.groupby("segment_id", sort=True):
        seg_df = seg_df.drop(columns=["missing_run_id", "is_long_gap", "segment_break"]).reset_index(drop=True)
        segments.append(seg_df)
    return segments


def summarize_candidate(candidate_id, seg_df, seg_completeness):
    return {
        "candidate_id": candidate_id,
        "df": seg_df,
        "rows": len(seg_df),
        "start_time": seg_df["time"].iloc[0],
        "end_time": seg_df["time"].iloc[-1],
        "completeness": float(seg_completeness),
        "interpolated_rows": int(seg_df["is_interpolated"].sum()),
    }


def build_candidate_segments(merged_df):
    candidate_segments = []
    next_candidate_id = 0

    for seg_df in split_segments_by_long_gaps(merged_df):
        seg_completeness = 1.0 - seg_df["is_missing"].mean()
        if seg_completeness < MIN_SEGMENT_COMPLETENESS or len(seg_df) < MIN_SEGMENT_LENGTH:
            continue

        seg_df = seg_df.copy()
        seg_df["is_interpolated"] = seg_df["is_missing"].fillna(False).astype(bool)
        seg_df = interpolate_segment(seg_df, MODEL_FEATURE_COLS)
        if len(seg_df) < MIN_SEGMENT_LENGTH:
            continue

        time_diff = seg_df["time"].diff().dt.total_seconds().div(60)
        continuity_break = (time_diff.notna()) & (time_diff != 10)
        subsegment_ids = continuity_break.cumsum()

        for _, subseg in seg_df.groupby(subsegment_ids, sort=True):
            subseg = subseg.reset_index(drop=True)
            if len(subseg) < MIN_SEGMENT_LENGTH:
                continue

            candidate_segments.append(summarize_candidate(next_candidate_id, subseg, seg_completeness))
            next_candidate_id += 1

    return candidate_segments


def select_best_candidate(candidate_segments):
    if not candidate_segments:
        return None

    return max(
        candidate_segments,
        key=lambda item: (
            item["rows"],
            item["completeness"],
            -item["interpolated_rows"],
            -item["start_time"].value,
        ),
    )


def prepare_selected_output(best_candidate):
    selected_df = best_candidate["df"].copy().reset_index(drop=True)
    selected_df["segment_id"] = 0
    selected_df["source_segment_id"] = int(best_candidate["candidate_id"])
    selected_df["segment_completeness"] = float(best_candidate["completeness"])
    selected_df["selection_rule"] = SELECTION_RULE
    return selected_df


def process_single_file(file_path):
    print(f"\nProcessing file: {file_path}")
    file_name = os.path.basename(file_path)

    df = prepare_raw_dataframe(file_path)
    full_time_range = pd.date_range(df["time"].min(), df["time"].max(), freq=SAMPLING_INTERVAL, tz=df["time"].dt.tz)
    merged_df = pd.DataFrame({"time": full_time_range}).merge(df, on="time", how="left", sort=True)

    merged_df["is_missing"] = merged_df[MODEL_FEATURE_COLS].isna().any(axis=1)
    total_points = len(merged_df)
    missing_points = int(merged_df["is_missing"].sum())
    completeness = 1.0 - missing_points / max(total_points, 1)
    print(
        f"  grid points={total_points}, missing={missing_points}, "
        f"raw completeness={completeness * 100:.2f}%"
    )

    candidate_segments = build_candidate_segments(merged_df)
    print(f"  valid candidate segments: {len(candidate_segments)}")
    best_candidate = select_best_candidate(candidate_segments)
    if best_candidate is None:
        print(f"  no valid segment retained for {file_name}")
        return merged_df, [], None, None

    selected_df = prepare_selected_output(best_candidate)
    output_path = os.path.join(OUTPUT_DIR, f"processed_{file_name}")
    selected_df.to_csv(output_path, index=False)

    retained_ratio = len(selected_df) / max(total_points, 1)
    print(
        f"  selected candidate={best_candidate['candidate_id']} rows={best_candidate['rows']} "
        f"completeness={best_candidate['completeness']:.4f} "
        f"interpolated_rows={best_candidate['interpolated_rows']}"
    )
    print(f"  saved {len(selected_df)} rows ({retained_ratio * 100:.2f}% retained)")
    print(f"  output={output_path}")

    summary_row = {
        "file_name": file_name,
        "selected_candidate_id": int(best_candidate["candidate_id"]),
        "selected_rows": int(best_candidate["rows"]),
        "selected_start_time": str(best_candidate["start_time"]),
        "selected_end_time": str(best_candidate["end_time"]),
        "selected_completeness": float(best_candidate["completeness"]),
        "selected_interpolated_rows": int(best_candidate["interpolated_rows"]),
        "candidate_count": len(candidate_segments),
        "selection_rule": SELECTION_RULE,
    }
    return merged_df, candidate_segments, selected_df, summary_row


def plot_missing_data_pattern(merged_dfs, file_names, candidate_groups, selected_frames):
    fig, axes = plt.subplots(len(merged_dfs), 1, figsize=(12, 3 * len(merged_dfs)), sharex=True)
    if len(merged_dfs) == 1:
        axes = [axes]

    for ax, merged_df, file_name, candidates, selected_df in zip(axes, merged_dfs, file_names, candidate_groups, selected_frames):
        missing_indices = merged_df.index[merged_df["is_missing"]]
        ax.vlines(missing_indices, ymin=0, ymax=1, color="black", linewidth=0.5, alpha=0.7)

        time_to_index = pd.Series(merged_df.index.to_numpy(), index=merged_df["time"])

        for candidate in candidates:
            start_idx = int(time_to_index.loc[candidate["start_time"]])
            end_idx = int(time_to_index.loc[candidate["end_time"]])
            ax.axvspan(start_idx, end_idx, facecolor="red", alpha=0.15, edgecolor="red", linewidth=1.0)

        if selected_df is not None and not selected_df.empty:
            selected_df = selected_df.copy()
            selected_df["time"] = pd.to_datetime(selected_df["time"], utc=True)
            start_idx = int(time_to_index.loc[selected_df["time"].iloc[0]])
            end_idx = int(time_to_index.loc[selected_df["time"].iloc[-1]])
            source_segment_id = int(selected_df["source_segment_id"].iloc[0])
            rect = plt.Rectangle(
                (start_idx, 0.02),
                max(end_idx - start_idx, 1),
                0.96,
                fill=False,
                edgecolor="limegreen",
                linewidth=2.2,
            )
            ax.add_patch(rect)
            ax.text(
                start_idx,
                1.02,
                f"chosen seg {source_segment_id}",
                color="green",
                fontsize=9,
                ha="left",
                va="bottom",
            )

        ax.set_ylabel(file_name.replace(".csv", ""), fontsize=11)
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.3)

    axes[-1].set_xlabel("Time index")
    plt.suptitle(
        "Missing Data Pattern and Candidate Segments\nRed: candidates, Green box: selected segment",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "missing_data_pattern.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved missing-data plot to {output_path}")


def save_selection_summary(selection_rows):
    if not selection_rows:
        return

    summary_df = pd.DataFrame(selection_rows)
    summary_df = summary_df.sort_values("file_name").reset_index(drop=True)
    output_path = os.path.join(OUTPUT_DIR, "selected_segments_summary.csv")
    summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved selection summary to {output_path}")


def main():
    merged_dfs = []
    candidate_groups = []
    selected_frames = []
    selection_rows = []
    valid_file_names = []

    for file_path in INPUT_FILES:
        if not os.path.exists(file_path):
            print(f"Missing input file: {file_path}")
            continue

        try:
            merged_df, candidate_segments, selected_df, summary_row = process_single_file(file_path)
        except Exception as exc:
            print(f"Failed to process {file_path}: {exc}")
            continue

        merged_dfs.append(merged_df)
        candidate_groups.append(candidate_segments)
        selected_frames.append(selected_df)
        valid_file_names.append(file_path)
        if summary_row is not None:
            selection_rows.append(summary_row)

    if merged_dfs:
        plot_missing_data_pattern(merged_dfs, valid_file_names, candidate_groups, selected_frames)
    save_selection_summary(selection_rows)

    total_original = sum(len(df) for df in merged_dfs)
    total_retained = sum(len(df) for df in selected_frames if df is not None)
    print("\n" + "=" * 60)
    print("Preprocessing summary")
    print("=" * 60)
    print(f"total grid points: {total_original}")
    print(f"total retained rows: {total_retained}")
    if total_original > 0:
        print(f"overall retained ratio: {total_retained / total_original * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
