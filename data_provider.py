import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


# This module reads user-preprocessed Saildrone CSV files. It does not download
# raw NOAA PMEL ERDDAP/NetCDF data or perform mission-level raw-data conversion.
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
EXPECTED_INTERVAL_MINUTES = 10


class USVDataset(Dataset):
    def __init__(self, file_paths, flag="train", seq_len=96, pred_len=6, global_boat_ids=None, scaler=None):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.flag = flag
        self.file_paths = file_paths
        self.global_boat_ids = global_boat_ids if global_boat_ids is not None else list(range(len(file_paths)))

        self.feature_cols = [
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
        self.target_cols = ["WS_TRUE", "WD_TRUE"]

        self.scaler = scaler if scaler is not None else StandardScaler()
        self.x_data, self.y_data, self.boat_ids = self._read_data()

        self.segment_sample_counts = [max(len(x) - self.seq_len - self.pred_len + 1, 0) for x in self.x_data]
        self.total_len = sum(self.segment_sample_counts)

    def _segment_score(self, seg_df):
        completeness = float(seg_df["segment_completeness"].iloc[0]) if "segment_completeness" in seg_df.columns else 1.0
        interpolated_rows = int(seg_df["is_interpolated"].sum()) if "is_interpolated" in seg_df.columns else 0
        start_value = pd.to_datetime(seg_df["time"].iloc[0], utc=True).value
        return (len(seg_df), completeness, -interpolated_rows, -start_value)

    def _select_best_segment(self, segments):
        if not segments:
            return []
        best_segment = max(segments, key=self._segment_score)
        return [best_segment]

    def _load_continuous_segments(self, file_path):
        df = pd.read_csv(file_path)
        required_cols = ["time"] + self.feature_cols + self.target_cols
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"{file_path} is missing required columns: {missing_cols}")

        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

        if "segment_id" not in df.columns:
            df["segment_id"] = 0

        segments = []
        for _, seg_df in df.groupby("segment_id", sort=True):
            seg_df = seg_df.sort_values("time").drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
            seg_df = seg_df.replace([np.inf, -np.inf], np.nan)
            seg_df = seg_df.dropna(subset=self.feature_cols + self.target_cols).reset_index(drop=True)
            if len(seg_df) < self.seq_len + self.pred_len:
                continue

            time_diff = seg_df["time"].diff().dt.total_seconds().div(60)
            continuity_break = (time_diff.notna()) & (time_diff != EXPECTED_INTERVAL_MINUTES)
            subsegment_ids = continuity_break.cumsum()

            for _, subseg_df in seg_df.groupby(subsegment_ids, sort=True):
                subseg_df = subseg_df.reset_index(drop=True)
                if len(subseg_df) >= self.seq_len + self.pred_len:
                    segments.append(subseg_df)

        return self._select_best_segment(segments)

    def _split_bounds(self, length):
        train_end = int(length * TRAIN_RATIO)
        val_end = train_end + int(length * VAL_RATIO)
        return train_end, val_end

    def _fit_scaler_if_needed(self):
        if self.flag != "train" or hasattr(self.scaler, "mean_"):
            return

        train_frames = []
        for file_path in self.file_paths:
            for seg_df in self._load_continuous_segments(file_path):
                train_end, _ = self._split_bounds(len(seg_df))
                train_slice = seg_df.iloc[:train_end]
                if len(train_slice) >= self.seq_len + self.pred_len:
                    train_frames.append(train_slice[self.feature_cols])

        if not train_frames:
            raise ValueError("No valid training slices found for scaler fitting.")

        train_combined = pd.concat(train_frames, axis=0)
        self.scaler.fit(train_combined.values)

    def _read_data(self):
        self._fit_scaler_if_needed()

        all_x, all_y, all_boat_ids = [], [], []
        for local_idx, file_path in enumerate(self.file_paths):
            segments = self._load_continuous_segments(file_path)

            for seg_df in segments:
                train_end, val_end = self._split_bounds(len(seg_df))
                if self.flag == "train":
                    split_df = seg_df.iloc[:train_end].copy()
                elif self.flag == "val":
                    split_df = seg_df.iloc[train_end:val_end].copy()
                else:
                    split_df = seg_df.iloc[val_end:].copy()

                if len(split_df) < self.seq_len + self.pred_len:
                    continue

                x_val = self.scaler.transform(split_df[self.feature_cols].values)

                y_raw = split_df[self.target_cols].values
                ws = y_raw[:, 0:1]
                wd_deg = y_raw[:, 1:2]
                wd_rad = np.deg2rad(wd_deg)
                wd_sin = np.sin(wd_rad)
                wd_cos = np.cos(wd_rad)
                y_val = np.concatenate([ws, wd_sin, wd_cos], axis=1)

                all_x.append(x_val)
                all_y.append(y_val)
                all_boat_ids.append(self.global_boat_ids[local_idx])

        return all_x, all_y, all_boat_ids

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        for seg_idx, sample_count in enumerate(self.segment_sample_counts):
            if index < sample_count:
                s_begin = index
                s_end = s_begin + self.seq_len
                r_begin = s_end
                r_end = r_begin + self.pred_len

                seq_x = self.x_data[seg_idx][s_begin:s_end]
                seq_y = self.y_data[seg_idx][r_begin:r_end]
                boat_id = self.boat_ids[seg_idx]
                return (
                    torch.tensor(seq_x, dtype=torch.float32),
                    torch.tensor(seq_y, dtype=torch.float32),
                    torch.tensor(boat_id, dtype=torch.long),
                )
            index -= sample_count

        raise IndexError("Index out of bounds")


if __name__ == "__main__":
    from experiment_config import ALL_BOAT_FILES

    print("Testing USVDataset...")
    train_dataset = USVDataset(file_paths=ALL_BOAT_FILES, flag="train", seq_len=36, pred_len=6)
    print(f"train samples: {len(train_dataset)}")
    if len(train_dataset) > 0:
        print(f"sample X shape: {train_dataset[0][0].shape}")
        print(f"sample Y shape: {train_dataset[0][1].shape}")
