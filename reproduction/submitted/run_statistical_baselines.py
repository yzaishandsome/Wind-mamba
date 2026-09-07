import math
from pathlib import Path

import numpy as np
import pandas as pd

from data_provider import USVDataset
from experiment_config import EXPERIMENT_VERSION, TARGET_BOAT_FILES, boat_tag


SEQ_LEN = 36
PRED_LEN = 6
OUTPUT_DIR = Path("comparison_outputs") / EXPERIMENT_VERSION / "statistical_baselines"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = OUTPUT_DIR / "statistical_baseline_summary.csv"
MEAN_PATH = OUTPUT_DIR / "statistical_baseline_mean.csv"


def circular_mae_deg(pred_sin, pred_cos, true_sin, true_cos):
    pred_deg = (np.degrees(np.arctan2(pred_sin, pred_cos)) + 360.0) % 360.0
    true_deg = (np.degrees(np.arctan2(true_sin, true_cos)) + 360.0) % 360.0
    diff = np.abs(pred_deg - true_deg)
    return np.minimum(diff, 360.0 - diff)


def fit_ridge_forecaster(features, targets, ridge=1e-4):
    if len(features) == 0:
        return None
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    gram = x_aug.T @ x_aug
    reg = ridge * np.eye(gram.shape[0])
    reg[-1, -1] = 0.0
    return np.linalg.solve(gram + reg, x_aug.T @ y)


def predict_with_weights(weights, features, fallback):
    if weights is None:
        return fallback
    x = np.asarray(features, dtype=np.float64)
    x_aug = np.concatenate([x, [1.0]])
    return float(x_aug @ weights)


def arima_forecast_1d(history, horizon, order=3):
    history = np.asarray(history, dtype=np.float64)
    if len(history) < order + 2:
        return np.repeat(history[-1], horizon)

    diff = np.diff(history)
    features, targets = [], []
    for idx in range(order, len(diff)):
        features.append(diff[idx - order : idx][::-1])
        targets.append(diff[idx])

    weights = fit_ridge_forecaster(features, targets)
    diff_buffer = list(diff)
    current = float(history[-1])
    forecasts = []
    for _ in range(horizon):
        lag_values = diff_buffer[-order:][::-1]
        if len(lag_values) < order:
            lag_values = [0.0] * (order - len(lag_values)) + lag_values
        next_diff = predict_with_weights(weights, lag_values, 0.0)
        current += next_diff
        diff_buffer.append(next_diff)
        forecasts.append(current)
    return np.asarray(forecasts)


def sarima_forecast_1d(history, horizon, order=3, seasonal_lag=6):
    history = np.asarray(history, dtype=np.float64)
    if len(history) < seasonal_lag + order + 2:
        return arima_forecast_1d(history, horizon, order=order)

    diff = np.diff(history)
    features, targets = [], []
    start_idx = max(order, seasonal_lag)
    for idx in range(start_idx, len(diff)):
        ar_lags = diff[idx - order : idx][::-1]
        seasonal = [diff[idx - seasonal_lag]]
        features.append(np.concatenate([ar_lags, seasonal]))
        targets.append(diff[idx])

    weights = fit_ridge_forecaster(features, targets)
    diff_buffer = list(diff)
    current = float(history[-1])
    forecasts = []
    for _ in range(horizon):
        ar_lags = diff_buffer[-order:][::-1]
        if len(ar_lags) < order:
            ar_lags = [0.0] * (order - len(ar_lags)) + ar_lags
        seasonal_value = diff_buffer[-seasonal_lag] if len(diff_buffer) >= seasonal_lag else 0.0
        next_diff = predict_with_weights(weights, np.concatenate([ar_lags, [seasonal_value]]), 0.0)
        current += next_diff
        diff_buffer.append(next_diff)
        forecasts.append(current)
    return np.asarray(forecasts)


def forecast_window(history_ws, history_sin, history_cos, method):
    if method == "ARIMA":
        ws_pred = arima_forecast_1d(history_ws, PRED_LEN)
        sin_pred = arima_forecast_1d(history_sin, PRED_LEN)
        cos_pred = arima_forecast_1d(history_cos, PRED_LEN)
    elif method == "SARIMA":
        ws_pred = sarima_forecast_1d(history_ws, PRED_LEN)
        sin_pred = sarima_forecast_1d(history_sin, PRED_LEN)
        cos_pred = sarima_forecast_1d(history_cos, PRED_LEN)
    else:
        raise ValueError(f"Unsupported method: {method}")

    ws_pred = np.maximum(ws_pred, 0.0)
    norm = np.sqrt(sin_pred**2 + cos_pred**2)
    valid = norm > 1e-8
    sin_pred[valid] = sin_pred[valid] / norm[valid]
    cos_pred[valid] = cos_pred[valid] / norm[valid]
    sin_pred[~valid] = 0.0
    cos_pred[~valid] = 1.0
    return ws_pred, sin_pred, cos_pred


def evaluate_predictions(ws_pred, sin_pred, cos_pred, ws_true, sin_true, cos_true):
    ws_error = ws_pred - ws_true
    ws_rmse = math.sqrt(float(np.mean(ws_error**2)))
    ws_mae = float(np.mean(np.abs(ws_error)))
    wd_mae = float(np.mean(circular_mae_deg(sin_pred, cos_pred, sin_true, cos_true)))
    sse = float(np.sum(ws_error**2))
    sst = float(np.sum((ws_true - np.mean(ws_true)) ** 2))
    ws_r2 = 1.0 - sse / max(sst, 1e-12)
    return {
        "test_ws_rmse": ws_rmse,
        "test_ws_mae": ws_mae,
        "test_wd_mae": wd_mae,
        "test_ws_r2": ws_r2,
    }


def target_test_frame(target_file):
    dataset = USVDataset([target_file], flag="train", seq_len=SEQ_LEN, pred_len=PRED_LEN)
    segments = dataset._load_continuous_segments(target_file)
    if not segments:
        raise ValueError(f"No valid segment found for {target_file}")
    seg_df = segments[0]
    _, val_end = dataset._split_bounds(len(seg_df))
    return seg_df.iloc[val_end:].reset_index(drop=True)


def evaluate_target(target_file, method):
    df = target_test_frame(target_file)
    ws = df["WS_TRUE"].to_numpy(dtype=np.float64)
    wd_rad = np.deg2rad(df["WD_TRUE"].to_numpy(dtype=np.float64))
    wd_sin = np.sin(wd_rad)
    wd_cos = np.cos(wd_rad)

    ws_preds, sin_preds, cos_preds = [], [], []
    ws_trues, sin_trues, cos_trues = [], [], []
    sample_count = max(len(df) - SEQ_LEN - PRED_LEN + 1, 0)
    for idx in range(sample_count):
        history_slice = slice(idx, idx + SEQ_LEN)
        future_slice = slice(idx + SEQ_LEN, idx + SEQ_LEN + PRED_LEN)
        ws_pred, sin_pred, cos_pred = forecast_window(
            ws[history_slice],
            wd_sin[history_slice],
            wd_cos[history_slice],
            method,
        )
        ws_preds.append(ws_pred)
        sin_preds.append(sin_pred)
        cos_preds.append(cos_pred)
        ws_trues.append(ws[future_slice])
        sin_trues.append(wd_sin[future_slice])
        cos_trues.append(wd_cos[future_slice])

    return evaluate_predictions(
        np.asarray(ws_preds),
        np.asarray(sin_preds),
        np.asarray(cos_preds),
        np.asarray(ws_trues),
        np.asarray(sin_trues),
        np.asarray(cos_trues),
    )


def main():
    rows = []
    for method in ["ARIMA", "SARIMA"]:
        for target_file in TARGET_BOAT_FILES:
            tag = boat_tag(target_file)
            metrics = evaluate_target(target_file, method)
            row = {"model": method, "target": tag, **metrics}
            rows.append(row)
            print(
                f"[{method} | {tag.upper()}] "
                f"WS-RMSE={metrics['test_ws_rmse']:.3f} | "
                f"WS-MAE={metrics['test_ws_mae']:.3f} | "
                f"WD-MAE={metrics['test_wd_mae']:.2f} | "
                f"R2={metrics['test_ws_r2']:.3f}"
            )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    mean_df = summary_df.groupby("model", as_index=False)[
        ["test_ws_rmse", "test_ws_mae", "test_wd_mae", "test_ws_r2"]
    ].mean()
    mean_df.to_csv(MEAN_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved summary to {SUMMARY_PATH}")
    print(f"Saved mean table to {MEAN_PATH}")


if __name__ == "__main__":
    main()
