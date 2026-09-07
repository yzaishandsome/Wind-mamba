# Pre-specified Frequency and Gust Regimes

Definitions frozen before target-test evaluation on 2026-09-04.

## High-Frequency Energy Ratio

For each 36-step physical input window, compute the real FFT of UWND and VWND. With N=36, the real FFT bins are 0 through 18.

- Exclude the DC bin 0.
- Low non-zero band: bins 1 through 9.
- High band: bins 10 through 18.
- Energy is squared complex magnitude, summed over UWND and VWND.
- High-frequency energy ratio is high-band energy divided by total non-zero-frequency energy.

Quartile cut points are fitted only on source-vessel training windows. Target-test windows are assigned to Q1-Q4 using those frozen cut points. No target-test distribution is used to define the bands or cut points.

## Gust-Rich Regime

The GUST_WND_MEAN threshold is the exact 95th percentile of raw GUST_WND_MEAN values in source-vessel training slices only. A target-test forecast sample is classified as gust-rich when the maximum GUST_WND_MEAN in its 36-step input window exceeds this frozen threshold. The sample label is applied to all six forecast horizons.

## Diagnostic Metrics

For Full, Magnitude-only, and No-FFT variants, report WS-RMSE and circular WD-MAE by frequency quartile and gust regime. Intermediate feature high-frequency ratios may be extracted with forward hooks, but are interpreted only as representation-response diagnostics, not as a physical decomposition proof.
