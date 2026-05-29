# Wind-Mamba

This repository contains a compact implementation of Wind-Mamba, a persistence-guided dual-domain forecasting model for short-term marine wind prediction on unmanned surface vehicles (USVs).

## Files

- `model.py`: Wind-Mamba model definition, loss functions, and optional Mamba backend support.
- `main.py`: training and evaluation entry point for the transfer and target-only protocols.
- `data_provider.py`: chronological split, training-only normalization, and sliding-window dataset construction.
- `experiment_config.py`: vessel file configuration and source/target vessel split.

## Expected Data Layout

The training script expects processed Saildrone vessel files under:

```text
processed_data/
  processed_sd1031.csv
  processed_sd1033.csv
  ...
  processed_sd1091.csv
```

Each processed CSV should include the columns used in `data_provider.py`, including location, vessel-motion variables, wind-vector components, gust wind speed, air temperature, barometric pressure, true wind speed, and true wind direction.

Raw Saildrone mission data are publicly available from the NOAA PMEL Saildrone mission archive and ERDDAP/NetCDF access service. This repository does not include raw data, processed data, trained weights, or paper figures.

## Installation

Install the common dependencies:

```bash
pip install -r requirements.txt
```

The official `mamba-ssm` package is optional. If it is unavailable, the code can use the custom PyTorch fallback implementation defined in `model.py`.

## Usage

```bash
python main.py
```

Key settings can be controlled by environment variables:

```bash
EDGEWIND_SEQ_LEN=36
EDGEWIND_PRED_LEN=6
EDGEWIND_SEED=42
EDGEWIND_LOSS_MODE=smoothl1_dircos
EDGEWIND_WD_WEIGHT=1.0
```

Outputs are written to `weights/` and `figures/`, both of which are ignored by Git.
