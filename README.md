# Wind-Mamba

This repository contains a compact implementation of Wind-Mamba, a persistence-guided dual-domain forecasting model for short-term marine wind prediction on unmanned surface vehicles (USVs).

## Files

- `model.py`: Wind-Mamba model definition, loss functions, and optional Mamba backend support.
- `main.py`: training and evaluation entry point for the transfer and target-only protocols.
- `data_provider.py`: chronological split, training-only normalization, and sliding-window dataset construction.
- `experiment_config.py`: vessel file configuration and source/target vessel split.

## Expected Data Layout

This repository does not include raw or processed Saildrone data. Raw Saildrone mission files are publicly available from the NOAA PMEL Saildrone archive and ERDDAP/NetCDF access service:

<https://data.pmel.noaa.gov/pmel/erddap/info/index.html?page=1&itemsPerPage=1000>

After downloading and preprocessing the mission files, place the processed CSV files under `processed_data/` by default:

```text
processed_data/
  processed_sd1031.csv
  processed_sd1033.csv
  ...
  processed_sd1091.csv
```

If your processed files are stored elsewhere, set the data root before running:

```bash
export WIND_MAMBA_DATA_ROOT=/path/to/your/processed_data
```

On Windows PowerShell:

```powershell
$env:WIND_MAMBA_DATA_ROOT="D:\path\to\your\processed_data"
```

Alternatively, edit `DATA_ROOT`, `VESSEL_IDS`, and `TARGET_VESSEL_IDS` in `experiment_config.py` according to your local file layout.

Each processed CSV should include the columns used in `data_provider.py`, including location, vessel-motion variables, wind-vector components, gust wind speed, air temperature, barometric pressure, true wind speed, and true wind direction.

The repository intentionally excludes raw data, processed data, trained weights, and paper figures.

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
