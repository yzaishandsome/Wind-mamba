# Wind-Mamba

![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8-red)
![Task](https://img.shields.io/badge/Task-Marine%20Wind%20Forecasting-0b7285)
![Protocol](https://img.shields.io/badge/Protocol-ASOC%20Major%20Revision-orange)

Wind-Mamba is a persistence-guided dual-domain model for six-step marine wind
speed and direction forecasting on unmanned sailboats. It combines bounded
wind-vector residual learning, a Mamba temporal pathway, a real/imaginary
Fourier pathway, and gated temporal-spectral fusion.

## Project updates

- **2026-09-07:** Added the full Applied Soft Computing revision reproduction package.
- **2026-06-17:** Added chronological wind-speed interval calibration.
- **2026-05-31:** Added the Wind-Mamba architecture schematic.

## Architecture

![Wind-Mamba dual-domain architecture](assets/wind_mamba_architecture.png)

## Repository contents

```text
Wind-mamba/
  model.py                         Readable Wind-Mamba implementation
  main.py                          Public training/evaluation entry point
  data_provider.py                 Chronological splitting and windows
  interval.py                      Clean chronological conformal entry point
  experiment_config.py             Vessel and protocol constants
  configs/asoc_revision.json       Frozen ASOC revision configuration
  preprocessing/                   Raw-to-processed pipeline and manifests
  reproduction/submitted/          Submitted benchmark execution snapshot
  reproduction/revision/           Revision experiment execution snapshot
  results/submitted/               Submitted benchmark summaries
  results/revision/                Formal revision summaries and Fig. 9 data
  scripts/generate_tables.py       Tables 4, 8, 9, and 11-15 source tables
  scripts/generate_fig9.py         Revised clean-conformal Fig. 9
  requirements.txt
  environment.yml
```

The files under `reproduction/submitted/` preserve the names and execution
structure used by the reported benchmark. The top-level modules provide a
cleaner public interface. Both use the same 10-variable input order and the
same chronological data loader. All executable upper-tail analyses use the
source-training-only p95 of 10.580439745930407 m/s; the rounded full-data p95
of 10.59 m/s is a descriptive manuscript statistic, not an experimental
threshold in this repository.

## Data

Raw and processed Saildrone records are not redistributed. Download the public
mission data from the [NOAA PMEL ERDDAP catalog](https://data.pmel.noaa.gov/pmel/erddap/info/index.html?page=1&itemsPerPage=1000).
The required vessel IDs and export assumptions are documented in
[`preprocessing/DATA_ACQUISITION.md`](preprocessing/DATA_ACQUISITION.md).

The study uses ten retained sequences. SD1042 and SD1091 are held-out targets;
the other eight vessels are source vessels. Exact UTC ranges are in
[`preprocessing/retained_segments.csv`](preprocessing/retained_segments.csv),
and record counts plus SHA-256 checksums are in
[`preprocessing/processed_file_manifest.csv`](preprocessing/processed_file_manifest.csv).

The preprocessing path is:

```text
NOAA CSV
  -> timestamps rounded to a 10-minute grid
  -> circular duplicate aggregation for angular fields
  -> split at gaps longer than one 10-minute step
  -> circular interpolation for COG/HDG/wind direction
  -> deterministic retained-segment selection
  -> WS_TRUE/WD_TRUE derived from UWND_MEAN/VWND_MEAN
  -> processed CSV
```

COG and HDG are restored to degrees after circular interpolation and then
z-score transformed using training-slice statistics before entering the model.
The WD prediction target is encoded as sine and cosine.

## Reported configuration

| Item | Value |
|---|---|
| Input variables | 10 |
| Observation window | 36 steps (6 hours) |
| Forecast horizon | 6 steps (1 hour) |
| Window stride | 1 step (10 minutes) |
| Main split | 60% train / 20% validation / 20% test |
| Hidden dimension | 96 |
| Mamba layers / state dimension | 3 / 16 |
| Direction-loss weight | 1.0 |
| Batch size | 32 |
| Optimizer | AdamW |
| Pretraining | 60 epochs, learning rate 2e-4 |
| Finetuning | 20 epochs, learning rate 5e-5 |
| Early-stopping patience | 8 |
| Seeds | 42, 43, 44, 45, 46 |
| Experimental upper-tail threshold | source-training p95 = 10.580439745930407 m/s |

The complete machine-readable configuration is
[`configs/asoc_revision.json`](configs/asoc_revision.json).

## Environment

The reported environment used Python 3.9.25, PyTorch 2.8.0, and CUDA 12.8.
Create it with either:

```bash
conda env create -f environment.yml
conda activate wind-mamba
```

or:

```bash
python -m pip install -r requirements.txt
```

The reported checkpoints use the checkpoint-compatible custom PyTorch selective
state-space implementation. On supported Linux/CUDA systems, `mamba-ssm` may be
installed separately for backend experiments, but its numerical and runtime
behavior can differ from the custom backend. CPU execution is supported but is
substantially slower for training.

## Reproducing the ASOC revision experiments

Run commands from the repository root unless a step explicitly changes
directory. On Windows PowerShell, replace `export NAME=value` with
`$env:NAME="value"`.

### 1. Download and preprocess data

```bash
python preprocessing/preprocess_saildrone.py \
  --raw-dir raw_data \
  --output-dir processed_data
python scripts/build_data_manifest.py --data-root processed_data
```

Compare the generated manifest with the checked-in checksum manifest before
training. If processed data are stored elsewhere:

```bash
export WIND_MAMBA_DATA_ROOT=/absolute/path/to/processed_data
```

### 2. Source pretraining and target finetuning

The submitted runner performs source-vessel pretraining followed by independent
finetuning and evaluation on SD1042 and SD1091. Run the five seeds as follows:

```bash
cd reproduction/submitted
export WIND_MAMBA_DATA_ROOT=/absolute/path/to/processed_data
export EDGEWIND_REPEAT_SEEDS=42,43,44,45,46
export EDGEWIND_REPEAT_STEPS=edgewind
export EDGEWIND_REPEAT_SUFFIX_PREFIX=rev4_seed
export EDGEWIND_MAMBA_BACKEND=custom
python run_repeated_core_experiments.py
cd ../..
```

For each seed, source and target checkpoints are written below
`reproduction/submitted/weights/transfer_two_targets_v1/edgewind/rev4_seed_<seed>/`.
Validation is used for early stopping and best-checkpoint selection; test data
are used only for final evaluation.

### 3. Main benchmark baselines

Deep transfer baselines:

```bash
cd reproduction/submitted
export WIND_MAMBA_DATA_ROOT=/absolute/path/to/processed_data
export EDGEWIND_REPEAT_SEEDS=42,43,44,45,46
export EDGEWIND_REPEAT_STEPS=baselines
export EDGEWIND_REPEAT_SUFFIX_PREFIX=rev4_seed
export EDGEWIND_RUN_NO_TRANSFER=0
python run_repeated_core_experiments.py
python run_statistical_baselines.py
python run_persistence.py
cd ../..
```

`baselines.py` contains BP, CNN-LSTM, Improved TCN-LSTM, Informer,
Autoformer, TimesNet, Standard Transformer, and Standard Mamba. The rolling
ARIMA/SARIMA controls are in `run_statistical_baselines.py`. Adaptation details
are stated in the class docstrings and runner configuration.

### 4. Submitted module, loss, and backbone controls

```bash
cd reproduction/submitted
export WIND_MAMBA_DATA_ROOT=/absolute/path/to/processed_data
export EDGEWIND_REPEAT_SEEDS=42,43,44,45,46
export EDGEWIND_REPEAT_STEPS=ablation
export EDGEWIND_REPEAT_SUFFIX_PREFIX=rev6_fresh_seed
python run_repeated_core_experiments.py

export EDGEWIND_REPEAT_STEPS=loss,replacement
export EDGEWIND_REPEAT_SUFFIX_PREFIX=rev4_seed
python run_repeated_core_experiments.py
cd ../..
```

These commands cover the module ablation, loss ablation, weighted loss,
Mamba/Transformer backbone control, and the no-FFT checkpoint reused by the
revision diagnostics. The separate suffixes match the formal run families:
module variants use `rev6_fresh_seed_*`, while loss and backbone controls use
`rev4_seed_*`.

### 5. Revision spectral and residual-bound experiments

Point `WIND_MAMBA_CHECKPOINT_ROOT` to the directory that contains the `weights/`
folder produced in Steps 2-4:

```bash
export WIND_MAMBA_DATA_ROOT=/absolute/path/to/processed_data
export WIND_MAMBA_CHECKPOINT_ROOT=/absolute/path/to/Wind-mamba/reproduction/submitted
export WIND_MAMBA_REVISION_OUTPUT=/absolute/path/to/revision_outputs
python reproduction/revision/run_task2_spectral_phase.py
python reproduction/revision/run_task3_residual_bounds.py
python reproduction/revision/run_task4_weighted_loss.py
```

This covers Full/Magnitude-only/No-FFT spectral controls, training-defined
frequency-energy and gust-rich diagnostics, five residual-bound variants, and
the source-training-p95 weighted-loss rerun.

### 6. Circular-input sensitivity

```bash
export WIND_MAMBA_DATA_ROOT=/absolute/path/to/processed_data
export WIND_MAMBA_CHECKPOINT_ROOT=/absolute/path/to/Wind-mamba/reproduction/submitted
export WIND_MAMBA_REVISION_OUTPUT=/absolute/path/to/revision_outputs
python reproduction/revision/circular/run_circular_sensitivity.py
python reproduction/revision/run_task5_circular_3seed.py
```

The first command produces the seed-42 circular run; the second adds seeds 43
and 44 and aggregates the three-seed 10D-degree versus 12D-sine/cosine control.

### 7. DPFMformer closest-prior-art comparison

```bash
export WIND_MAMBA_DATA_ROOT=/absolute/path/to/processed_data
export WIND_MAMBA_CHECKPOINT_ROOT=/absolute/path/to/Wind-mamba/reproduction/submitted
export WIND_MAMBA_REVISION_OUTPUT=/absolute/path/to/revision_outputs
export DPFMFORMER_PAPER_PATH=/absolute/path/to/1-s2.0-S0360544225028671-main.pdf
python reproduction/revision/run_task1_dpfmformer.py --mode smoke --seeds 42
python reproduction/revision/run_task1_dpfmformer.py --mode train --seeds 42 43 44 45 46
python reproduction/revision/run_task1_dpfmformer.py --mode aggregate
```

This is a **paper-based reimplementation for the reviewer-requested
comparison; no official implementation was available**. It follows Hong et
al., *Energy* 332 (2025) 137225 and documents frozen paper-level ambiguities in
[`docs/dpfmformer_reimplementation_spec.md`](docs/dpfmformer_reimplementation_spec.md).
The marine adaptation changes only the input projection, multi-horizon output,
and joint WS/sin(WD)/cos(WD) target. Both the original frequency-kernel-loss
control and common benchmark-loss control are included.

### 8. Horizon, persistence, and clean conformal analyses

```bash
export WIND_MAMBA_DATA_ROOT=/absolute/path/to/processed_data
export WIND_MAMBA_CHECKPOINT_ROOT=/absolute/path/to/Wind-mamba/reproduction/submitted
export WIND_MAMBA_HORIZON_OUTPUT=/absolute/path/to/horizon_outputs
export WIND_MAMBA_CONFORMAL_OUTPUT=/absolute/path/to/conformal_outputs
python reproduction/revision/run_horizon_and_bootstrap.py
python interval.py
```

The horizon script reports t+1 through t+6 errors and a paired chronological
moving-block bootstrap rather than an IID t-test. `interval.py` executes the
clean 60/20/10/10 chronological split-conformal workflow: the 20% validation
block is used only for early stopping, the next 10% for calibration, and the
final 10% for interval evaluation. It does not use validation-selected margin
inflation or final-test tuning.

### 9. Regenerate manuscript tables and Fig. 9

The checked-in formal summaries permit table and figure regeneration without
redistributing private checkpoints:

```bash
python scripts/generate_tables.py --output-dir generated_tables
python scripts/generate_fig9.py \
  --output generated_figures/nature_fig9_conformal_interval_revision.pdf
```

The table command produces numeric sources for Table 4, Table 8, Table 9, and
Tables 11-15. For Table 8 it recomputes target means and five-seed sample
standard deviations from the released target-level rows and verifies the
explicit `full` and `mlp_decoder` row mapping. The Fig. 9 command uses the locked first chronological SD1042
upper-tail case and the clean conformal interval data in `results/revision/`.
The exact source-file mapping is listed in
[`docs/RESULT_PROVENANCE.md`](docs/RESULT_PROVENANCE.md).

Run the release-integrity check before starting a reproduction:

```bash
python scripts/verify_release.py
```

## Released and intentionally omitted files

Included:

- exact preprocessing and retained-segment metadata;
- submitted and revision execution scripts;
- formal aggregate CSV outputs used in the revised tables;
- locked derived values needed to regenerate Fig. 9;
- environment and step-by-step commands.

Not included:

- NOAA raw or processed mission records, because they are public upstream and
  large; acquisition and checksum verification are provided instead;
- trained checkpoints and per-window prediction arrays, because of repository
  size; all training commands and expected output locations are documented;
- third-party official source code not authored in this project.

Checkpoint and prediction-path columns in the released metric CSV files are
set to `not_released`; the numerical metrics, sample counts, thresholds, model
variants, targets, and seeds are unchanged.

## Version recovery

The public state immediately before the ASOC major-revision reproducibility
sync is preserved by the Git tag `pre-asoc-major-repro-20260907`. This provides
a stable recovery point independently of later changes on `main`. The first
complete reproduction package remains available as
`asoc-major-revision-repro-v1`; the final audited package, including the locked
Table 8 mapping, is tagged `asoc-major-revision-repro-v2`.

## Citation

Please cite the Wind-Mamba manuscript if this code supports your work. For the
closest-prior-art reproduction, also cite:

> J.-T. Hong, S. Han, J. Yan, and Y.-Q. Liu, Dual-path frequency
> Mamba-Transformer model for wind power forecasting, Energy 332 (2025) 137225.
