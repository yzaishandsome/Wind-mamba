# ASOC revision reproducibility checklist

This checklist maps the public package to the material requested during review.

| Item | Public location |
|---|---|
| Raw-data acquisition | `preprocessing/DATA_ACQUISITION.md` |
| Raw-to-processed pipeline | `preprocessing/preprocess_saildrone.py` |
| Retained temporal segments | `preprocessing/retained_segments.csv` |
| Record counts and checksums | `preprocessing/processed_file_manifest.csv` |
| Source/target vessels and splits | `configs/asoc_revision.json`, `experiment_config.py` |
| Sliding-window implementation | `reproduction/submitted/data_provider.py` |
| Wind-Mamba model | `reproduction/submitted/model.py` |
| Deep and statistical baselines | `reproduction/submitted/baselines.py`, `run_baselines.py`, `run_statistical_baselines.py` |
| Five-seed training and transfer | `reproduction/submitted/run_repeated_core_experiments.py`, `main.py` |
| Module/loss/backbone ablations | `reproduction/submitted/run_ablation.py`, `run_loss_ablation.py`, `run_mamba_transformer_replacement.py` |
| Spectral sensitivity | `reproduction/revision/run_task2_spectral_phase.py` |
| Residual-bound sensitivity | `reproduction/revision/run_task3_residual_bounds.py` |
| Training-p95 weighted loss | `reproduction/revision/run_task4_weighted_loss.py` |
| Circular-input sensitivity | `reproduction/revision/run_task5_circular_3seed.py` |
| Horizon and block-bootstrap analysis | `reproduction/revision/run_horizon_and_bootstrap.py` |
| Clean chronological conformal analysis | `reproduction/revision/run_conformal_clean.py` |
| DPFMformer paper-based comparison | `reproduction/revision/run_task1_dpfmformer.py`, `docs/dpfmformer_reimplementation_spec.md` |
| Formal result summaries | `results/submitted/`, `results/revision/` |
| Table and Fig. 9 generation | `scripts/generate_tables.py`, `scripts/generate_fig9.py` |
| Environment | `requirements.txt`, `environment.yml` |
| End-to-end commands | `README.md` |

Trained checkpoints, per-window prediction arrays, and NOAA raw/processed data
are intentionally omitted for size and redistribution reasons. Commands,
expected paths, public acquisition instructions, checksums, and formal summary
outputs are supplied instead.
