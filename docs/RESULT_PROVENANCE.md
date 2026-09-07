# Result provenance

The release keeps the formal numeric outputs separate from generated
presentation tables. `scripts/generate_tables.py` reads these files without
changing metric values.

| Manuscript item | Released numeric source |
|---|---|
| Table 4 repeated deep benchmark | `results/submitted/repeated_main_raw.csv`, `repeated_main_wilcoxon.csv` |
| Table 4 deterministic controls | `results/submitted/persistence_baseline_summary.csv`, `statistical_baseline_summary.csv` |
| Table 8 module ablation | `results/submitted/repeated_core_ablation_summary.csv`, `repeated_core_ablation_raw.csv` |
| Table 9 loss analysis | `results/revision/weighted_loss_revision_summary.csv` |
| Table 11 intervals | `results/revision/conformal_clean_final_metrics.csv`, `chronological_split_counts.csv` |
| Table 12 horizons | `results/revision/horizon_metrics_five_seed_summary.csv` |
| Table 13 spectral controls | `results/revision/spectral_phase_five_seed_summary.csv`, `spectral_frequency_regime_summary.csv`, `gust_subset_summary.csv` |
| Table 14 residual bounds | `results/revision/residual_bound_five_seed_summary.csv`, `residual_bound_regime_summary.csv` |
| Table 15 closest prior art | `results/revision/closest_prior_art_five_seed_summary.csv` |
| Fig. 9 | `results/revision/fig9_candidate_sd1042.npz`, `fig9_candidate_metadata.json` |

Local checkpoint and prediction paths in detailed metric files are replaced by
`not_released` in this public package. This redaction does not alter seeds,
variants, targets, parameter counts, sample counts, thresholds, or metrics.
