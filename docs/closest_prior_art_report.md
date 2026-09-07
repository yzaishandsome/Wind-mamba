# Closest Prior-Art DPFMformer Comparison Report

## Source and audit status

The complete Energy paper is the sole primary method source. Its identity and SHA256 are recorded in `SOURCE_AVAILABLE.md`. No official author implementation was identified, so this is a declared paper-based reimplementation. The architecture and every paper-level ambiguity were frozen in `dpfmformer_reimplementation_spec.md` before the seed-42 smoke test. The smoke test passed without constructing a target test loader.

## What was compared

1. `paper_kfl_dircos`: base DPFMformer with the paper's FK/time-domain loss at alpha 0.9 plus the benchmark direction-cosine loss.
2. `common_smoothl1_dircos`: the identical DPFMformer architecture with the common benchmark SmoothL1 WS plus direction-cosine loss.

Both use the same 10D inputs, L=36, H=6, 60/20/20 chronological splits, source vessels, SD1042/SD1091 targets, transfer protocol, early stopping rule, and seeds 42-46 as Wind-Mamba. DPFMformer has 13,095 trainable parameters; submitted Wind-Mamba has 512,595.

## Five-seed results

| Model/loss | Target | WS-RMSE | WS-MAE | WD-MAE | R2 | Params |
|---|---|---:|---:|---:|---:|---:|
| paper_kfl_dircos | sd1042 | 1.2888 +/- 0.0905 | 0.9693 +/- 0.0803 | 14.70 +/- 1.40 | 0.8376 +/- 0.0225 | 13,095 |
| paper_kfl_dircos | sd1091 | 0.7935 +/- 0.0227 | 0.5615 +/- 0.0255 | 9.24 +/- 0.39 | 0.8312 +/- 0.0097 | 13,095 |
| paper_kfl_dircos | target_mean | 1.0411 +/- 0.0535 | 0.7654 +/- 0.0501 | 11.97 +/- 0.85 | 0.8344 +/- 0.0149 | 13,095 |
| common_smoothl1_dircos | sd1042 | 1.2109 +/- 0.0386 | 0.9090 +/- 0.0350 | 15.10 +/- 1.29 | 0.8570 +/- 0.0090 | 13,095 |
| common_smoothl1_dircos | sd1091 | 0.7798 +/- 0.0148 | 0.5524 +/- 0.0140 | 9.62 +/- 0.39 | 0.8370 +/- 0.0062 | 13,095 |
| common_smoothl1_dircos | target_mean | 0.9954 +/- 0.0224 | 0.7307 +/- 0.0200 | 12.36 +/- 0.79 | 0.8470 +/- 0.0061 | 13,095 |

## Difference from Wind-Mamba

The deltas below are `DPFMformer - Wind-Mamba`; positive error deltas and negative R2 deltas favor Wind-Mamba.

| DPFMformer loss | Target | Delta WS-RMSE | Delta WS-MAE | Delta WD-MAE | Delta R2 |
|---|---|---:|---:|---:|---:|
| paper_kfl_dircos | sd1042 | +0.2284 | +0.2411 | +4.52 | -0.0529 |
| paper_kfl_dircos | sd1091 | +0.0695 | +0.0710 | +2.18 | -0.0284 |
| paper_kfl_dircos | target_mean | +0.1490 | +0.1561 | +3.35 | -0.0406 |
| common_smoothl1_dircos | sd1042 | +0.1505 | +0.1808 | +4.92 | -0.0334 |
| common_smoothl1_dircos | sd1091 | +0.0558 | +0.0620 | +2.57 | -0.0225 |
| common_smoothl1_dircos | target_mean | +0.1032 | +0.1214 | +3.75 | -0.0280 |

For target-mean WS-RMSE, paper-loss DPFMformer beats Wind-Mamba in 0/5 matched seeds and the common-loss control does so in 0/5. Full target- and seed-specific sign counts are in `dpfmformer_seed_target_win_counts.csv`; no claim of statistical significance is made from these descriptive counts.

## Methodological distinction

DPFMformer performs multiscale smoothing/downsampling, applies FFT at each scale, processes real coefficients with MLP-attention-Mamba and imaginary coefficients with an MLP, reconstructs by inverse FFT, and mixes scales bottom-up. Wind-Mamba instead predicts bounded residual horizontal wind vectors around persistence, uses a deeper 96-dimensional temporal Mamba branch, applies a separate projected real/imaginary spectral-convolution branch, gates temporal-spectral fusion, and decodes all horizons from the fused context. DPFMformer contains no persistence prior, vessel embedding, bounded residual, Wind-Mamba spectral convolution, or Wind-Mamba fusion gate in this comparison.

## Reviewer 5-1 coverage

Reviewer 5-1 is covered experimentally by a five-seed primary paper-loss comparison and a same-architecture common-loss control. The response must state that no official implementation was available and identify the declared paper ambiguities; it should not claim bitwise reproduction of the authors' code. DPFMformer-MEC was not included because MEC is a separate validation-error correction system rather than the requested dual-path base architecture.

## Manuscript-stage decision

The experiment package is ready for human interpretation. It does not modify the submitted manuscript. A revised main-comparison table or a dedicated closest-prior-art table should report both loss controls, parameter counts, target-specific values, and five-seed mean +/- sample standard deviation.
