# DPFMformer Paper-Based Reimplementation Specification

Status: frozen before the seed-42 smoke test and formal training.

## 1. Source and scope

The implementation follows the complete Energy paper by Hong et al. (2025), especially Fig. 2, Eqs. (1)-(6) and (10), Table 2, and the descriptions in Sections 2.2.1-2.2.5. No architecture is inferred from the title or an abstract. No official author implementation was found.

The paper studies **univariate wind-power forecasting**. The present comparison minimally adapts the base DPFMformer architecture to the submitted Wind-Mamba benchmark for joint marine wind-speed and wind-direction forecasting.

## 2. DPFMformer versus DPFMformer-MEC

- **DPFMformer** is the base forecasting architecture: multiscale processing, embedding, Fourier real/imaginary dual paths, bottom-up multiscale reconstruction, and the frequency-kernel loss.
- **DPFMformer-MEC** adds a second-stage maximal-information-coefficient-weighted error-correction pipeline. It forms a validation-error sequence, pretrains an LSTM error model, weights meteorological variables using MIC, uses a residual correction network, and adds the predicted error to the base forecast.

Reviewer 5-1 asks for the closest dual-path frequency-domain Mamba architecture. Therefore the primary baseline is **base DPFMformer**, without MEC. MEC is not trained because it is a post-hoc validation-error correction system rather than the dual-path forecasting architecture, and it would introduce a separate validation-use protocol that is not part of the submitted common benchmark.

## 3. Original multiscale processing

The paper states that the input series is first smoothed with a 25-step moving average and then decomposed into multiple resolutions. Section 2.2.1 formally defines scale `m` by average pooling to length `P / 2^m`. It also reports a window size of 6, but does not give a concrete value for the maximum scale index `M` or an explicit list of scale lengths.

Frozen task adaptation:

- Apply a leakage-safe causal 25-step moving average within each 36-step input window. Left-edge replication supplies the unavailable prefix; no future value is used.
- Construct three power-of-two scales with lengths 36, 18, and 9 using average pooling with kernel and stride 2.
- The three-scale choice is fixed before training. It preserves the paper's `P / 2^m` construction while retaining at least the reported six-step granularity for the 36-step benchmark input.
- No scale count is selected from validation or test performance.

## 4. Embedding

The paper uses token and temporal embeddings and omits positional embedding. It identifies token embedding as a one-dimensional convolution but does not report the temporal-feature encoding details.

Frozen task adaptation:

- A shared circular-padded Conv1d token projection maps the same 10 benchmark inputs to `d_model=16` at each scale.
- No calendar/time covariates are added because the common benchmark input is fixed to the same 10 variables and the existing dataset loader does not expose separate temporal marks.
- No positional embedding is introduced.

## 5. Fourier decomposition and dual paths

For every embedded scale, an orthonormal real FFT is applied along the temporal dimension. Orthonormal scaling is a reproducible numerical convention; the paper does not specify FFT normalization.

### Real component path

The paper's prose reports layer normalization, MLP processing, multi-head self-attention, and then Mamba. Equation (3) gives the order

`Fr_out = Mamba(MH-Attn(MLP(Fr_in))) + Fr_in`.

Frozen realization:

1. LayerNorm.
2. Two-linear-layer GELU MLP with dropout 0.1.
3. Four-head self-attention over frequency bins with dropout 0.1.
4. One custom selective-state-space Mamba layer with `d_state=16`, convolution kernel 4, and expansion 2.
5. Outer residual addition of the original real coefficients.

The number of attention heads and the number of repeated dual-path blocks are not reported. Four heads divide the reported 16-dimensional representation exactly; one block per scale follows the single-block equations. Neither is tuned on target validation or test results.

### Imaginary component path

Equation (4) specifies

`Fi_out = MLP(Fi_in) + Fi_in`.

The implementation uses LayerNorm followed by the same two-linear-layer GELU MLP and dropout 0.1, with an outer residual addition. It does not add attention or Mamba to this path.

### Reconstruction

- Recombine the processed real and imaginary tensors into a complex spectrum.
- Apply inverse real FFT at the corresponding scale length.
- Reconstruct bottom-up from the highest temporal resolution to lower resolutions.
- Each bottom-up mapper uses two linear layers with GELU and maps the temporal dimension from 36 to 18 and then 18 to 9 before residual fusion, matching the paper's Eq. (5) and its MLP description.

The paper does not specify the final scalar forecasting head. The minimal joint-task adaptation uses the final reconstructed multiscale representation, LayerNorm, and a compact two-linear-layer head that emits `H x 3` values.

## 6. Joint marine forecasting output adaptation

Only the unavoidable task interface is changed:

- Input: the same 10-dimensional standardized Saildrone window used by every benchmark model.
- Output horizon: `H=6`.
- Output channels: wind speed, `sin(WD)`, and `cos(WD)`.
- Wind speed is passed through ReLU, consistent with the project's other direct deep baselines.
- The two directional outputs are normalized to a unit vector before metric computation.

The implementation does **not** contain Wind-Mamba's persistence prior, bounded residual prediction, vessel embedding, frequency convolution, spectral gate, temporal-spectral fusion gate, attention pooling, or GRU residual decoder.

## 7. Frequency-kernel loss

The paper defines a bounded frequency discrepancy of the form

`L_FK = mean(1 - exp(-|FFT(y) - FFT(y_hat)|^2))`

and combines it with time-domain MSE:

`L_CFK = alpha * L_FK + (1 - alpha) * L_MSE`.

The paper reports a grid-search optimum of `alpha=0.9` for DPFMformer. The printed equation omits an explicit complex modulus even though a real nonnegative loss requires it. The implementation therefore uses squared complex magnitude, `real(diff)^2 + imag(diff)^2`, and averages over batch and forecast-frequency bins. The FFT is taken over each six-step wind-speed forecast using orthonormal normalization.

For joint WS/WD adaptation:

`L_total = L_CFK(WS) + lambda_d * L_DirCos`, with `lambda_d=1.0`.

This preserves the paper's KFL/FK mechanism for its scalar quantity while applying the benchmark's normalized directional cosine objective to the added wind-direction target.

## 8. Controlled common-loss version

A second version uses the identical DPFMformer architecture with the submitted common benchmark loss:

`SmoothL1(WS) + lambda_d * L_DirCos`, with `lambda_d=1.0`.

This separates architecture behavior from the specialized original FK loss. Architecture and training protocol are otherwise identical between the two versions.

## 9. Paper-reported original hyperparameters

Verified against Table 2 and Section 2.2.5 of the complete PDF:

- Mamba SSM state dimension: 16.
- Mamba convolution kernel size: 4.
- Mamba model dimension: 16.
- Dropout: 0.1.
- Learning rate: 0.0001.
- Training epochs: 10.
- Optimizer: Adam.
- Loss: KFL/FK loss.
- FK/time-domain coefficient for DPFMformer: `alpha=0.9`.

For the fair Saildrone comparison, architecture-specific settings above are retained, while optimization schedule, chronological transfer protocol, batch size, early stopping, and source/target data follow the submitted benchmark rather than the original wind-farm schedule.

## 10. Fair benchmark protocol

- Inputs: same 10 variables and same training-fitted StandardScaler.
- Input/output: `L=36`, `H=6`.
- Chronological split: 60% train, 20% validation, 20% test independently within each retained vessel segment.
- Source vessels: the same eight non-target vessels.
- Target vessels: SD1042 and SD1091.
- Transfer: source pretraining, then separate target-specific finetuning.
- Seeds: 42, 43, 44, 45, and 46.
- Batch size: 32.
- Pretraining: up to 60 epochs, AdamW, learning rate `2e-4`.
- Finetuning: up to 20 epochs, AdamW, learning rate `5e-5`.
- Weight decay: `1e-4`; cosine schedule; gradient clipping at 1.0; patience 8.
- Checkpoint selection: validation score only, `0.7 * WS-RMSE + 0.3 * WD-MAE / 100`.
- Test data are evaluated only after the best checkpoint is fixed.

## 11. Smoke-test rule

Seed 42 first receives a shape/finite-gradient smoke test using source train and validation batches only. No target test performance is examined and no architectural choice is changed after the smoke test. Formal seeds 42-46 then use the frozen specification above.

## 12. Known paper-level ambiguities

The complete article does not report the exact maximum scale index, repeated real/imag block counts shown schematically as `N x` and `M x`, attention-head count, temporal-embedding implementation, FFT normalization, moving-average edge handling, or final forecasting-head design. These items are implemented as declared above and are limitations of a paper-based reimplementation, not claims of bitwise reproduction.
