# Pre-specified Magnitude-Only Spectral Variant

Formal definition frozen before the retained training run and before any target-test evaluation on 2026-09-04.

## Purpose

This variant tests whether retaining the real and imaginary Fourier components provides value beyond a branch that receives spectral magnitude only. It is not described as a dedicated phase estimator.

## Mathematical Definition

Let `P` be the linearly projected input sequence and let

`F = rFFT(P)`.

The submitted Full branch processes `Re(F)` and `Im(F)` separately. The magnitude-only branch instead forms

`M = |F|`.

Both convolution/normalization paths receive only `M`. Their outputs are combined with the fixed scale `1/sqrt(2)`. The reconstructed spectrum is

`M_tilde = softplus((A(M) + B(M))/sqrt(2))`,

`F_mag = complex(M_tilde, 0)`.

The time-domain branch output is obtained by `irFFT(F_mag)` and passed through the same output projection and residual connection as the submitted frequency extractor.

The softplus operation guarantees a non-negative reconstructed magnitude and the imaginary component is fixed to zero, so this is a strict zero-phase reconstruction. The original complex phase, `angle(F)`, is neither passed to the branch nor reintroduced during reconstruction.

## Controlled Factors

- Same 10-dimensional input and degree/z-score COG/HDG representation.
- Same L=36, H=6, transfer protocol, loss, optimizer, learning rates, batch size, epoch limits, patience, decoder, residual formulation, and seeds 42-46.
- Same two convolution and normalization paths as the Full branch, so the trainable parameter count remains matched to the submitted frequency extractor.
- The only intended change is replacing separate real/imaginary inputs and complex reconstruction with magnitude-only inputs and zero-phase reconstruction.

## Pre-run Definition Amendment

An initial four-epoch source-pretraining pilot used a zero-imaginary reconstruction without enforcing a non-negative real spectrum. Because negative real coefficients correspond to phase pi, that pilot did not satisfy the strict zero-phase wording and was stopped before target finetuning or target-test evaluation. Its incomplete files are excluded from every reported result. The formal retained variant above adds the fixed softplus operation solely to make the pre-specified mathematical definition exact; this decision was not based on target-test performance.

## Interpretation Boundary

The Full model may be described as retaining complex spectral information through separate real/imaginary processing. Neither the Full nor magnitude-only branch will be described as an explicit physical phase model without additional evidence.
