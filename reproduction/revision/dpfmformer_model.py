"""Paper-based DPFMformer adaptation for the isolated major-revision study.

This module implements the architecture frozen in
closest_prior_art/dpfmformer_reimplementation_spec.md. It intentionally does
not import or reuse Wind-Mamba's persistence, vessel embedding, residual,
spectral-convolution, fusion-gate, or decoder modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import SelectiveSSM


class PaperMLP(nn.Module):
    """Two dense layers, GELU, and dropout as depicted in the source paper."""

    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class TokenEmbedding(nn.Module):
    """Shared one-dimensional convolutional token embedding."""

    def __init__(self, input_dim: int, model_dim: int) -> None:
        super().__init__()
        self.projection = nn.Conv1d(
            input_dim,
            model_dim,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.projection(values.transpose(1, 2)).transpose(1, 2)


class DualPathFrequencyBlock(nn.Module):
    """Real MLP-attention-Mamba path and imaginary residual MLP path."""

    def __init__(
        self,
        model_dim: int,
        state_dim: int,
        convolution_kernel: int,
        attention_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.real_norm = nn.LayerNorm(model_dim)
        self.real_mlp = PaperMLP(model_dim, dropout)
        self.real_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.real_mamba = SelectiveSSM(
            d_model=model_dim,
            d_state=state_dim,
            d_conv=convolution_kernel,
            expand=2,
        )
        self.real_dropout = nn.Dropout(dropout)

        self.imag_norm = nn.LayerNorm(model_dim)
        self.imag_mlp = PaperMLP(model_dim, dropout)

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        real_input = spectrum.real
        real_update = self.real_mlp(self.real_norm(real_input))
        real_update, _ = self.real_attention(
            real_update,
            real_update,
            real_update,
            need_weights=False,
        )
        real_update = self.real_mamba(real_update)
        real_output = real_input + self.real_dropout(real_update)

        imag_input = spectrum.imag
        imag_output = imag_input + self.imag_mlp(self.imag_norm(imag_input))
        return torch.complex(real_output, imag_output)


class TemporalBottomUp(nn.Module):
    """Map a higher-resolution temporal representation to the next scale."""

    def __init__(self, input_length: int, output_length: int, dropout: float) -> None:
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Linear(input_length, output_length),
            nn.GELU(),
            nn.Linear(output_length, output_length),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.mapping(values.transpose(1, 2)).transpose(1, 2)


class DPFMformerMarine(nn.Module):
    """Minimal joint WS/WD adaptation of the paper's base DPFMformer."""

    def __init__(
        self,
        input_dim: int = 10,
        sequence_length: int = 36,
        prediction_length: int = 6,
        model_dim: int = 16,
        state_dim: int = 16,
        convolution_kernel: int = 4,
        attention_heads: int = 4,
        dropout: float = 0.1,
        moving_average_kernel: int = 25,
        scale_lengths: tuple[int, ...] = (36, 18, 9),
    ) -> None:
        super().__init__()
        if scale_lengths[0] != sequence_length:
            raise ValueError("The first scale must equal the input sequence length.")
        if any(left // 2 != right for left, right in zip(scale_lengths, scale_lengths[1:])):
            raise ValueError("DPFMformer scales must follow power-of-two average pooling.")
        if model_dim % attention_heads != 0:
            raise ValueError("model_dim must be divisible by attention_heads.")

        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.model_dim = model_dim
        self.moving_average_kernel = moving_average_kernel
        self.scale_lengths = tuple(scale_lengths)

        self.token_embedding = TokenEmbedding(input_dim, model_dim)
        self.dual_path = DualPathFrequencyBlock(
            model_dim=model_dim,
            state_dim=state_dim,
            convolution_kernel=convolution_kernel,
            attention_heads=attention_heads,
            dropout=dropout,
        )
        self.bottom_up = nn.ModuleList(
            [
                TemporalBottomUp(input_length, output_length, dropout)
                for input_length, output_length in zip(scale_lengths, scale_lengths[1:])
            ]
        )
        final_width = scale_lengths[-1] * model_dim
        self.output_norm = nn.LayerNorm(model_dim)
        self.output_head = nn.Sequential(
            nn.Linear(final_width, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, prediction_length * 3),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _causal_moving_average(self, values: torch.Tensor) -> torch.Tensor:
        channel_first = values.transpose(1, 2)
        padded = F.pad(
            channel_first,
            (self.moving_average_kernel - 1, 0),
            mode="replicate",
        )
        averaged = F.avg_pool1d(
            padded,
            kernel_size=self.moving_average_kernel,
            stride=1,
        )
        return averaged.transpose(1, 2)

    def _multiscale_inputs(self, values: torch.Tensor) -> list[torch.Tensor]:
        scales = [self._causal_moving_average(values)]
        for _ in self.scale_lengths[1:]:
            pooled = F.avg_pool1d(
                scales[-1].transpose(1, 2),
                kernel_size=2,
                stride=2,
            ).transpose(1, 2)
            scales.append(pooled)
        return scales

    def _process_scale(self, values: torch.Tensor) -> torch.Tensor:
        embedded = self.token_embedding(values)
        spectrum = torch.fft.rfft(embedded, dim=1, norm="ortho")
        processed = self.dual_path(spectrum)
        return torch.fft.irfft(processed, n=values.size(1), dim=1, norm="ortho")

    @staticmethod
    def _decode_joint_output(raw_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wind_speed = F.relu(raw_output[..., 0:1])
        raw_direction = raw_output[..., 1:3]
        norm = torch.linalg.vector_norm(raw_direction, dim=-1, keepdim=True)
        fallback = torch.zeros_like(raw_direction)
        fallback[..., 1] = 1.0
        direction = torch.where(norm > 1e-6, raw_direction / norm.clamp_min(1e-6), fallback)
        return wind_speed, direction[..., 0:1], direction[..., 1:2]

    def forward(self, values: torch.Tensor, boat_id: torch.Tensor | None = None):
        del boat_id
        time_representations = [self._process_scale(scale) for scale in self._multiscale_inputs(values)]
        reconstructed = time_representations[0]
        for mapper, current_scale in zip(self.bottom_up, time_representations[1:]):
            reconstructed = current_scale + mapper(reconstructed)
        normalized = self.output_norm(reconstructed)
        raw_output = self.output_head(normalized.flatten(start_dim=1))
        raw_output = raw_output.reshape(values.size(0), self.prediction_length, 3)
        return self._decode_joint_output(raw_output)


class DPFMformerKFLDirectionLoss(nn.Module):
    """Paper FK/MSE objective for WS plus benchmark direction-cosine loss."""

    def __init__(self, alpha: float = 0.9, direction_weight: float = 1.0) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        self.alpha = float(alpha)
        self.direction_weight = float(direction_weight)

    def forward(
        self,
        wind_speed_pred: torch.Tensor,
        direction_sin_pred: torch.Tensor,
        direction_cos_pred: torch.Tensor,
        wind_speed_true: torch.Tensor,
        direction_sin_true: torch.Tensor,
        direction_cos_true: torch.Tensor,
    ):
        predicted_spectrum = torch.fft.rfft(wind_speed_pred.squeeze(-1), dim=1, norm="ortho")
        true_spectrum = torch.fft.rfft(wind_speed_true.squeeze(-1), dim=1, norm="ortho")
        difference = true_spectrum - predicted_spectrum
        squared_complex_distance = difference.real.square() + difference.imag.square()
        frequency_kernel_loss = (1.0 - torch.exp(-squared_complex_distance)).mean()
        time_mse = F.mse_loss(wind_speed_pred, wind_speed_true)
        wind_speed_loss = self.alpha * frequency_kernel_loss + (1.0 - self.alpha) * time_mse

        predicted_direction = F.normalize(
            torch.cat([direction_sin_pred, direction_cos_pred], dim=-1),
            dim=-1,
        )
        true_direction = F.normalize(
            torch.cat([direction_sin_true, direction_cos_true], dim=-1),
            dim=-1,
        )
        direction_loss = (1.0 - torch.sum(predicted_direction * true_direction, dim=-1)).mean()
        total = wind_speed_loss + self.direction_weight * direction_loss
        return total, wind_speed_loss, direction_loss
