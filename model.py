import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as OfficialMamba

    HAS_OFFICIAL_MAMBA = True
except ImportError:
    OfficialMamba = None
    HAS_OFFICIAL_MAMBA = False


DEFAULT_UPPER_TAIL_WS_THRESHOLD = 10.59


def resolve_mamba_backend(backend=None):
    backend = (backend or os.getenv("WIND_MAMBA_BACKEND", "custom")).lower()
    if backend == "auto":
        return "official" if HAS_OFFICIAL_MAMBA else "custom"
    if backend == "official":
        if not HAS_OFFICIAL_MAMBA:
            raise ImportError(
                "WIND_MAMBA_BACKEND=official was requested, but mamba_ssm is not installed. "
                "Install mamba-ssm on Linux or switch back to the custom backend."
            )
        return "official"
    if backend == "custom":
        return "custom"
    if backend == "reference":
        return "reference"
    raise ValueError(f"Unsupported Mamba backend: {backend}")


class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        nn.init.constant_(self.dt_proj.bias, -4.0)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).reshape(d_state, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        _, seq_len, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = self.conv1d(x.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x = F.silu(x)

        x_param = self.x_proj(x)
        b_proj, c_proj = x_param.chunk(2, dim=-1)
        dt = torch.clamp(F.softplus(self.dt_proj(x)), min=1e-5, max=0.1)
        A = -torch.exp(self.A_log.float())
        y = self._stable_parallel_scan(x, dt, A, b_proj, c_proj)
        return self.out_proj(y * F.silu(z))

    def _stable_parallel_scan(self, u, dt, A, B, C):
        batch_size, seq_len, d_inner = u.shape
        d_state = A.shape[0]

        dt = dt.unsqueeze(-1)
        A_bar = torch.exp(torch.clamp(dt * A.T, min=-80.0, max=0.0))
        B_u = (dt * B.unsqueeze(2)) * u.unsqueeze(-1)

        h = torch.zeros(batch_size, d_inner, d_state, device=u.device, dtype=u.dtype)
        outputs = []
        for i in range(seq_len):
            h = A_bar[:, i] * h + B_u[:, i]
            outputs.append(torch.einsum("bds,bs->bd", h, C[:, i, :]))
        return torch.stack(outputs, dim=1) + u * self.D


class ReferenceSelectiveSSM(nn.Module):
    """Reference PyTorch selective state-space block for reproducible experiments."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto"):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        nn.init.constant_(self.dt_proj.bias, -4.0)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        _, seq_len, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = self.conv1d(x.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x = F.silu(x)

        x_param = self.x_proj(x)
        dt_param, b_proj, c_proj = torch.split(
            x_param,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        dt = torch.clamp(F.softplus(self.dt_proj(dt_param)), min=1e-5, max=0.1)
        A = -torch.exp(self.A_log.float())
        y = self._scan(x, dt, A, b_proj, c_proj)
        return self.out_proj(y * F.silu(z))

    def _scan(self, u, dt, A, B, C):
        batch_size, seq_len, d_inner = u.shape
        h = torch.zeros(batch_size, d_inner, self.d_state, device=u.device, dtype=u.dtype)
        outputs = []
        for i in range(seq_len):
            a_bar = torch.exp(torch.clamp(dt[:, i].unsqueeze(-1) * A, min=-80.0, max=0.0))
            b_u = dt[:, i].unsqueeze(-1) * B[:, i].unsqueeze(1) * u[:, i].unsqueeze(-1)
            h = a_bar * h + b_u
            outputs.append(torch.einsum("bds,bs->bd", h, C[:, i, :]))
        return torch.stack(outputs, dim=1) + u * self.D


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2, dropout=0.2, backend=None):
        super().__init__()
        self.backend = resolve_mamba_backend(backend)
        self.norm = nn.LayerNorm(d_model)
        if self.backend == "official":
            self.ssm = OfficialMamba(d_model=d_model, d_state=d_state, d_conv=4, expand=expand)
        elif self.backend == "reference":
            self.ssm = ReferenceSelectiveSSM(d_model, d_state=d_state, d_conv=4, expand=expand)
        else:
            self.ssm = SelectiveSSM(d_model, d_state=d_state, d_conv=4, expand=expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.ssm(self.norm(x)))


class FrequencyTurbulenceExtractor(nn.Module):
    def __init__(self, in_dim, hidden_dim, seq_len):
        super().__init__()
        self.freq_proj = nn.Linear(in_dim, hidden_dim)
        self.real_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.imag_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.real_norm = nn.LayerNorm(hidden_dim)
        self.imag_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        _, seq_len, _ = x.shape
        x_proj = self.freq_proj(x)
        x_fft = torch.fft.rfft(x_proj, dim=1, norm="ortho")

        x_real = self.real_conv(x_fft.real.transpose(1, 2)).transpose(1, 2)
        x_imag = self.imag_conv(x_fft.imag.transpose(1, 2)).transpose(1, 2)
        x_real = self.real_norm(x_real)
        x_imag = self.imag_norm(x_imag)

        h_time = torch.fft.irfft(torch.complex(x_real, x_imag), dim=1, norm="ortho", n=seq_len)
        return self.out_proj(h_time) + x_proj


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        scores = self.score(x)
        weights = torch.softmax(scores.squeeze(-1), dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class WindMambaModel(nn.Module):
    def __init__(
        self,
        in_dim=10,
        seq_len=36,
        pred_len=6,
        hidden_dim=96,
        d_state=16,
        n_mamba_layers=3,
        num_boats=4,
        feature_stats=None,
        mamba_backend=None,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.u_idx = 5
        self.v_idx = 6

        stats = feature_stats or {}
        self.register_buffer("u_mean", torch.tensor(float(stats.get("u_mean", 0.0))))
        self.register_buffer("u_scale", torch.tensor(float(stats.get("u_scale", 1.0))))
        self.register_buffer("v_mean", torch.tensor(float(stats.get("v_mean", 0.0))))
        self.register_buffer("v_scale", torch.tensor(float(stats.get("v_scale", 1.0))))
        self.mamba_backend = resolve_mamba_backend(mamba_backend)

        self.boat_embedding = nn.Embedding(num_boats, hidden_dim)
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.mamba_backbone = nn.Sequential(
            *[
                MambaBlock(hidden_dim, d_state=d_state, backend=self.mamba_backend)
                for _ in range(n_mamba_layers)
            ]
        )
        self.turbulence_extractor = FrequencyTurbulenceExtractor(in_dim, hidden_dim, seq_len)

        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.pool = AttentionPooling(hidden_dim)

        self.horizon_embedding = nn.Embedding(pred_len, hidden_dim)
        self.decoder_input_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.horizon_decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        self.u_delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.v_delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.max_uv_delta = 4.0
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _last_wind_vector(self, x):
        last_u = x[:, -1, self.u_idx] * self.u_scale + self.u_mean
        last_v = x[:, -1, self.v_idx] * self.v_scale + self.v_mean
        return last_u, last_v

    def _vector_to_ws_angle(self, u, v):
        ws = torch.sqrt(u ** 2 + v ** 2 + 1e-6)
        angle = torch.atan2(u, v) + math.pi
        angle = torch.remainder(angle, 2 * math.pi)
        return ws, angle

    def _persistence_baseline(self, x):
        last_u, last_v = self._last_wind_vector(x)
        return self._vector_to_ws_angle(last_u, last_v)

    def forward(self, x, boat_id):
        batch_size, _, _ = x.shape
        base_u, base_v = self._last_wind_vector(x)

        h_input = self.input_proj(x) + self.boat_embedding(boat_id).unsqueeze(1)
        h_m = self.mamba_backbone(h_input)
        h_f = self.turbulence_extractor(x)

        gate = 0.5 + torch.sigmoid(self.gate_network(torch.cat([h_m, h_f], dim=-1)))
        h_combined = torch.cat([h_m, gate * h_f], dim=-1)
        h_fused_seq = h_m + self.fusion_proj(h_combined)
        h_context = self.pool(h_fused_seq)

        horizon_ids = torch.arange(self.pred_len, device=x.device)
        horizon_tokens = self.horizon_embedding(horizon_ids).unsqueeze(0).expand(batch_size, -1, -1)
        decoder_input = self.decoder_input_proj(
            torch.cat([h_context.unsqueeze(1).expand(-1, self.pred_len, -1), horizon_tokens], dim=-1)
        )
        h_state, _ = self.horizon_decoder(decoder_input, h_context.unsqueeze(0).contiguous())

        u_delta = self.max_uv_delta * torch.tanh(self.u_delta_head(h_state))
        v_delta = self.max_uv_delta * torch.tanh(self.v_delta_head(h_state))

        pred_u = base_u.unsqueeze(1).unsqueeze(-1) + u_delta
        pred_v = base_v.unsqueeze(1).unsqueeze(-1) + v_delta
        ws_pred, pred_angle = self._vector_to_ws_angle(pred_u.squeeze(-1), pred_v.squeeze(-1))

        ws_pred = torch.clamp(ws_pred.unsqueeze(-1), min=0.0)
        pred_angle = pred_angle.unsqueeze(-1)
        return ws_pred, torch.sin(pred_angle), torch.cos(pred_angle)


class UpperTailWeightedDirectionalLoss(nn.Module):
    def __init__(
        self,
        upper_tail_ws_threshold=DEFAULT_UPPER_TAIL_WS_THRESHOLD,
        upper_tail_weight=2.0,
        wd_weight=1.0,
    ):
        super().__init__()
        self.upper_tail_ws_threshold = upper_tail_ws_threshold
        self.upper_tail_weight = upper_tail_weight
        self.wd_weight = wd_weight

    def forward(self, ws_pred, wd_sin_pred, wd_cos_pred, ws_true, wd_sin_true, wd_cos_true):
        ws_loss = F.smooth_l1_loss(ws_pred, ws_true, reduction="none")
        ws_weights = 1.0 + (ws_true > self.upper_tail_ws_threshold).float() * (self.upper_tail_weight - 1.0)
        ws_loss = (ws_loss * ws_weights).mean()

        pred_dir = F.normalize(torch.cat([wd_sin_pred, wd_cos_pred], dim=-1), dim=-1)
        true_dir = F.normalize(torch.cat([wd_sin_true, wd_cos_true], dim=-1), dim=-1)
        cos_sim = torch.sum(pred_dir * true_dir, dim=-1)
        wd_loss = (1.0 - cos_sim).mean()
        return ws_loss + self.wd_weight * wd_loss, ws_loss, wd_loss


class Loss_MSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ws_pred, wd_sin_pred, wd_cos_pred, ws_true, wd_sin_true, wd_cos_true):
        ws_loss = F.mse_loss(ws_pred, ws_true)
        wd_loss = F.mse_loss(
            torch.cat([wd_sin_pred, wd_cos_pred], dim=-1),
            torch.cat([wd_sin_true, wd_cos_true], dim=-1),
        )
        return ws_loss + wd_loss, ws_loss, wd_loss


class Loss_MAE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ws_pred, wd_sin_pred, wd_cos_pred, ws_true, wd_sin_true, wd_cos_true):
        ws_loss = F.l1_loss(ws_pred, ws_true)
        wd_loss = F.l1_loss(
            torch.cat([wd_sin_pred, wd_cos_pred], dim=-1),
            torch.cat([wd_sin_true, wd_cos_true], dim=-1),
        )
        return ws_loss + wd_loss, ws_loss, wd_loss


class Loss_SmoothL1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ws_pred, wd_sin_pred, wd_cos_pred, ws_true, wd_sin_true, wd_cos_true):
        ws_loss = F.smooth_l1_loss(ws_pred, ws_true)
        wd_loss = F.l1_loss(
            torch.cat([wd_sin_pred, wd_cos_pred], dim=-1),
            torch.cat([wd_sin_true, wd_cos_true], dim=-1),
        )
        return ws_loss + wd_loss, ws_loss, wd_loss


class Loss_SmoothL1_DirCos(nn.Module):
    def __init__(self, wd_weight=1.0):
        super().__init__()
        self.wd_weight = wd_weight

    def forward(self, ws_pred, wd_sin_pred, wd_cos_pred, ws_true, wd_sin_true, wd_cos_true):
        ws_loss = F.smooth_l1_loss(ws_pred, ws_true)
        pred_dir = F.normalize(torch.cat([wd_sin_pred, wd_cos_pred], dim=-1), dim=-1)
        true_dir = F.normalize(torch.cat([wd_sin_true, wd_cos_true], dim=-1), dim=-1)
        wd_loss = (1.0 - torch.sum(pred_dir * true_dir, dim=-1)).mean()
        return ws_loss + self.wd_weight * wd_loss, ws_loss, wd_loss


class LossSmoothL1UpperTail(nn.Module):
    def __init__(
        self,
        upper_tail_ws_threshold=DEFAULT_UPPER_TAIL_WS_THRESHOLD,
        upper_tail_weight=2.0,
    ):
        super().__init__()
        self.upper_tail_ws_threshold = upper_tail_ws_threshold
        self.upper_tail_weight = upper_tail_weight

    def forward(self, ws_pred, wd_sin_pred, wd_cos_pred, ws_true, wd_sin_true, wd_cos_true):
        ws_loss = F.smooth_l1_loss(ws_pred, ws_true, reduction="none")
        ws_weights = 1.0 + (ws_true > self.upper_tail_ws_threshold).float() * (self.upper_tail_weight - 1.0)
        ws_loss = (ws_loss * ws_weights).mean()
        wd_loss = F.l1_loss(
            torch.cat([wd_sin_pred, wd_cos_pred], dim=-1),
            torch.cat([wd_sin_true, wd_cos_true], dim=-1),
        )
        return ws_loss + wd_loss, ws_loss, wd_loss


def build_loss(
    loss_mode="smoothl1_dircos",
    upper_tail_ws_threshold=DEFAULT_UPPER_TAIL_WS_THRESHOLD,
    upper_tail_weight=2.0,
    wd_weight=1.0,
):
    loss_mode = (loss_mode or "smoothl1_dircos").strip().lower()
    if loss_mode in {"mse", "l2", "l2_dirl2"}:
        return Loss_MSE()
    if loss_mode in {"mae", "l1", "l1_dirl1"}:
        return Loss_MAE()
    if loss_mode in {"smoothl1", "smoothl1_dirl1"}:
        return Loss_SmoothL1()
    if loss_mode in {"smoothl1_upper_tail", "smoothl1_upper_tail_dirl1"}:
        return LossSmoothL1UpperTail(
            upper_tail_ws_threshold=upper_tail_ws_threshold,
            upper_tail_weight=upper_tail_weight,
        )
    if loss_mode in {"smoothl1_dircos", "dircos"}:
        return Loss_SmoothL1_DirCos(wd_weight=wd_weight)
    if loss_mode in {"smoothl1_upper_tail_dircos", "upper_tail"}:
        return UpperTailWeightedDirectionalLoss(
            upper_tail_ws_threshold=upper_tail_ws_threshold,
            upper_tail_weight=upper_tail_weight,
            wd_weight=wd_weight,
        )
    raise ValueError(f"Unsupported loss mode: {loss_mode}")
