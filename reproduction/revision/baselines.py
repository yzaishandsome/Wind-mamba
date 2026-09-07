import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _project_wind_outputs(pred):
    ws_pred = F.relu(pred[..., 0:1])

    raw_dir = pred[..., 1:3]
    dir_norm = torch.linalg.vector_norm(raw_dir, dim=-1, keepdim=True)
    fallback_dir = torch.zeros_like(raw_dir)
    fallback_dir[..., 1] = 1.0
    wd_vec = torch.where(dir_norm > 1e-6, raw_dir / dir_norm.clamp_min(1e-6), fallback_dir)
    return ws_pred, wd_vec[..., 0:1], wd_vec[..., 1:2]


class SequenceAttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class MultiHorizonDecoder(nn.Module):
    def __init__(self, hidden_dim, pred_len, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.horizon_embedding = nn.Embedding(pred_len, hidden_dim)
        self.decoder_input_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, context):
        batch_size = context.size(0)
        horizon_ids = torch.arange(self.pred_len, device=context.device)
        horizon_tokens = self.horizon_embedding(horizon_ids).unsqueeze(0).expand(batch_size, -1, -1)
        decoder_input = self.decoder_input_proj(
            torch.cat([context.unsqueeze(1).expand(-1, self.pred_len, -1), horizon_tokens], dim=-1)
        )
        decoder_state, _ = self.decoder(decoder_input, context.unsqueeze(0).contiguous())
        return self.head(decoder_state)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len, hidden_dim):
        super().__init__()
        self.position = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        nn.init.normal_(self.position, mean=0.0, std=0.02)

    def forward(self, x):
        return x + self.position[:, : x.size(1), :]


class Baseline_BP(nn.Module):
    def __init__(self, in_dim=10, seq_len=96, pred_len=6, hidden_dim=128):
        super().__init__()
        self.pred_len = pred_len
        self.flatten_dim = in_dim * seq_len
        self.net = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, pred_len * 3),
        )

    def forward(self, x, boat_id=None):
        del boat_id
        batch_size = x.size(0)
        x_flat = x.reshape(batch_size, -1)
        pred = self.net(x_flat).reshape(batch_size, self.pred_len, 3)
        return _project_wind_outputs(pred)


class Baseline_CNNLSTM(nn.Module):
    def __init__(self, in_dim=10, seq_len=96, pred_len=6, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.pred_len = pred_len

        self.cnn = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.temporal_proj = nn.Linear(hidden_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.pool = SequenceAttentionPool(hidden_dim)
        self.decoder = MultiHorizonDecoder(hidden_dim, pred_len, dropout=dropout)

    def forward(self, x, boat_id=None):
        del boat_id
        h = self.cnn(x.unsqueeze(1))
        h = h.mean(dim=-1).transpose(1, 2)
        h = self.temporal_proj(h)
        h, _ = self.lstm(h)
        context = self.pool(h)
        pred = self.decoder(context)
        return _project_wind_outputs(pred)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(self.net(x) + residual)


class Baseline_TCNLSTM(nn.Module):
    def __init__(self, in_dim=10, seq_len=96, pred_len=6, hidden_dim=64):
        super().__init__()
        del seq_len
        self.pred_len = pred_len

        layers = []
        num_channels = [hidden_dim] * 3
        for idx in range(3):
            dilation = 2 ** idx
            in_channels = in_dim if idx == 0 else num_channels[idx - 1]
            layers.append(
                TemporalBlock(
                    in_channels,
                    num_channels[idx],
                    kernel_size=3,
                    stride=1,
                    dilation=dilation,
                    padding=(3 - 1) * dilation,
                    dropout=0.2,
                )
            )
        self.tcn = nn.Sequential(*layers)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.pool = SequenceAttentionPool(hidden_dim)
        self.decoder = MultiHorizonDecoder(hidden_dim, pred_len, dropout=0.2)

    def forward(self, x, boat_id=None):
        del boat_id
        tcn_out = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        lstm_out, _ = self.lstm(tcn_out)
        context = self.pool(lstm_out)
        pred = self.decoder(context)
        return _project_wind_outputs(pred)


class Baseline_LSTM(nn.Module):
    """Plain LSTM baseline with the same multi-horizon output convention."""

    def __init__(self, in_dim=10, seq_len=96, pred_len=6, hidden_dim=96, num_layers=2, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.pos_encoding = LearnablePositionalEncoding(seq_len, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.pool = SequenceAttentionPool(hidden_dim)
        self.decoder = MultiHorizonDecoder(hidden_dim, pred_len, dropout=dropout)

    def forward(self, x, boat_id=None):
        del boat_id
        h = self.pos_encoding(self.proj(x))
        h, _ = self.lstm(h)
        context = self.pool(h)
        pred = self.decoder(context)
        return _project_wind_outputs(pred)


class Baseline_GRU(nn.Module):
    """Plain GRU baseline used to isolate recurrent-state modelling effects."""

    def __init__(self, in_dim=10, seq_len=96, pred_len=6, hidden_dim=96, num_layers=2, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.pos_encoding = LearnablePositionalEncoding(seq_len, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.pool = SequenceAttentionPool(hidden_dim)
        self.decoder = MultiHorizonDecoder(hidden_dim, pred_len, dropout=dropout)

    def forward(self, x, boat_id=None):
        del boat_id
        h = self.pos_encoding(self.proj(x))
        h, _ = self.gru(h)
        context = self.pool(h)
        pred = self.decoder(context)
        return _project_wind_outputs(pred)


class Baseline_Mamba(nn.Module):
    def __init__(self, in_dim=10, seq_len=96, pred_len=6, hidden_dim=96, n_layers=3, mamba_backend=None):
        super().__init__()
        self.pred_len = pred_len
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.pos_encoding = LearnablePositionalEncoding(seq_len, hidden_dim)

        from model import MambaBlock

        self.mamba = nn.Sequential(
            *[MambaBlock(d_model=hidden_dim, d_state=16, backend=mamba_backend) for _ in range(n_layers)]
        )
        self.pool = SequenceAttentionPool(hidden_dim)
        self.decoder = MultiHorizonDecoder(hidden_dim, pred_len, dropout=0.1)

    def forward(self, x, boat_id=None):
        del boat_id
        h = self.pos_encoding(self.proj(x))
        h = self.mamba(h)
        context = self.pool(h)
        pred = self.decoder(context)
        return _project_wind_outputs(pred)


class Baseline_Transformer(nn.Module):
    def __init__(self, in_dim=10, seq_len=96, pred_len=6, hidden_dim=64, num_layers=3):
        super().__init__()
        self.pred_len = pred_len
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.pos_encoding = LearnablePositionalEncoding(seq_len, hidden_dim)
        self.input_dropout = nn.Dropout(0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = SequenceAttentionPool(hidden_dim)
        self.decoder = MultiHorizonDecoder(hidden_dim, pred_len, dropout=0.1)

    def forward(self, x, boat_id=None):
        del boat_id
        h = self.input_dropout(self.pos_encoding(self.proj(x)))
        h = self.encoder(h)
        context = self.pool(h)
        pred = self.decoder(context)
        return _project_wind_outputs(pred)


class InformerDistillationLayer(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_in, kernel_size=3, padding=1, padding_mode="circular")
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.max_pool(self.activation(self.norm(self.conv(x))))
        return x.transpose(1, 2)


class Baseline_Informer(nn.Module):
    def __init__(self, in_dim=10, seq_len=96, pred_len=6, hidden_dim=64, num_layers=3):
        super().__init__()
        self.pred_len = pred_len
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.pos_encoding = LearnablePositionalEncoding(seq_len, hidden_dim)
        self.input_dropout = nn.Dropout(0.1)

        encoder_layers = []
        for _ in range(num_layers):
            encoder_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=4,
                    dim_feedforward=hidden_dim * 4,
                    dropout=0.1,
                    batch_first=True,
                    activation="gelu",
                )
            )
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.distill_layers = nn.ModuleList(
            [InformerDistillationLayer(hidden_dim) for _ in range(max(0, num_layers - 1))]
        )
        self.pool = SequenceAttentionPool(hidden_dim)
        self.decoder = MultiHorizonDecoder(hidden_dim, pred_len, dropout=0.1)

    def forward(self, x, boat_id=None):
        del boat_id
        h = self.input_dropout(self.pos_encoding(self.proj(x)))
        for idx, encoder in enumerate(self.encoder_layers):
            h = encoder(h)
            if idx < len(self.distill_layers):
                h = self.distill_layers[idx](h)
        context = self.pool(h)
        pred = self.decoder(context)
        return _project_wind_outputs(pred)


class MovingAverageDecomposition(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=self.padding, count_include_pad=False)

    def forward(self, x):
        trend = self.avg_pool(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


class Baseline_Autoformer(nn.Module):
    """Decomposition-based Transformer baseline inspired by Autoformer."""

    def __init__(self, in_dim=10, seq_len=96, pred_len=6, hidden_dim=96, num_layers=3):
        super().__init__()
        self.pred_len = pred_len
        self.decomp = MovingAverageDecomposition(kernel_size=5)
        self.seasonal_proj = nn.Linear(in_dim, hidden_dim)
        self.trend_proj = nn.Linear(in_dim, hidden_dim)
        self.pos_encoding = LearnablePositionalEncoding(seq_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.pool = SequenceAttentionPool(hidden_dim)
        self.decoder = MultiHorizonDecoder(hidden_dim, pred_len, dropout=0.1)

    def forward(self, x, boat_id=None):
        del boat_id
        seasonal, trend = self.decomp(x)
        h_seasonal = self.pos_encoding(self.seasonal_proj(seasonal))
        h_seasonal = self.encoder(h_seasonal)
        h_trend = self.trend_proj(trend)
        h = self.fusion(torch.cat([h_seasonal, h_trend], dim=-1))
        context = self.pool(h)
        pred = self.decoder(context)
        return _project_wind_outputs(pred)


class TimesBlock(nn.Module):
    def __init__(self, hidden_dim, periods=(3, 6, 12), dropout=0.1):
        super().__init__()
        self.periods = periods
        self.period_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), padding=(1, 1), groups=hidden_dim),
                    nn.GELU(),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                    nn.Dropout(dropout),
                )
                for _ in periods
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def _period_conv(self, x, period, conv):
        batch_size, seq_len, hidden_dim = x.shape
        pad_len = (period - seq_len % period) % period
        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))
        padded_len = x.size(1)
        x_2d = x.reshape(batch_size, padded_len // period, period, hidden_dim).permute(0, 3, 1, 2)
        y = conv(x_2d).permute(0, 2, 3, 1).reshape(batch_size, padded_len, hidden_dim)
        return y[:, :seq_len]

    def forward(self, x):
        residual = x
        period_outputs = [self._period_conv(x, period, conv) for period, conv in zip(self.periods, self.period_convs)]
        h = torch.stack(period_outputs, dim=0).mean(dim=0)
        h = self.norm(residual + h)
        return self.norm(h + self.ffn(h))


class Baseline_TimesNet(nn.Module):
    """Multi-period convolutional baseline inspired by TimesNet."""

    def __init__(self, in_dim=10, seq_len=96, pred_len=6, hidden_dim=96, num_layers=3):
        super().__init__()
        self.pred_len = pred_len
        del seq_len
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.Sequential(*[TimesBlock(hidden_dim) for _ in range(num_layers)])
        self.pool = SequenceAttentionPool(hidden_dim)
        self.decoder = MultiHorizonDecoder(hidden_dim, pred_len, dropout=0.1)

    def forward(self, x, boat_id=None):
        del boat_id
        h = self.proj(x)
        h = self.blocks(h)
        context = self.pool(h)
        pred = self.decoder(context)
        return _project_wind_outputs(pred)
