"""SoftMaskGAT encoders, graph attention and readout (Chen et al.)."""

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .temporal import build_temporal_encoder

N_ANCHOR_FEATURES = 16


class TemporalEncoder(nn.Module):
    """Encode one channel with the GAP branch."""

    def __init__(self, in_channels: int = 2, hidden_dim: int = 64,
                 n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        layers = []
        ch = in_channels
        for i in range(n_layers):
            out_ch = hidden_dim if i == n_layers - 1 \
                else max(16, hidden_dim // (2 ** (n_layers - 1 - i)))
            layers.extend([
                nn.Conv1d(ch, out_ch, kernel_size=7, padding=3),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(4),
                nn.Dropout(dropout),
            ])
            ch = out_ch
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = self.pool(x).squeeze(-1)
        return x


class QualityAwareGATLayer(nn.Module):
    """Neighbour attention with binary masking or finite log-quality penalties."""

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4,
                 dropout: float = 0.1,
                 quality_combine: str = "binary",
                 quality_floor: float = 1e-3,
                 quality_temperature: float = 1.0):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.quality_combine = quality_combine
        self.quality_floor = quality_floor
        self.quality_temperature = quality_temperature
        head_dim = out_dim // n_heads

        self.W = nn.Linear(in_dim, head_dim * n_heads, bias=False)
        self.a_src = nn.Parameter(torch.zeros(n_heads, head_dim))
        self.a_dst = nn.Parameter(torch.zeros(n_heads, head_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

    def _build_log_q(self, mask: Optional[torch.Tensor],
                     quality: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Return the soft log-quality penalty, or use the hard binary path."""

        if self.quality_combine == "binary":
            return None

        if quality is None:

            if mask is None:
                return None
            q = mask.float()
        elif self.quality_combine == "soft_only":
            q = quality
        elif self.quality_combine == "soft_and_binary":
            q = quality * (mask.float() if mask is not None else 1.0)
        elif self.quality_combine == "max":
            mfloat = mask.float() if mask is not None else torch.zeros_like(quality)
            q = torch.max(quality, mfloat)
        else:
            raise ValueError(f"Unknown quality_combine: {self.quality_combine}")

        q = q.clamp(min=self.quality_floor, max=1.0)
        log_q = torch.log(q) / max(self.quality_temperature, 1e-6)
        return log_q

    def forward(self, x: torch.Tensor, adj: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                quality: Optional[torch.Tensor] = None) -> torch.Tensor:

        B, N, _ = x.shape
        H = self.n_heads
        D = self.out_dim // H

        h = self.W(x).view(B, N, H, D)
        e_src = (h * self.a_src).sum(-1)
        e_dst = (h * self.a_dst).sum(-1)
        attn = self.leaky_relu(e_src.unsqueeze(2) + e_dst.unsqueeze(1))

        adj_mask = adj.unsqueeze(0).unsqueeze(-1)
        attn = attn.masked_fill(adj_mask == 0, float("-inf"))

        if self.quality_combine == "binary":

            if mask is not None:
                src_mask = mask.unsqueeze(1).unsqueeze(-1)
                attn = attn.masked_fill(~src_mask, float("-inf"))
                dst_mask = mask.unsqueeze(2).unsqueeze(-1)
                attn = attn.masked_fill(~dst_mask, float("-inf"))
        else:

            log_q = self._build_log_q(mask, quality)
            if log_q is not None:

                attn = attn + log_q.unsqueeze(1).unsqueeze(-1)
                attn = attn + log_q.unsqueeze(2).unsqueeze(-1)

        attn = F.softmax(attn, dim=2)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        h_perm = h.permute(0, 2, 1, 3)
        attn_perm = attn.permute(0, 3, 1, 2)
        out = torch.matmul(attn_perm, h_perm)
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)
        return out


class GATBlock(nn.Module):
    """Apply graph attention, residual addition and layer normalisation."""

    def __init__(self, dim: int, n_heads: int, dropout: float,
                 quality_combine: str = "binary",
                 quality_floor: float = 1e-3,
                 quality_temperature: float = 1.0):
        super().__init__()
        self.gat = QualityAwareGATLayer(
            dim, dim, n_heads, dropout,
            quality_combine=quality_combine,
            quality_floor=quality_floor,
            quality_temperature=quality_temperature,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, adj, mask, quality=None):
        h_new = self.gat(h, adj, mask, quality)
        return self.norm(h_new + h)


class MaskedReadout(nn.Module):
    """Average node embeddings using quality or binary availability weights."""

    def forward(self, h: torch.Tensor, mask: torch.Tensor,
                quality: Optional[torch.Tensor] = None,
                use_quality: bool = False) -> torch.Tensor:
        if use_quality and quality is not None:
            w = quality.float()
        else:
            w = mask.float()
        w = w.unsqueeze(-1)
        denom = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return (h * w).sum(dim=1) / denom.squeeze(1)


class Classifier(nn.Module):
    """Map a participant representation to a binary-classification logit."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class TemporalFrontend(nn.Module):
    """Apply the GAP encoder separately to each channel."""

    def __init__(self, hidden_dim: int, n_temporal_layers: int, dropout: float):
        super().__init__()
        self.temporal = TemporalEncoder(
            in_channels=2, hidden_dim=hidden_dim,
            n_layers=n_temporal_layers, dropout=dropout,
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        B, T, N, M = signal.shape
        x = signal.permute(0, 2, 1, 3).reshape(B * N, T, M)
        return self.temporal(x).view(B, N, -1)


class TemporalFrontendV2(nn.Module):
    """Apply the attention encoder separately to each channel."""

    def __init__(self, hidden_dim: int, encoder_type: str = "cnn_attn",
                 dropout: float = 0.3, n_stem_layers: int = 2,
                 encoder_kwargs: Optional[dict] = None):
        super().__init__()
        kw = dict(in_channels=2, hidden_dim=hidden_dim,
                  n_stem_layers=n_stem_layers, dropout=dropout)
        if encoder_kwargs:
            kw.update(encoder_kwargs)
        self.encoder_type = encoder_type
        self.temporal = build_temporal_encoder(encoder_type, **kw)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:

        B, T, N, M = signal.shape

        x = signal.permute(0, 2, 3, 1).reshape(B * N, M, T)
        return self.temporal(x).view(B, N, -1)


class GATBackend(nn.Module):
    """Apply graph blocks, weighted readout and classification."""

    def __init__(self, hidden_dim: int, n_heads: int, n_gat_layers: int,
                 dropout: float,
                 quality_combine: str = "binary",
                 quality_floor: float = 1e-3,
                 quality_temperature: float = 1.0,
                 readout_use_quality: bool = False):
        super().__init__()
        self.gat_blocks = nn.ModuleList([
            GATBlock(hidden_dim, n_heads, dropout,
                     quality_combine=quality_combine,
                     quality_floor=quality_floor,
                     quality_temperature=quality_temperature)
            for _ in range(n_gat_layers)
        ])
        self.readout = MaskedReadout()
        self.readout_use_quality = readout_use_quality
        self.classifier = Classifier(hidden_dim, hidden_dim, dropout)

    def forward(self, h, adj, mask, quality=None):
        for block in self.gat_blocks:
            h = block(h, adj, mask, quality)
        g = self.readout(h, mask, quality=quality, use_quality=self.readout_use_quality)
        return self.classifier(g)


class ModelV5_SoftMask(nn.Module):
    """SoftMaskGAT with global average pooling over time."""

    def __init__(self, hidden_dim: int = 64, n_heads: int = 4,
                 n_gat_layers: int = 2, n_temporal_layers: int = 2,
                 dropout: float = 0.3, residual_scale: float = 1.0,
                 residual_mlp_layers: int = 2,
                 quality_combine: str = "soft_only",
                 quality_floor: float = 1e-3,
                 quality_temperature: float = 1.0,
                 readout_use_quality: bool = True,
                 anchor_use_quality: bool = True,
                 **kwargs):
        super().__init__()
        self.residual_scale = residual_scale
        self.anchor_use_quality = anchor_use_quality

        self.ts_frontend = TemporalFrontend(hidden_dim, n_temporal_layers, dropout)

        if residual_mlp_layers == 1:
            self.anc_residual = nn.Sequential(
                nn.LayerNorm(N_ANCHOR_FEATURES),
                nn.Linear(N_ANCHOR_FEATURES, hidden_dim),
            )
        else:
            self.anc_residual = nn.Sequential(
                nn.LayerNorm(N_ANCHOR_FEATURES),
                nn.Linear(N_ANCHOR_FEATURES, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.backend = GATBackend(
            hidden_dim, n_heads, n_gat_layers, dropout,
            quality_combine=quality_combine,
            quality_floor=quality_floor,
            quality_temperature=quality_temperature,
            readout_use_quality=readout_use_quality,
        )

    def forward(self, signal, adj, mask, x_anc, quality=None):
        h_ts = self.ts_frontend(signal)
        h_anc = self.anc_residual(x_anc)

        if self.anchor_use_quality and quality is not None:
            gate = quality.unsqueeze(-1).float()
        else:
            gate = mask.unsqueeze(-1).float()
        h_anc = h_anc * gate

        h_fused = h_ts + self.residual_scale * h_anc
        return self.backend(h_fused, adj, mask, quality=quality)


class ModelV6_RealSTGAT(nn.Module):
    """SoftMaskGAT with attention-based temporal aggregation."""

    def __init__(self, hidden_dim: int = 64, n_heads: int = 4,
                 n_gat_layers: int = 2, n_temporal_layers: int = 2,
                 dropout: float = 0.3, residual_scale: float = 1.0,
                 residual_mlp_layers: int = 2,
                 quality_combine: str = "soft_only",
                 quality_floor: float = 1e-3,
                 quality_temperature: float = 1.0,
                 readout_use_quality: bool = True,
                 anchor_use_quality: bool = True,

                 temporal_encoder_type: str = "cnn_attn",
                 temporal_encoder_kwargs: Optional[dict] = None,
                 **kwargs):
        super().__init__()
        self.residual_scale = residual_scale
        self.anchor_use_quality = anchor_use_quality

        self.ts_frontend = TemporalFrontendV2(
            hidden_dim=hidden_dim,
            encoder_type=temporal_encoder_type,
            dropout=dropout,
            n_stem_layers=n_temporal_layers,
            encoder_kwargs=temporal_encoder_kwargs,
        )

        if residual_mlp_layers == 1:
            self.anc_residual = nn.Sequential(
                nn.LayerNorm(N_ANCHOR_FEATURES),
                nn.Linear(N_ANCHOR_FEATURES, hidden_dim),
            )
        else:
            self.anc_residual = nn.Sequential(
                nn.LayerNorm(N_ANCHOR_FEATURES),
                nn.Linear(N_ANCHOR_FEATURES, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.backend = GATBackend(
            hidden_dim, n_heads, n_gat_layers, dropout,
            quality_combine=quality_combine,
            quality_floor=quality_floor,
            quality_temperature=quality_temperature,
            readout_use_quality=readout_use_quality,
        )

    def forward(self, signal, adj, mask, x_anc, quality=None):

        h_ts = self.ts_frontend(signal)
        h_anc = self.anc_residual(x_anc)

        if self.anchor_use_quality and quality is not None:
            gate = quality.unsqueeze(-1).float()
        else:
            gate = mask.unsqueeze(-1).float()
        h_anc = h_anc * gate

        h_fused = h_ts + self.residual_scale * h_anc
        return self.backend(h_fused, adj, mask, quality=quality)
