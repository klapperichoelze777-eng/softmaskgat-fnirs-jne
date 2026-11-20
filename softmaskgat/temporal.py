"""Temporal attention encoder used by SoftMaskGAT."""

import math

import torch
from torch import nn


class _ConvStem(nn.Module):
    """Shared form of the two-stage convolutional stem."""

    def __init__(
        self,
        in_channels: int = 2,
        hidden_dim: int = 64,
        n_layers: int = 2,
        kernel_size: int = 7,
        pool_size: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        ch = in_channels
        for i in range(n_layers):
            out_ch = (hidden_dim if i == n_layers - 1
                      else max(16, hidden_dim // (2 ** (n_layers - 1 - i))))
            layers.extend([
                nn.Conv1d(ch, out_ch, kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(pool_size),
                nn.Dropout(dropout),
            ])
            ch = out_ch
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.encoder(x)


class _SinCosPositionalEncoding(nn.Module):
    """Add fixed sine and cosine positional encodings."""

    def __init__(self, dim: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x + self.pe[:, : x.size(1), :]


class CnnAttnEncoder(nn.Module):
    """Use positional encoding, pre-normalised attention and mean pooling."""

    def __init__(
        self,
        in_channels: int = 2,
        hidden_dim: int = 64,
        n_stem_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.3,
        max_pe_len: int = 256,
        **kwargs,
    ):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

        self.stem = _ConvStem(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            n_layers=n_stem_layers,
            dropout=dropout,
        )
        self.pos_enc = _SinCosPositionalEncoding(hidden_dim, max_len=max_pe_len)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim

        self.return_attn = False
        self._last_attn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        feat = feat.transpose(1, 2).contiguous()
        feat = self.pos_enc(feat)

        attn_in = self.norm1(feat)
        if self.return_attn:
            attn_out, attn_w = self.attn(
                attn_in, attn_in, attn_in,
                need_weights=True, average_attn_weights=False,
            )

            self._last_attn = attn_w.detach().cpu()
        else:
            attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        feat = feat + attn_out
        feat = feat + self.ffn(self.norm2(feat))
        emb = feat.mean(dim=1)
        return emb



def build_temporal_encoder(encoder_type, **kwargs):
    if encoder_type != "cnn_attn":
        raise ValueError("The published attention branch uses cnn_attn.")
    return CnnAttnEncoder(**kwargs)
