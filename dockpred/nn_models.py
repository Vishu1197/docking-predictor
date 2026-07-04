"""Deep-learning architectures for the tabular docking-score regressor.

These definitions are the single source of truth shared by the training pipeline
(``src.train_dl``) and the inference package (``dockpred.base_models``), so a
checkpoint always loads back into the exact architecture that produced it. The
module depends only on ``torch`` so the shippable package stays lightweight.

Each network predicts the **standardised** target; the training-time target
mean/std are stored alongside the checkpoint and applied by the inference
wrapper to recover raw kcal/mol. ``forward`` returns a shape ``(batch,)`` tensor.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(sizes: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU(),
                   nn.BatchNorm1d(sizes[i + 1]), nn.Dropout(dropout)]
    layers += [nn.Linear(sizes[-2], sizes[-1])]
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.BatchNorm1d(dim),
            nn.Dropout(dropout), nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualTabularNet(nn.Module):
    """Deep residual MLP (the strongest DL model from the original project)."""

    def __init__(self, input_dim: int, hidden: int = 512, n_blocks: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.BatchNorm1d(hidden))
        self.residuals = nn.Sequential(
            *[ResidualBlock(hidden, dropout) for _ in range(n_blocks)])
        self.output_layer = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))

    def forward(self, x):
        x = self.input_layer(x)
        x = self.residuals(x)
        return self.output_layer(x).squeeze(-1)


class TabularDNN(nn.Module):
    """Plain deep MLP with batch-norm and dropout."""

    def __init__(self, input_dim: int, hidden: tuple[int, ...] = (1024, 512, 256),
                 dropout: float = 0.3):
        super().__init__()
        self.network = _mlp([input_dim, *hidden, 1], dropout)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class WideDeepNet(nn.Module):
    """Wide (linear) + deep (MLP) network; the wide path preserves linear signal."""

    def __init__(self, input_dim: int, hidden: tuple[int, ...] = (512, 256, 128),
                 dropout: float = 0.2):
        super().__init__()
        self.wide = nn.Linear(input_dim, 1)
        self.deep = _mlp([input_dim, *hidden, 1], dropout)

    def forward(self, x):
        return (self.wide(x) + self.deep(x)).squeeze(-1)


class GatedAttentionNet(nn.Module):
    """Feature-wise gated attention + MLP.

    A learned attention gate reweights each descriptor before a deep MLP. Unlike
    a token-transformer this is O(d) per sample (no sequence self-attention), so
    it is cheap on CPU while still modelling per-feature importance.
    """

    def __init__(self, input_dim: int, hidden: tuple[int, ...] = (512, 256, 128),
                 dropout: float = 0.2):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.Tanh(),
            nn.Linear(input_dim, input_dim), nn.Sigmoid())
        self.network = _mlp([input_dim, *hidden, 1], dropout)

    def forward(self, x):
        x = x * self.gate(x)
        return self.network(x).squeeze(-1)


class FTTransformerLite(nn.Module):
    """Lightweight feature-tokeniser transformer for tabular data (CPU-friendly).

    Each scalar feature is projected to a ``d_token`` embedding; a CLS token
    attends over the feature tokens through a few transformer encoder layers and
    its final representation is decoded to the target. Kept small (few layers,
    modest token width) so it is tractable on CPU.
    """

    def __init__(self, input_dim: int, d_token: int = 32, n_layers: int = 2,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_token = d_token
        self.feature_emb = nn.Parameter(torch.randn(input_dim, d_token) * 0.02)
        self.feature_bias = nn.Parameter(torch.zeros(input_dim, d_token))
        self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 2,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.Linear(d_token, 1))

    def forward(self, x):
        # x: (b, f) -> tokens (b, f, d)
        tokens = x.unsqueeze(-1) * self.feature_emb + self.feature_bias
        cls = self.cls.expand(x.size(0), -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        out = self.encoder(seq)
        return self.head(out[:, 0]).squeeze(-1)


# name -> (class, default constructor kwargs)
ARCHITECTURES: dict[str, tuple[type[nn.Module], dict]] = {
    "ResidualTabularNet": (ResidualTabularNet, {}),
    "TabularDNN": (TabularDNN, {}),
    "WideDeepNet": (WideDeepNet, {}),
    "GatedAttentionNet": (GatedAttentionNet, {}),
    "FTTransformerLite": (FTTransformerLite, {}),
}


def build_network(name: str, input_dim: int, **kwargs) -> nn.Module:
    if name not in ARCHITECTURES:
        raise ValueError(f"Unknown DL architecture '{name}'. "
                         f"Known: {sorted(ARCHITECTURES)}")
    cls, defaults = ARCHITECTURES[name]
    return cls(input_dim, **{**defaults, **kwargs})
