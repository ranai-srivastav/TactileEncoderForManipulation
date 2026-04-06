from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn

try:
    from .attention_pooling import AttentionPooling1D
except ImportError:  # pragma: no cover - direct script execution
    from attention_pooling import AttentionPooling1D


class ConvBlock1D(nn.Module):
    """A lightweight 1D convolution block over within-second sensor sequences."""

    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerSecondTimeseriesConvEncoder(nn.Module):
    """Encode a within-second sensor sequence into one token.

    Input:
        values: (batch, seconds, items, input_dim)
        valid_mask: (batch, seconds, items)

    Output:
        tokens: (batch, seconds, embed_dim)
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        conv_channels: int = 128,
        num_conv_layers: int = 3,
        kernel_size: int = 5,
        num_pool_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim: Final[int] = input_dim
        self.embed_dim: Final[int] = embed_dim
        self.conv_channels: Final[int] = conv_channels
        self.input_proj = nn.Linear(input_dim, conv_channels)
        self.conv_blocks = nn.ModuleList([
            ConvBlock1D(conv_channels, kernel_size=kernel_size, dropout=dropout)
            for _ in range(num_conv_layers)
        ])
        self.output_proj = nn.Linear(conv_channels, embed_dim)
        self.pool = AttentionPooling1D(dim=embed_dim, num_heads=num_pool_heads, dropout=dropout)

    def forward(self, values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        assert values.ndim == 4, f"Expected (B, T, M, D), got {tuple(values.shape)}"
        assert valid_mask.ndim == 3, f"Expected (B, T, M), got {tuple(valid_mask.shape)}"
        batch, seconds, items, input_dim = values.shape
        assert input_dim == self.input_dim, f"Expected input_dim={self.input_dim}, got {input_dim}"
        assert valid_mask.shape == (batch, seconds, items)

        hidden = self.input_proj(values)  # (B, T, M, C)
        hidden = hidden.reshape(batch * seconds, items, self.conv_channels)
        mask = valid_mask.reshape(batch * seconds, items)

        hidden = hidden.transpose(1, 2)  # (BT, C, M)
        expanded_mask = mask.unsqueeze(1).float()
        hidden = hidden * expanded_mask
        for block in self.conv_blocks:
            hidden = block(hidden)
            hidden = hidden * expanded_mask
        hidden = hidden.transpose(1, 2)  # (BT, M, C)
        hidden = self.output_proj(hidden)
        pooled = self.pool(hidden, mask)
        return pooled.reshape(batch, seconds, self.embed_dim)


def _smoke_test() -> None:
    torch.manual_seed(0)
    module = PerSecondTimeseriesConvEncoder(input_dim=6, embed_dim=64, conv_channels=32, num_conv_layers=2, num_pool_heads=4)
    values = torch.randn(2, 5, 11, 6)
    valid_mask = torch.tensor([
        [[True] * 9 + [False] * 2 for _ in range(5)],
        [[True] * 11 for _ in range(5)],
    ])
    tokens = module(values, valid_mask)
    assert tokens.shape == (2, 5, 64)
    print('PerSecondTimeseriesConvEncoder smoke test passed:', tuple(tokens.shape))


if __name__ == '__main__':
    _smoke_test()
