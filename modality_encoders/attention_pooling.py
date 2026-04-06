from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn


class AttentionPooling1D(nn.Module):
    """Pool a variable-length token set with a learned query.

    Input shape:
        tokens: (batch, items, dim)
        valid_mask: (batch, items), True where token is valid

    Output shape:
        pooled: (batch, dim)
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim: Final[int] = dim
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        assert tokens.ndim == 3, f"Expected (B, N, D), got {tuple(tokens.shape)}"
        assert valid_mask.ndim == 2, f"Expected (B, N), got {tuple(valid_mask.shape)}"
        batch, items, dim = tokens.shape
        assert dim == self.dim, f"Expected dim={self.dim}, got {dim}"
        assert valid_mask.shape == (batch, items)
        query = self.query.expand(batch, -1, -1)
        safe_tokens = tokens.clone()
        safe_mask = valid_mask.clone()
        empty_rows = ~safe_mask.any(dim=1)
        if bool(empty_rows.any()):
            safe_tokens[empty_rows, 0] = 0.0
            safe_mask[empty_rows, 0] = True
        key_padding_mask = ~safe_mask
        pooled, _ = self.attn(query=query, key=safe_tokens, value=safe_tokens, key_padding_mask=key_padding_mask)
        pooled = pooled.squeeze(1)
        return self.norm(pooled)


def _smoke_test() -> None:
    torch.manual_seed(0)
    module = AttentionPooling1D(dim=32, num_heads=4)
    tokens = torch.randn(3, 7, 32)
    valid_mask = torch.tensor([
        [True, True, True, True, False, False, False],
        [True, True, True, True, True, True, True],
        [True, False, False, False, False, False, False],
    ])
    pooled = module(tokens, valid_mask)
    assert pooled.shape == (3, 32)
    print('AttentionPooling1D smoke test passed:', tuple(pooled.shape))


if __name__ == '__main__':
    _smoke_test()
