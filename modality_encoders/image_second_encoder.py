from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn
import timm

try:
    from .attention_pooling import AttentionPooling1D
except ImportError:  # pragma: no cover - direct script execution
    from attention_pooling import AttentionPooling1D


class PerSecondImageViTEncoder(nn.Module):
    """Encode all frames within a second using a ViT frame encoder + attention pooling.

    Input:
        frames: (batch, seconds, items, channels, height, width)
        valid_mask: (batch, seconds, items)

    Output:
        tokens: (batch, seconds, embed_dim)
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        vit_model_name: str = 'vit_small_patch16_224',
        pretrained: bool = False,
        max_items_per_second: int = 32,
        num_pool_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels: Final[int] = in_channels
        self.embed_dim: Final[int] = embed_dim
        self.max_items_per_second: Final[int] = max_items_per_second
        self.frame_encoder = timm.create_model(vit_model_name, pretrained=pretrained, num_classes=0, in_chans=3)
        for param in self.frame_encoder.parameters():
            param.requires_grad = False
        self.frame_encoder.eval()
        frame_dim = getattr(self.frame_encoder, 'num_features')
        self.frame_proj = nn.Linear(frame_dim, embed_dim)
        self.item_position_embedding = nn.Embedding(max_items_per_second, embed_dim)
        self.pool = AttentionPooling1D(dim=embed_dim, num_heads=num_pool_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.normal_(self.item_position_embedding.weight, std=0.02)

    def train(self, mode: bool = True) -> PerSecondImageViTEncoder:
        super().train(mode)
        self.frame_encoder.eval()
        return self

    def _adapt_channels(self, frames: torch.Tensor) -> torch.Tensor:
        if self.in_channels == 3:
            return frames
        if self.in_channels == 1:
            return frames.repeat(1, 1, 1, 3, 1, 1)
        raise ValueError(f'Unsupported in_channels={self.in_channels}')

    def forward(self, frames: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        assert frames.ndim == 6, f"Expected (B, T, M, C, H, W), got {tuple(frames.shape)}"
        assert valid_mask.ndim == 3, f"Expected (B, T, M), got {tuple(valid_mask.shape)}"
        batch, seconds, items, channels, height, width = frames.shape
        assert channels == self.in_channels, f"Expected channels={self.in_channels}, got {channels}"
        assert valid_mask.shape == (batch, seconds, items)
        assert items <= self.max_items_per_second, f"items={items} exceeds max_items_per_second={self.max_items_per_second}"

        frames = self._adapt_channels(frames)
        frames = frames.reshape(batch * seconds * items, 3, height, width)
        with torch.no_grad():
            frame_tokens = self.frame_encoder(frames)
        frame_tokens = self.frame_proj(frame_tokens)
        frame_tokens = frame_tokens.reshape(batch * seconds, items, self.embed_dim)

        positions = torch.arange(items, device=frame_tokens.device)
        frame_tokens = frame_tokens + self.item_position_embedding(positions).unsqueeze(0)
        frame_tokens = self.norm(frame_tokens)
        pooled = self.pool(frame_tokens, valid_mask.reshape(batch * seconds, items))
        return pooled.reshape(batch, seconds, self.embed_dim)


def _smoke_test() -> None:
    torch.manual_seed(0)
    module = PerSecondImageViTEncoder(in_channels=3, embed_dim=64, vit_model_name='vit_tiny_patch16_224', max_items_per_second=8, num_pool_heads=4)
    frames = torch.randn(2, 3, 5, 3, 224, 224)
    valid_mask = torch.tensor([
        [[True, True, True, False, False] for _ in range(3)],
        [[True, True, True, True, True] for _ in range(3)],
    ])
    tokens = module(frames, valid_mask)
    assert tokens.shape == (2, 3, 64)
    print('PerSecondImageViTEncoder smoke test passed:', tuple(tokens.shape))


if __name__ == '__main__':
    _smoke_test()
