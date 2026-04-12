from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn

try:
    from .attention_pooling import AttentionPooling1D
except ImportError:  # pragma: no cover - direct script execution
    from attention_pooling import AttentionPooling1D


class TemporalSpatialConvBlock(nn.Module):
    """Factorized temporal-then-spatial block over a within-second frame stack."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_kernel_size: int,
        spatial_kernel_size: int,
        spatial_stride: int,
        dropout: float,
    ) -> None:
        super().__init__()
        temporal_padding = temporal_kernel_size // 2
        spatial_padding = spatial_kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(temporal_kernel_size, 1, 1),
                padding=(temporal_padding, 0, 0),
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=(1, spatial_kernel_size, spatial_kernel_size),
                stride=(1, spatial_stride, spatial_stride),
                padding=(0, spatial_padding, spatial_padding),
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            nn.Dropout3d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalImageBranch(nn.Module):
    """Encode one second of raw or delta frames into per-frame latent features."""

    def __init__(
        self,
        in_channels: int,
        stem_channels: int,
        branch_channels: int,
        spatial_downsample: int,
        num_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            TemporalSpatialConvBlock(
                in_channels=in_channels,
                out_channels=stem_channels,
                temporal_kernel_size=3,
                spatial_kernel_size=7,
                spatial_stride=spatial_downsample,
                dropout=dropout,
            )
        ]
        channels = stem_channels
        for _ in range(max(num_blocks - 1, 0)):
            next_channels = branch_channels
            layers.append(
                TemporalSpatialConvBlock(
                    in_channels=channels,
                    out_channels=next_channels,
                    temporal_kernel_size=3,
                    spatial_kernel_size=3,
                    spatial_stride=2,
                    dropout=dropout,
                )
            )
            channels = next_channels
        self.net = nn.Sequential(*layers)
        self.output_channels: Final[int] = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 5, f"Expected (B, C, M, H, W), got {tuple(x.shape)}"
        hidden = self.net(x)
        hidden = hidden.mean(dim=(-1, -2))
        return hidden.transpose(1, 2)


class PerSecondImageTemporalConvEncoder(nn.Module):
    """Encode one token per second using raw and delta frame branches."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        max_items_per_second: int = 32,
        num_pool_heads: int = 4,
        dropout: float = 0.1,
        stem_channels: int = 16,
        branch_channels: int = 48,
        spatial_downsample: int = 4,
        num_branch_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels: Final[int] = in_channels
        self.embed_dim: Final[int] = embed_dim
        self.max_items_per_second: Final[int] = max_items_per_second
        self.raw_branch = TemporalImageBranch(
            in_channels=3,
            stem_channels=stem_channels,
            branch_channels=branch_channels,
            spatial_downsample=spatial_downsample,
            num_blocks=num_branch_blocks,
            dropout=dropout,
        )
        self.delta_branch = TemporalImageBranch(
            in_channels=3,
            stem_channels=stem_channels,
            branch_channels=branch_channels,
            spatial_downsample=spatial_downsample,
            num_blocks=num_branch_blocks,
            dropout=dropout,
        )
        pool_heads = max(1, min(num_pool_heads, self.raw_branch.output_channels))
        self.raw_pool = AttentionPooling1D(dim=self.raw_branch.output_channels, num_heads=pool_heads, dropout=dropout)
        self.delta_pool = AttentionPooling1D(dim=self.delta_branch.output_channels, num_heads=pool_heads, dropout=dropout)
        self.fuse = nn.Sequential(
            nn.LayerNorm(self.raw_branch.output_channels * 2),
            nn.Linear(self.raw_branch.output_channels * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def _adapt_channels(self, frames: torch.Tensor) -> torch.Tensor:
        if self.in_channels == 3:
            return frames
        if self.in_channels == 1:
            return frames.repeat(1, 1, 1, 3, 1, 1)
        raise ValueError(f'Unsupported in_channels={self.in_channels}')

    def _last_valid_frames(self, frames: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch, seconds, items, channels, height, width = frames.shape
        counts = valid_mask.long().sum(dim=-1)
        last_indices = counts.sub(1).clamp_min(0)
        gather_index = last_indices.view(batch, seconds, 1, 1, 1, 1).expand(-1, -1, 1, channels, height, width)
        last_frames = frames.gather(dim=2, index=gather_index).squeeze(2)
        has_valid = counts > 0
        return last_frames * has_valid.view(batch, seconds, 1, 1, 1)

    def _compute_delta_frames(self, frames: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        previous_frames = torch.zeros_like(frames)
        previous_frames[:, :, 1:] = frames[:, :, :-1]
        previous_last = self._last_valid_frames(frames, valid_mask)
        previous_frames[:, 1:, 0] = previous_last[:, :-1]
        delta = frames - previous_frames
        return delta * valid_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    def forward(self, frames: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        assert frames.ndim == 6, f"Expected (B, T, M, C, H, W), got {tuple(frames.shape)}"
        assert valid_mask.ndim == 3, f"Expected (B, T, M), got {tuple(valid_mask.shape)}"
        batch, seconds, items, channels, height, width = frames.shape
        assert channels == self.in_channels, f"Expected channels={self.in_channels}, got {channels}"
        assert valid_mask.shape == (batch, seconds, items)
        assert items <= self.max_items_per_second, f"items={items} exceeds max_items_per_second={self.max_items_per_second}"

        frames = self._adapt_channels(frames)
        frames = frames * valid_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        deltas = self._compute_delta_frames(frames, valid_mask)

        raw_branch_input = frames.reshape(batch * seconds, items, 3, height, width).transpose(1, 2)
        delta_branch_input = deltas.reshape(batch * seconds, items, 3, height, width).transpose(1, 2)
        raw_features = self.raw_branch(raw_branch_input)
        delta_features = self.delta_branch(delta_branch_input)
        mask = valid_mask.reshape(batch * seconds, items)

        raw_summary = self.raw_pool(raw_features, mask)
        delta_summary = self.delta_pool(delta_features, mask)
        fused = self.fuse(torch.cat([raw_summary, delta_summary], dim=-1))
        return fused.reshape(batch, seconds, self.embed_dim)


def _smoke_test() -> None:
    torch.manual_seed(0)
    module = PerSecondImageTemporalConvEncoder(
        in_channels=3,
        embed_dim=64,
        max_items_per_second=8,
        num_pool_heads=4,
        stem_channels=8,
        branch_channels=16,
        num_branch_blocks=2,
    )
    frames = torch.randn(2, 3, 5, 3, 64, 64)
    valid_mask = torch.tensor([
        [[True, True, True, False, False] for _ in range(3)],
        [[True, True, True, True, True] for _ in range(3)],
    ])
    tokens = module(frames, valid_mask)
    assert tokens.shape == (2, 3, 64)
    loss = tokens.square().mean()
    loss.backward()
    print('PerSecondImageTemporalConvEncoder smoke test passed:', tuple(tokens.shape))


if __name__ == '__main__':
    _smoke_test()
