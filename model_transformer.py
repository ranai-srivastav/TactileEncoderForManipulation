from __future__ import annotations

from typing import Dict, Final, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from modality_encoders import (
    PerSecondImageTemporalConvEncoder,
    PerSecondImageViTEncoder,
    PerSecondTimeseriesConvEncoder,
)

IMAGE_MODALITIES: Final[tuple[str, ...]] = ('tactile', 'rgb', 'depth', 'side_cam', 'top_cam')
TIMESERIES_MODALITIES: Final[tuple[str, ...]] = ('ft', 'gripper', 'robot')
STATIC_MODALITIES: Final[tuple[str, ...]] = ('gripper_force',)
DEFAULT_MODALITIES: Final[tuple[str, ...]] = IMAGE_MODALITIES + TIMESERIES_MODALITIES + STATIC_MODALITIES

IMAGE_CHANNELS: Final[Mapping[str, int]] = {
    'tactile': 3,
    'rgb': 3,
    'depth': 1,
    'side_cam': 3,
    'top_cam': 3,
}
TIMESERIES_DIMS: Final[Mapping[str, int]] = {
    'ft': 6,
    'gripper': 2,
    'robot': 12,
}


class ScalarProjector(nn.Module):
    """Project scalar metadata to the shared token space."""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.embed_dim: Final[int] = embed_dim
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2 and x.shape[-1] == 1, f'Expected (B, 1), got {tuple(x.shape)}'
        return self.net(x)


class SlipTransformer(nn.Module):
    """Hierarchical transformer for slip classification.

    Stage 1: modality-specific encoders compress all observations within one second
    into one token per modality.

    Stage 2: a transformer consumes the flattened token sequence ordered by
    `(second, modality)` plus learned time and modality embeddings.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        modalities: Optional[Sequence[str]] = None,
        max_seconds: int = 64,
        image_encoder_type: str = 'vit',
        image_encoder_model_name: str = 'vit_small_patch16_224',
        vit_pretrained: bool = False,
        max_items_per_second: Optional[Dict[str, int]] = None,
        timeseries_conv_channels: int = 128,
        timeseries_num_conv_layers: int = 3,
        timeseries_kernel_size: int = 5,
        scalar_hidden_dim: int = 128,
        image_temporal_stem_channels: int = 16,
        image_temporal_branch_channels: int = 48,
        image_temporal_num_blocks: int = 3,
        image_temporal_spatial_downsample: int = 4,
    ) -> None:
        super().__init__()
        self.modalities: Final[tuple[str, ...]] = tuple(modalities or DEFAULT_MODALITIES)
        self.modality_to_idx: Final[dict[str, int]] = {name: idx for idx, name in enumerate(self.modalities)}
        self.max_seconds: Final[int] = max_seconds
        self.d_model: Final[int] = d_model
        self.max_items_per_second = dict(max_items_per_second or {
            'tactile': 30,
            'rgb': 5,
            'depth': 5,
            'side_cam': 30,
            'top_cam': 30,
            'ft': 100,
            'gripper': 10,
            'robot': 2970,
        })

        if image_encoder_type not in {'vit', 'resnet', 'temporal_cnn'}:
            raise ValueError(f'Unsupported image_encoder_type={image_encoder_type}')

        if image_encoder_type == 'temporal_cnn':
            self.image_encoders = nn.ModuleDict({
                name: PerSecondImageTemporalConvEncoder(
                    in_channels=IMAGE_CHANNELS[name],
                    embed_dim=d_model,
                    max_items_per_second=self.max_items_per_second[name],
                    num_pool_heads=nhead,
                    dropout=dropout,
                    stem_channels=image_temporal_stem_channels,
                    branch_channels=image_temporal_branch_channels,
                    spatial_downsample=image_temporal_spatial_downsample,
                    num_branch_blocks=image_temporal_num_blocks,
                )
                for name in self.modalities if name in IMAGE_CHANNELS
            })
        else:
            self.image_encoders = nn.ModuleDict({
                name: PerSecondImageViTEncoder(
                    in_channels=IMAGE_CHANNELS[name],
                    embed_dim=d_model,
                    frame_model_name=image_encoder_model_name,
                    pretrained=vit_pretrained,
                    max_items_per_second=self.max_items_per_second[name],
                    num_pool_heads=nhead,
                    dropout=dropout,
                )
                for name in self.modalities if name in IMAGE_CHANNELS
            })
        self.timeseries_encoders = nn.ModuleDict({
            name: PerSecondTimeseriesConvEncoder(
                input_dim=TIMESERIES_DIMS[name],
                embed_dim=d_model,
                conv_channels=timeseries_conv_channels,
                num_conv_layers=timeseries_num_conv_layers,
                kernel_size=timeseries_kernel_size,
                num_pool_heads=nhead,
                dropout=dropout,
            )
            for name in self.modalities if name in TIMESERIES_DIMS
        })
        self.scalar_encoders = nn.ModuleDict({
            'gripper_force': ScalarProjector(embed_dim=d_model, hidden_dim=scalar_hidden_dim, dropout=dropout)
        }) if 'gripper_force' in self.modalities else nn.ModuleDict()

        self.time_embedding = nn.Embedding(max_seconds, d_model)
        self.modality_embedding = nn.Embedding(len(self.modalities), d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.input_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.time_embedding.weight, std=0.02)
        nn.init.normal_(self.modality_embedding.weight, std=0.02)

    def _encode_scalar_modality(self, batch: Dict[str, torch.Tensor], name: str, seconds: int) -> torch.Tensor:
        scalar = batch[name]
        embedded = self.scalar_encoders[name](scalar)
        return embedded.unsqueeze(1).expand(-1, seconds, -1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert 'seconds' in batch, 'Batch must contain seconds'
        seconds_tensor = batch['seconds']
        assert seconds_tensor.ndim == 2, f'Expected seconds shape (B, T), got {tuple(seconds_tensor.shape)}'
        batch_size, seconds = seconds_tensor.shape
        assert seconds <= self.max_seconds, f'Sequence length {seconds} exceeds max_seconds={self.max_seconds}'

        time_ids = torch.arange(seconds, device=seconds_tensor.device)
        time_emb = self.time_embedding(time_ids).unsqueeze(0)

        token_blocks: List[torch.Tensor] = []
        padding_masks: List[torch.Tensor] = []
        for name in self.modalities:
            if name in self.image_encoders:
                token = self.image_encoders[name](batch[name], batch[f'{name}_valid_mask'])
                valid = batch[f'{name}_valid_mask'].any(dim=-1)
            elif name in self.timeseries_encoders:
                token = self.timeseries_encoders[name](batch[name], batch[f'{name}_valid_mask'])
                valid = batch[f'{name}_valid_mask'].any(dim=-1)
            elif name in self.scalar_encoders:
                token = self._encode_scalar_modality(batch, name, seconds)
                valid = batch['sequence_mask']
            else:
                raise KeyError(f'Unsupported modality: {name}')

            assert token.shape == (batch_size, seconds, self.d_model), (
                f'Expected {(batch_size, seconds, self.d_model)} for {name}, got {tuple(token.shape)}'
            )
            modality_idx = torch.full((1,), self.modality_to_idx[name], device=seconds_tensor.device, dtype=torch.long)
            modality_emb = self.modality_embedding(modality_idx).view(1, 1, self.d_model)
            token = self.input_norm(token + time_emb + modality_emb)
            token_blocks.append(token)
            padding_masks.append(~valid)

        tokens = torch.stack(token_blocks, dim=2).reshape(batch_size, seconds * len(self.modalities), self.d_model)
        padding_mask = torch.stack(padding_masks, dim=2).reshape(batch_size, seconds * len(self.modalities))
        cls = self.cls_token.expand(batch_size, -1, -1)
        cls_mask = torch.zeros(batch_size, 1, device=seconds_tensor.device, dtype=torch.bool)
        tokens = torch.cat([cls, tokens], dim=1)
        padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        encoded = self.transformer(tokens, src_key_padding_mask=padding_mask)
        cls_out = encoded[:, 0]
        logits = self.head(cls_out).squeeze(-1)
        assert logits.shape == (batch_size,), f'Expected logits shape {(batch_size,)}, got {tuple(logits.shape)}'
        return logits


def _make_fake_batch(batch_size: int = 1, seconds: int = 2, modalities: Sequence[str] = ('rgb', 'ft', 'gripper_force')) -> Dict[str, torch.Tensor]:
    torch.manual_seed(0)
    batch: Dict[str, torch.Tensor] = {
        'seconds': torch.arange(seconds).unsqueeze(0).repeat(batch_size, 1),
        'sequence_mask': torch.ones(batch_size, seconds, dtype=torch.bool),
        'gripper_force': torch.randn(batch_size, 1),
    }
    image_shapes = {
        'tactile': (4, 3),
        'rgb': (3, 3),
        'depth': (2, 1),
        'side_cam': (4, 3),
        'top_cam': (4, 3),
    }
    for name, (items, channels) in image_shapes.items():
        if name not in modalities:
            continue
        batch[name] = torch.randn(batch_size, seconds, items, channels, 224, 224)
        batch[f'{name}_valid_mask'] = torch.ones(batch_size, seconds, items, dtype=torch.bool)
    ts_shapes = {'ft': (8, 6), 'gripper': (4, 2), 'robot': (8, 12)}
    for name, (items, dim) in ts_shapes.items():
        if name not in modalities:
            continue
        batch[name] = torch.randn(batch_size, seconds, items, dim)
        batch[f'{name}_valid_mask'] = torch.ones(batch_size, seconds, items, dtype=torch.bool)
    return batch


def _smoke_test() -> None:
    smoke_modalities = ('rgb', 'ft', 'gripper_force')
    model = SlipTransformer(
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        modalities=smoke_modalities,
        image_encoder_type='temporal_cnn',
        image_encoder_model_name='temporal_cnn_small',
        max_seconds=4,
        max_items_per_second={'rgb': 3, 'ft': 8, 'gripper': 4, 'robot': 8, 'tactile': 4, 'depth': 2, 'side_cam': 4, 'top_cam': 4},
        timeseries_conv_channels=32,
        scalar_hidden_dim=32,
        image_temporal_stem_channels=8,
        image_temporal_branch_channels=16,
        image_temporal_num_blocks=2,
    )
    batch = _make_fake_batch(modalities=smoke_modalities)
    logits = model(batch)
    assert logits.shape == (1,)
    print('SlipTransformer smoke test passed:', tuple(logits.shape))


if __name__ == '__main__':
    _smoke_test()
