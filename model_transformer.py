import math
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn


IMAGE_MODALITIES = ('tactile', 'rgb', 'depth', 'side_cam', 'top_cam')
TIMESERIES_MODALITIES = ('ft', 'gripper', 'robot')
STATIC_MODALITIES = ('gripper_force',)
DEFAULT_MODALITIES = IMAGE_MODALITIES + TIMESERIES_MODALITIES + STATIC_MODALITIES


class TinyImageEncoder(nn.Module):
    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        widths = [32, 64, 128]
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(widths[0]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(widths[0], widths[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(widths[1]),
            nn.GELU(),
            nn.Conv2d(widths[1], widths[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(widths[2]),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(widths[2], out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StatsProjector(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # values: (B, T, M, D), valid_mask: (B, T, M)
        mask = valid_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=2).clamp_min(1.0)
        mean = (values * mask).sum(dim=2) / denom
        centered = (values - mean.unsqueeze(2)) * mask
        var = (centered.square().sum(dim=2) / denom)
        std = torch.sqrt(var + 1e-6)
        features = torch.cat([mean, std], dim=-1)
        return self.net(features)


class ScalarProjector(nn.Module):
    def __init__(self, out_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SlipTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        modalities: Optional[Sequence[str]] = None,
        max_seconds: int = 64,
        stats_hidden_dim: int = 256,
    ):
        super().__init__()
        self.modalities = tuple(modalities or DEFAULT_MODALITIES)
        self.modality_to_idx = {name: i for i, name in enumerate(self.modalities)}
        self.max_seconds = max_seconds
        self.d_model = d_model

        image_channels = {
            'tactile': 3,
            'rgb': 3,
            'depth': 1,
            'side_cam': 3,
            'top_cam': 3,
        }
        ts_dims = {
            'ft': 6,
            'gripper': 2,
            'robot': 12,
        }

        self.image_encoders = nn.ModuleDict({
            name: TinyImageEncoder(in_channels=image_channels[name], out_dim=d_model)
            for name in self.modalities if name in image_channels
        })
        self.timeseries_encoders = nn.ModuleDict({
            name: StatsProjector(input_dim=ts_dims[name], out_dim=d_model, hidden_dim=stats_hidden_dim)
            for name in self.modalities if name in ts_dims
        })
        self.scalar_encoders = nn.ModuleDict({
            'gripper_force': ScalarProjector(out_dim=d_model, hidden_dim=stats_hidden_dim)
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
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.time_embedding.weight, std=0.02)
        nn.init.normal_(self.modality_embedding.weight, std=0.02)

    def _encode_image_modality(self, batch: Dict[str, torch.Tensor], name: str) -> torch.Tensor:
        frames = batch[name]  # (B, T, M, C, H, W)
        valid_mask = batch[f'{name}_valid_mask']  # (B, T, M)
        mask = valid_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
        denom = mask.sum(dim=2).clamp_min(1.0)
        second_images = (frames * mask).sum(dim=2) / denom  # (B, T, C, H, W)
        B, T = second_images.shape[:2]
        encoded = self.image_encoders[name](second_images.reshape(B * T, *second_images.shape[2:]))
        return encoded.reshape(B, T, self.d_model)

    def _encode_timeseries_modality(self, batch: Dict[str, torch.Tensor], name: str) -> torch.Tensor:
        return self.timeseries_encoders[name](batch[name], batch[f'{name}_valid_mask'])

    def _encode_scalar_modality(self, batch: Dict[str, torch.Tensor], name: str, T: int) -> torch.Tensor:
        scalar = batch[name]  # (B, 1)
        embedded = self.scalar_encoders[name](scalar)  # (B, D)
        return embedded.unsqueeze(1).expand(-1, T, -1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        seconds = batch['seconds']  # (B, T)
        B, T = seconds.shape
        if T > self.max_seconds:
            raise ValueError(f'Sequence length {T} exceeds max_seconds={self.max_seconds}')

        token_blocks: List[torch.Tensor] = []
        pad_masks: List[torch.Tensor] = []

        time_idx = torch.arange(T, device=seconds.device)
        time_emb = self.time_embedding(time_idx).unsqueeze(0)  # (1, T, D)

        for name in self.modalities:
            if name in self.image_encoders:
                token = self._encode_image_modality(batch, name)
                valid = batch[f'{name}_valid_mask'].any(dim=-1)
            elif name in self.timeseries_encoders:
                token = self._encode_timeseries_modality(batch, name)
                valid = batch[f'{name}_valid_mask'].any(dim=-1)
            elif name in self.scalar_encoders:
                token = self._encode_scalar_modality(batch, name, T)
                valid = torch.ones(B, T, device=seconds.device, dtype=torch.bool)
            else:
                raise KeyError(f'Unsupported modality: {name}')

            modality_idx = torch.full((1,), self.modality_to_idx[name], device=seconds.device, dtype=torch.long)
            modality_emb = self.modality_embedding(modality_idx).view(1, 1, self.d_model)
            token = self.input_norm(token + time_emb + modality_emb)
            token_blocks.append(token)
            pad_masks.append(~valid)

        tokens = torch.stack(token_blocks, dim=2).reshape(B, T * len(self.modalities), self.d_model)
        padding_mask = torch.stack(pad_masks, dim=2).reshape(B, T * len(self.modalities))

        cls = self.cls_token.expand(B, -1, -1)
        cls_mask = torch.zeros(B, 1, device=seconds.device, dtype=torch.bool)
        tokens = torch.cat([cls, tokens], dim=1)
        padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        encoded = self.transformer(tokens, src_key_padding_mask=padding_mask)
        cls_out = encoded[:, 0]
        return self.head(cls_out).squeeze(-1)
