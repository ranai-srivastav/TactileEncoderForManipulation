from __future__ import annotations

from typing import Iterable, Sequence

import open_clip
import timm
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


ALL_MODALITIES = ("V", "T", "FT", "G", "GF")


class OpenCLIPVisionEncoder(nn.Module):
    """OpenCLIP ViT-B/32 visual tower with state-dict-compatible key names."""

    def __init__(self) -> None:
        super().__init__()
        # Match the OpenAI CLIP ViT-B/32 architecture used in the run while
        # letting the checkpoint provide the actual weights.
        self.visual = open_clip.create_model(
            "ViT-B-32",
            pretrained=None,
            load_weights=False,
            force_quick_gelu=True,
        ).visual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual(x)


class TactileViTSmallEncoder(nn.Module):
    """timm ViT-S/16 split to match the checkpoint naming layout."""

    def __init__(self) -> None:
        super().__init__()
        backbone = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)

        self.encoder = nn.Module()
        self.encoder.cls_token = backbone.cls_token
        self.encoder.pos_embed = backbone.pos_embed
        self.encoder.patch_embed = backbone.patch_embed
        self.encoder.pos_drop = backbone.pos_drop
        self.encoder.blocks = nn.Sequential(*list(backbone.blocks[:3]))

        self.trunk = nn.Module()
        self.trunk.blocks = nn.Sequential(*list(backbone.blocks[3:]))
        self.trunk.norm = backbone.norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder.patch_embed(x)
        cls = self.encoder.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.encoder.pos_embed
        x = self.encoder.pos_drop(x)
        for block in self.encoder.blocks:
            x = block(x)
        for block in self.trunk.blocks:
            x = block(x)
        x = self.trunk.norm(x)
        return x[:, 0]


class CLIPT3(nn.Module):
    """Checkpoint-compatible CLIPT3 model for multimodal evaluation."""

    def __init__(
        self,
        hidden_dim: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        modalities: Iterable[str] | None = None,
    ) -> None:
        super().__init__()
        self.modalities = set(modalities or ALL_MODALITIES)
        self.bidirectional = bidirectional

        self.rgb_encoder = OpenCLIPVisionEncoder()
        self.tactile_encoder = TactileViTSmallEncoder()

        self.v_proj = nn.Linear(512, 512)
        self.t_proj = nn.Linear(384, 512)
        self.ft_proj = nn.Linear(6, 128)
        self.g_proj = nn.Linear(2, 64)
        self.gf_proj = nn.Linear(1, 32)

        fusion_dim = 512 + 512 + 128 + 64 + 32
        self.projection = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Kept under the historical attribute name `lstm` to match the
        # checkpoint key layout even though the temporal model is a GRU.
        self.lstm = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        classifier_in = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def set_modalities(self, modalities: Iterable[str]) -> None:
        self.modalities = set(modalities)

    def _encode_image_stream(
        self,
        frames: torch.Tensor,
        encoder: nn.Module,
    ) -> torch.Tensor:
        batch, steps, frames_per_step = frames.shape[:3]
        flat = frames.reshape(batch * steps * frames_per_step, *frames.shape[3:])
        encoded = encoder(flat)
        encoded = encoded.reshape(batch, steps, frames_per_step, -1)
        return encoded.mean(dim=2)

    def forward(
        self,
        tactile: torch.Tensor,
        rgb: torch.Tensor,
        ft: torch.Tensor,
        gripper: torch.Tensor,
        gripper_force: torch.Tensor,
        lengths: Sequence[int] | torch.Tensor,
    ) -> torch.Tensor:
        # Zero raw inputs for ablation, but keep the encoder path intact.
        if "T" not in self.modalities:
            tactile = tactile * 0.0
        if "V" not in self.modalities:
            rgb = rgb * 0.0
        if "FT" not in self.modalities:
            ft = ft * 0.0
        if "G" not in self.modalities:
            gripper = gripper * 0.0
        if "GF" not in self.modalities:
            gripper_force = gripper_force * 0.0

        v = self.v_proj(self._encode_image_stream(rgb, self.rgb_encoder))
        t = self.t_proj(self._encode_image_stream(tactile, self.tactile_encoder))
        ft_proj = self.ft_proj(ft)
        g_proj = self.g_proj(gripper)
        gf_proj = self.gf_proj(gripper_force).unsqueeze(1).expand(-1, ft.shape[1], -1)

        fused = torch.cat([v, t, ft_proj, g_proj, gf_proj], dim=-1)
        projected = self.projection(fused)

        if isinstance(lengths, torch.Tensor):
            length_list = lengths.detach().cpu().tolist()
        else:
            length_list = list(int(length) for length in lengths)

        packed = pack_padded_sequence(
            projected,
            length_list,
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.lstm(packed)

        if self.bidirectional:
            last = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            last = hidden[-1]
        return self.classifier(last)
