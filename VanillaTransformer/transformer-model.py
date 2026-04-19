import os
import sys

import timm
import torch
import torch.nn as nn

# repo root for encoders.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from encoders import T3TactileEncoder


class VanillaTransformer(nn.Module):
    """
    Minimal encoder-only multimodal model.

    Forward returns encoded modality tensors. Nothing is fused yet.
    """

    def __init__(
        self,
        frames_per_sec: int = 1,
        ft_dim: int = 6,
        gripper_dim: int = 2,
        max_timesteps: int = 20,
        hidden_dim: int = 768,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        modalities=None,
        max_visual_frames: int = 8,
    ):
        super().__init__()
        self.modalities = set(modalities or ["V", "T", "FT", "G", "GF"])
        self.hidden_dim = hidden_dim

        self.rgb_encoder = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.tactile_encoder = T3TactileEncoder(freeze=True)

        self.tactile_proj = nn.Linear(self.tactile_encoder.embed_dim, hidden_dim)
        self.ft_proj = nn.Sequential(nn.Linear(ft_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.gripper_proj = nn.Sequential(nn.Linear(gripper_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.gf_proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.LayerNorm(hidden_dim))

    def encode_rgb(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, f, c, h, w = rgb.shape
        x = rgb.reshape(b * t * f, c, h, w)
        x = self.rgb_encoder(x)
        x = x.reshape(b, t, f, -1)
        return x, x.mean(dim=2)

    def encode_tactile(self, tactile: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, f, c, h, w = tactile.shape
        x = tactile.reshape(b * t * f, c, h, w)
        x = self.tactile_encoder(x)
        x = self.tactile_proj(x)
        x = x.reshape(b, t, f, -1)
        return x, x.mean(dim=2)

    def forward(
        self,
        tactile: torch.Tensor,
        rgb: torch.Tensor,
        ft: torch.Tensor,
        gripper: torch.Tensor,
        gripper_force: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out = {}

        if "V" in self.modalities:
            rgb_frames, rgb_seconds = self.encode_rgb(rgb)
            out["rgb_frames"] = rgb_frames
            out["rgb"] = rgb_seconds

        if "T" in self.modalities:
            tactile_frames, tactile_seconds = self.encode_tactile(tactile)
            out["tactile_frames"] = tactile_frames
            out["tactile"] = tactile_seconds

        if "FT" in self.modalities:
            out["ft"] = self.ft_proj(ft)

        if "G" in self.modalities:
            out["gripper"] = self.gripper_proj(gripper)

        if "GF" in self.modalities:
            out["gripper_force"] = self.gf_proj(gripper_force).unsqueeze(1)

        return out


TransformerModel = VanillaTransformer
VanillaTransformerModel = VanillaTransformer
