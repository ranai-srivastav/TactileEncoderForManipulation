import os
import sys

import timm
import torch
import torch.nn as nn

# repo root for encoders.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from encoders import T3TactileEncoder


class Adapter(nn.Module):
    def __init__(self, dim: int, adapter_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.down = nn.Linear(dim, adapter_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(adapter_dim, dim)
        self.scale = nn.Parameter(torch.ones(1))
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.drop(self.act(self.down(x)))) * self.scale


class VanillaTransformer(nn.Module):
    """
    Minimal multimodal vanilla transformer classifier.

    Pipeline:
    1. Encode each modality to 768-d tokens
    2. Add global temporal sinusoidal encoding
    3. Add learned modality embedding
    4. Interleave all modality tokens in time order
    5. Prepend a CLS token
    6. Run a standard transformer encoder
    7. Classify from the CLS token
    """

    MODALITY_KEYS = ("V", "T", "FT", "G", "GF")
    MODALITY_NAMES = {
        "V": "rgb",
        "T": "tactile",
        "FT": "ft",
        "G": "gripper",
        "GF": "gripper_force",
    }

    def __init__(
        self,
        frames_per_sec: int = 1,
        ft_dim: int = 6,
        gripper_dim: int = 2,
        max_timesteps: int = 20,
        hidden_dim: int = 768,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        modalities=None,
        max_visual_frames: int = 8,
    ):
        super().__init__()
        self.modalities = set(modalities or self.MODALITY_KEYS)
        self.hidden_dim = hidden_dim

        self.rgb_encoder = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        for p in self.rgb_encoder.parameters():
            p.requires_grad = False

        self.tactile_encoder = T3TactileEncoder(freeze=True)
        self.tactile_proj = nn.Linear(self.tactile_encoder.embed_dim, hidden_dim)
        self.rgb_adapter = Adapter(hidden_dim, adapter_dim=64, dropout=dropout)
        self.tactile_adapter = Adapter(hidden_dim, adapter_dim=64, dropout=dropout)

        self.ft_proj = nn.Sequential(
            nn.Linear(ft_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.gripper_proj = nn.Sequential(
            nn.Linear(gripper_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.gf_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        self.modality_embeddings = nn.ParameterDict({
            key: nn.Parameter(torch.zeros(1, 1, hidden_dim)) for key in self.MODALITY_KEYS
        })
        for param in self.modality_embeddings.values():
            nn.init.trunc_normal_(param, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        for layer in self.transformer.layers:
            for p in layer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def _sinusoidal_from_positions(
        self,
        positions: torch.Tensor,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        positions = positions.to(device=device, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0, device=device)) / dim)
        )
        pe = torch.zeros(positions.shape[0], dim, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(positions * scale)
        pe[:, 1::2] = torch.cos(positions * scale)
        return pe.to(dtype=dtype)

    def _positions_for_rate(self, timesteps: int, samples_per_second: int, device: torch.device) -> torch.Tensor:
        seconds = torch.arange(timesteps, device=device, dtype=torch.float32).repeat_interleave(samples_per_second)
        offsets = torch.arange(samples_per_second, device=device, dtype=torch.float32) / max(samples_per_second, 1)
        offsets = offsets.repeat(timesteps)
        return seconds + offsets

    def _add_time_and_modality(self, tokens: torch.Tensor, positions: torch.Tensor, modality_key: str) -> torch.Tensor:
        time_pe = self._sinusoidal_from_positions(
            positions=positions,
            dim=tokens.shape[-1],
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return tokens + time_pe.unsqueeze(0) + self.modality_embeddings[modality_key]

    def encode_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        b, t, f, c, h, w = rgb.shape
        x = rgb.reshape(b * t * f, c, h, w).float()
        x = self.rgb_encoder(x)
        x = x + self.rgb_adapter(x)
        return x.reshape(b, t * f, -1)

    def encode_tactile(self, tactile: torch.Tensor) -> torch.Tensor:
        b, t, f, c, h, w = tactile.shape
        x = tactile.reshape(b * t * f, c, h, w).float()
        x = self.tactile_encoder(x)
        x = self.tactile_proj(x)
        x = x + self.tactile_adapter(x)
        return x.reshape(b, t * f, -1)

    def encode_ft(self, ft: torch.Tensor) -> torch.Tensor:
        b, t, fft, d = ft.shape
        assert d == 6
        x = self.ft_proj(ft)
        return x.reshape(b, t * fft, -1)

    def encode_gripper(self, gripper: torch.Tensor) -> torch.Tensor:
        b, t, fgripper, d = gripper.shape
        assert d == 2
        x = self.gripper_proj(gripper)
        return x.reshape(b, t * fgripper, -1)

    def encode_gripper_force(self, gripper_force: torch.Tensor) -> torch.Tensor:
        return self.gf_proj(gripper_force).unsqueeze(1)

    def _token_mask_for_rate(
        self,
        timestep_mask: torch.Tensor | None,
        samples_per_second: int,
    ) -> torch.Tensor | None:
        if timestep_mask is None:
            return None
        return timestep_mask.repeat_interleave(samples_per_second, dim=1)

    def encode_modalities(self, tactile, rgb, ft, gripper, gripper_force=None, timestep_mask=None) -> dict[str, torch.Tensor]:
        outputs = {}
        b, t = tactile.shape[:2]
        device = tactile.device

        if "V" in self.modalities:
            rgb_tokens = self.encode_rgb(rgb)
            rgb_positions = self._positions_for_rate(t, rgb.shape[2], device)
            outputs["rgb_tokens"] = self._add_time_and_modality(rgb_tokens, rgb_positions, "V")
            outputs["rgb_positions"] = rgb_positions
            outputs["rgb_mask"] = self._token_mask_for_rate(timestep_mask, rgb.shape[2])

        if "T" in self.modalities:
            tactile_tokens = self.encode_tactile(tactile)
            tactile_positions = self._positions_for_rate(t, tactile.shape[2], device)
            outputs["tactile_tokens"] = self._add_time_and_modality(tactile_tokens, tactile_positions, "T")
            outputs["tactile_positions"] = tactile_positions
            outputs["tactile_mask"] = self._token_mask_for_rate(timestep_mask, tactile.shape[2])

        if "FT" in self.modalities:
            ft_tokens = self.encode_ft(ft)
            ft_positions = self._positions_for_rate(t, ft.shape[2], device)
            outputs["ft_tokens"] = self._add_time_and_modality(ft_tokens, ft_positions, "FT")
            outputs["ft_positions"] = ft_positions
            outputs["ft_mask"] = self._token_mask_for_rate(timestep_mask, ft.shape[2])

        if "G" in self.modalities:
            gripper_tokens = self.encode_gripper(gripper)
            gripper_positions = self._positions_for_rate(t, gripper.shape[2], device)
            outputs["gripper_tokens"] = self._add_time_and_modality(gripper_tokens, gripper_positions, "G")
            outputs["gripper_positions"] = gripper_positions
            outputs["gripper_mask"] = self._token_mask_for_rate(timestep_mask, gripper.shape[2])

        if "GF" in self.modalities:
            if gripper_force is None:
                raise ValueError("GF modality requires gripper_force input.")
            gf_tokens = self.encode_gripper_force(gripper_force)
            outputs["gripper_force_tokens"] = gf_tokens + self.modality_embeddings["GF"]
            outputs["gripper_force_positions"] = torch.tensor([-1.0], device=device)
            outputs["gripper_force_mask"] = torch.ones(b, 1, device=device, dtype=torch.bool)

        return outputs

    def build_sequence(self, encoded: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        token_keys = ("rgb_tokens", "tactile_tokens", "ft_tokens", "gripper_tokens", "gripper_force_tokens")
        active_tokens = [encoded[k] for k in token_keys if k in encoded]
        if not active_tokens:
            raise ValueError("No active modalities to build a sequence from.")

        batch_size = active_tokens[0].shape[0]
        pieces = []
        positions = []
        offsets = {
            "rgb_tokens": 0.00,
            "tactile_tokens": 0.01,
            "ft_tokens": 0.02,
            "gripper_tokens": 0.03,
            "gripper_force_tokens": 0.00,
        }
        position_keys = {
            "rgb_tokens": "rgb_positions",
            "tactile_tokens": "tactile_positions",
            "ft_tokens": "ft_positions",
            "gripper_tokens": "gripper_positions",
            "gripper_force_tokens": "gripper_force_positions",
        }
        mask_keys = {
            "rgb_tokens": "rgb_mask",
            "tactile_tokens": "tactile_mask",
            "ft_tokens": "ft_mask",
            "gripper_tokens": "gripper_mask",
            "gripper_force_tokens": "gripper_force_mask",
        }
        masks = []
        for key in token_keys:
            if key in encoded:
                tokens = encoded[key]
                pieces.append(tokens)
                positions.append(encoded[position_keys[key]] + offsets[key])
                mask = encoded.get(mask_keys[key])
                if mask is None and any(k.endswith("_mask") for k in encoded):
                    mask = torch.ones(
                        tokens.shape[:2],
                        device=tokens.device,
                        dtype=torch.bool,
                    )
                masks.append(mask)

        sequence = torch.cat(pieces, dim=1)
        all_positions = torch.cat(positions, dim=0)
        order = torch.argsort(all_positions)
        sequence = sequence[:, order]
        cls = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls, sequence], dim=1)

        padding_mask = None
        if any(mask is not None for mask in masks):
            valid_mask = torch.cat(masks, dim=1)[:, order]
            cls_valid = torch.ones(batch_size, 1, device=valid_mask.device, dtype=torch.bool)
            valid_mask = torch.cat([cls_valid, valid_mask], dim=1)
            padding_mask = ~valid_mask

        return sequence, padding_mask

    def forward(
        self,
        tactile,
        rgb,
        ft,
        gripper,
        gripper_force=None,
        timestep_mask=None,
        return_debug: bool = False,
    ):
        encoded = self.encode_modalities(
            tactile,
            rgb,
            ft,
            gripper,
            gripper_force=gripper_force,
            timestep_mask=timestep_mask,
        )
        sequence, padding_mask = self.build_sequence(encoded)
        hidden = self.transformer(sequence, src_key_padding_mask=padding_mask)
        cls = self.norm(hidden[:, 0])
        logits = self.head(cls)

        if return_debug:
            debug = {k: v for k, v in encoded.items() if k.endswith("_tokens")}
            debug["sequence_tokens"] = sequence
            if padding_mask is not None:
                debug["padding_mask"] = padding_mask
            debug["cls_embedding"] = cls
            return logits, debug
        return logits
