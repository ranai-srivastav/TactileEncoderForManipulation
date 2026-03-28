from __future__ import annotations

from itertools import chain
from typing import Iterable, Sequence

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

ALL_MODALITIES = ("V", "T", "FT", "G", "GF")


class GraspStabilityLSTMVarLen(nn.Module):
    """
    Checkpoint-compatible variable-length ResNet50 baseline used by the
    variable-length PoseIt runs.
    """

    RESNET_EMB = 2048

    V_EMB, T_EMB = 512, 512
    FT_EMB, G_EMB, GF_EMB = 128, 64, 32
    PRE_LSTM_DIM = V_EMB + T_EMB + FT_EMB + G_EMB + GF_EMB

    def __init__(
        self,
        frames_per_sec: int = 1,
        ft_dim: int = 6,
        gripper_dim: int = 2,
        hidden_dim: int = 256,
        lstm_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
        freeze_resnet: bool = True,
        modalities: Iterable[str] | None = None,
        resnet_weights=ResNet50_Weights.DEFAULT,
    ) -> None:
        super().__init__()
        self.frames_per_sec = frames_per_sec
        self.ft_dim = ft_dim
        self.gripper_dim = gripper_dim
        self.modalities = set(modalities or ALL_MODALITIES)
        self.bidirectional = bidirectional

        self.rgb_encoder = resnet50(weights=resnet_weights)
        self.rgb_encoder.fc = nn.Identity()  # type: ignore[assignment]
        self.tactile_encoder = resnet50(weights=resnet_weights)
        self.tactile_encoder.fc = nn.Identity()  # type: ignore[assignment]

        if freeze_resnet:
            for param in chain(self.rgb_encoder.parameters(), self.tactile_encoder.parameters()):
                param.requires_grad = False

        resnet_out = frames_per_sec * self.RESNET_EMB
        self.v_proj = nn.Linear(resnet_out, self.V_EMB)
        self.t_proj = nn.Linear(resnet_out, self.T_EMB)
        self.ft_proj = nn.Linear(ft_dim, self.FT_EMB)
        self.g_proj = nn.Linear(gripper_dim, self.G_EMB)
        self.gf_proj = nn.Linear(1, self.GF_EMB)

        self.projection = nn.Sequential(
            nn.Linear(self.PRE_LSTM_DIM, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

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

    def train(self, mode: bool = True):
        super().train(mode)
        self.rgb_encoder.eval()
        self.tactile_encoder.eval()
        return self

    def forward(
        self,
        tactile: torch.Tensor,
        rgb: torch.Tensor,
        ft: torch.Tensor,
        gripper: torch.Tensor,
        gripper_force: torch.Tensor,
        lengths: Sequence[int] | torch.Tensor,
    ) -> torch.Tensor:
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

        batch_size, steps, frames_per_step = tactile.shape[:3]
        flat_steps = steps * frames_per_step

        tactile_emb = self.tactile_encoder(
            tactile.reshape(batch_size * flat_steps, *tactile.shape[3:])
        ).reshape(batch_size, steps, frames_per_step * self.RESNET_EMB)

        rgb_emb = self.rgb_encoder(
            rgb.reshape(batch_size * flat_steps, *rgb.shape[3:])
        ).reshape(batch_size, steps, frames_per_step * self.RESNET_EMB)

        v_emb = self.v_proj(rgb_emb)
        t_emb = self.t_proj(tactile_emb)
        ft_emb = self.ft_proj(ft)
        g_emb = self.g_proj(gripper)
        gf = gripper_force.unsqueeze(1).expand(batch_size, steps, 1)
        gf_emb = self.gf_proj(gf)

        fused = torch.cat([v_emb, t_emb, ft_emb, g_emb, gf_emb], dim=-1)
        projected = self.projection(fused)

        if isinstance(lengths, torch.Tensor):
            length_tensor = lengths.detach().cpu()
        else:
            length_tensor = torch.tensor(list(lengths), dtype=torch.long)

        packed = nn.utils.rnn.pack_padded_sequence(
            projected,
            length_tensor,
            batch_first=True,
            enforce_sorted=False,
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        idx = (length_tensor - 1).to(projected.device)
        if self.bidirectional:
            hidden = self.lstm.hidden_size
            last_fwd = lstm_out[torch.arange(batch_size, device=lstm_out.device), idx, :hidden]
            last_bwd = lstm_out[:, 0, hidden:]
            last = torch.cat([last_fwd, last_bwd], dim=-1)
        else:
            last = lstm_out[torch.arange(batch_size, device=lstm_out.device), idx, :]
        return self.classifier(last)
