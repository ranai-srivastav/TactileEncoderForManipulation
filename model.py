import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class GraspStabilityLSTM(nn.Module):
    """
    Predicts P(grasp success) from multimodal sensor data.

    Each sub-second frame is encoded independently, then the full
    temporal sequence is processed by a bidirectional LSTM.

    LSTM sequence length = T * frames_per_sec, where T is the number
    of integer-second buckets passed at runtime (not fixed here).

    Expected input shapes from dataloader (per batch element):
        tactile:       (T, F, 3, H, W)
        rgb:           (T, F, 3, H, W)
        ft:            (T, F * 6)          — F/T readings flattened per second
        gripper:       (T, F * 2)          — gripper readings flattened per second
        gripper_force: (1,)                — static force command
    """

    RESNET18_EMB = 512

    def __init__(
        self,
        frames_per_sec: int = 2,
        ft_channels: int = 6,
        gripper_channels: int = 2,
        hidden_dim: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        freeze_resnet: bool = True,
    ):
        super().__init__()
        self.frames_per_sec = frames_per_sec
        self.ft_channels = ft_channels
        self.gripper_channels = gripper_channels

        # --- vision encoders (ResNet18, FC stripped → 512-d) ---
        self.rgb_encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.rgb_encoder.fc = nn.Identity()

        self.tactile_encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.tactile_encoder.fc = nn.Identity()

        if freeze_resnet:
            for p in self.rgb_encoder.parameters():
                p.requires_grad = False
            for p in self.tactile_encoder.parameters():
                p.requires_grad = False

        # --- small encoders for low-dim modalities ---
        self.ft_encoder = nn.Sequential(
            nn.Linear(ft_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        self.gripper_encoder = nn.Sequential(
            nn.Linear(gripper_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        # --- per-timestep fusion: 512 + 512 + 64 + 32 + 1 = 1121 ---
        fusion_dim = self.RESNET18_EMB + self.RESNET18_EMB + 64 + 32 + 1
        self.projection = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --- LSTM ---
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # --- classifier (hidden_dim * 2 because bidirectional) ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def _encode_frames(self, tactile, rgb, ft, gripper, gripper_force):
        """Encode all modalities and fuse into per-frame embeddings.

        Returns:
            projected: (B, S, hidden_dim)  where S = T * frames_per_sec
        """
        B, T, F = tactile.shape[:3]
        S = T * F

        # --- encode every frame through ResNet ---
        rgb_emb = self.rgb_encoder(
            rgb.reshape(B * S, *rgb.shape[3:])
        ).reshape(B, S, self.RESNET18_EMB)

        tac_emb = self.tactile_encoder(
            tactile.reshape(B * S, *tactile.shape[3:])
        ).reshape(B, S, self.RESNET18_EMB)

        # --- un-flatten per-second readings to per-frame ---
        ft_per_frame = ft.reshape(B, T, self.frames_per_sec, self.ft_channels)
        ft_emb = self.ft_encoder(
            ft_per_frame.reshape(B, S, self.ft_channels)
        )

        gr_per_frame = gripper.reshape(B, T, self.frames_per_sec, self.gripper_channels)
        gr_emb = self.gripper_encoder(
            gr_per_frame.reshape(B, S, self.gripper_channels)
        )

        # --- broadcast static force to every frame ---
        force = gripper_force.unsqueeze(1).expand(B, S, 1)

        # --- fuse + project ---
        fused = torch.cat([rgb_emb, tac_emb, ft_emb, gr_emb, force], dim=-1)
        return self.projection(fused)

    def forward(self, tactile, rgb, ft, gripper, gripper_force):
        """
        Args:
            tactile:       (B, T, F, 3, H, W)
            rgb:           (B, T, F, 3, H, W)
            ft:            (B, T, F*6)
            gripper:       (B, T, F*2)
            gripper_force: (B, 1)

        Returns:
            (B, 1) raw logits.  Use BCEWithLogitsLoss for training or
            call .sigmoid() at inference for P(success).
        """
        projected = self._encode_frames(
            tactile, rgb, ft, gripper, gripper_force
        )  # (B, S, hidden_dim)

        lstm_out, _ = self.lstm(projected)  # (B, S, hidden_dim*2)
        return self.classifier(lstm_out[:, -1, :])  # (B, 1)
