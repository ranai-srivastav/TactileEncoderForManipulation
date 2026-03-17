import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class GraspStabilityLSTM(nn.Module):
    """
    Predicts P(grasp success) from multimodal sensor data.

    Each second of data is encoded independently (F1 image frames flattened,
    FT/gripper readings concatenated flat), then the full temporal sequence of
    L seconds is processed by a 2-layer bidirectional LSTM.

    Modalities can be selectively disabled at construction time via the
    `modalities` argument.  Disabled modalities are zeroed out before any
    computation, so gradient flow and model shape are unaffected.

    Modality keys:
        'V'  — RGB camera frames
        'T'  — GelSight tactile frames
        'FT' — Force-torque readings
        'G'  — Gripper state readings
        'GF' — Gripper force command (scalar metadata)

    Expected input shapes (per batch):
        tactile:       (B, T, F1, 3, H, W)
        rgb:           (B, T, F1, 3, H, W)
        ft:            (B, T, FT_DIM)   — F2*6 readings flattened per second
        gripper:       (B, T, GR_DIM)   — F2*2 readings flattened per second
        gripper_force: (B, 1)           — static force command

    Returns:
        (B, 1) raw logits.  Use BCEWithLogitsLoss for training or
        call .sigmoid() at inference for P(success).
    """

    RESNET_EMB = 2048  # ResNet50 penultimate-layer width

    def __init__(
        self,
        frames_per_sec: int = 1,   # F1 — image frames sampled per second
        ft_dim: int = 6,           # FT_DIM = F2 * 6
        gripper_dim: int = 2,      # GR_DIM = F2 * 2
        hidden_dim: int = 256,
        lstm_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
        freeze_resnet_rgb: bool = True,      # freeze RGB ResNet50
        freeze_resnet_tactile: bool = True,  # freeze tactile ResNet50
        freeze_projection: bool = False,     # freeze per-second projection MLP
        freeze_gru: bool = False,            # freeze GRU
        freeze_classifier: bool = False,     # freeze final classifier MLP
        n_outputs: int = 1,        # 1 = BCEWithLogitsLoss head; 2 = CrossEntropyLoss head
        modalities=None,           # collection of {'V','T','FT','G','GF'}; None = all
    ):
        super().__init__()
        self.frames_per_sec = frames_per_sec
        self.ft_dim         = ft_dim
        self.gripper_dim    = gripper_dim
        self.modalities     = set(modalities or ['V', 'T', 'FT', 'G', 'GF'])
        self.bidirectional  = bidirectional
        self.n_outputs      = n_outputs

        # --- vision encoders (ResNet50, FC stripped → 2048-d) ---
        self.rgb_encoder        = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.rgb_encoder.fc     = nn.Identity()  # type: ignore[assignment]
        self.tactile_encoder    = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.tactile_encoder.fc = nn.Identity()  # type: ignore[assignment]

        if freeze_resnet_rgb:
            for p in self.rgb_encoder.parameters():
                p.requires_grad = False
        if freeze_resnet_tactile:
            for p in self.tactile_encoder.parameters():
                p.requires_grad = False

        # --- per-second fusion projection ---
        # (freeze_projection applied after construction)
        # concat: [tac_emb (F1*2048), rgb_emb (F1*2048), ft (FT_DIM), grip (GR_DIM), gf (1)]
        pre_lstm_dim = frames_per_sec * self.RESNET_EMB * 2 + ft_dim + gripper_dim + 1
        self.projection = nn.Sequential(
            nn.Linear(pre_lstm_dim, hidden_dim *2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim *2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --- 2-layer LSTM/GRU (bidirectional or unidirectional) ---
        self.lstm = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        if freeze_projection:
            for p in self.projection.parameters():
                p.requires_grad = False

        if freeze_gru:
            for p in self.lstm.parameters():
                p.requires_grad = False

        # --- classifier (hidden_dim * 2 if bidirectional else hidden_dim) ---
        classifier_in = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_outputs),  # 1 = BCE logit; 2 = CE logits
        )
        if freeze_classifier:
            for p in self.classifier.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        # Keep frozen encoders in eval mode — their BN stats should not drift
        if not any(p.requires_grad for p in self.rgb_encoder.parameters()):
            self.rgb_encoder.eval()
        if not any(p.requires_grad for p in self.tactile_encoder.parameters()):
            self.tactile_encoder.eval()
        return self

    def forward(self, tactile, rgb, ft, gripper, gripper_force):
        """
        Args:
            tactile:       (B, T, F1, 3, H, W)
            rgb:           (B, T, F1, 3, H, W)
            ft:            (B, T, FT_DIM)
            gripper:       (B, T, GR_DIM)
            gripper_force: (B, 1)

        Returns:
            (B, 1) raw logits.
        """
        # --- modality masking (zero-out disabled inputs) ---
        if 'T'  not in self.modalities: tactile       = tactile       * 0.0
        if 'V'  not in self.modalities: rgb           = rgb           * 0.0
        if 'FT' not in self.modalities: ft            = ft            * 0.0
        if 'G'  not in self.modalities: gripper       = gripper       * 0.0
        if 'GF' not in self.modalities: gripper_force = gripper_force * 0.0

        B, T, F1 = tactile.shape[:3]
        S = T * F1  # total image frames across all seconds

        # --- encode all image frames through ResNet50, then flatten per second ---
        tac_emb = self.tactile_encoder(
            tactile.reshape(B * S, *tactile.shape[3:])
        ).reshape(B, T, F1 * self.RESNET_EMB)   # (B, T, F1*2048)

        rgb_emb = self.rgb_encoder(
            rgb.reshape(B * S, *rgb.shape[3:])
        ).reshape(B, T, F1 * self.RESNET_EMB)   # (B, T, F1*2048)

        # --- ft / gripper already flat per second; broadcast static force ---
        gf = gripper_force.unsqueeze(1).expand(B, T, 1)   # (B, T, 1)

        # --- fuse all modalities per second, project to hidden_dim ---
        fused     = torch.cat([tac_emb, rgb_emb, ft, gripper, gf], dim=-1)  # (B, T, pre_lstm_dim)
        projected = self.projection(fused)                                    # (B, T, hidden_dim)

        # --- LSTM over T seconds, classify from final hidden state ---
        lstm_out, _ = self.lstm(projected)
        if self.bidirectional:
            # Forward at T-1: seen 0→T-1; Backward at 0: seen T-1→0
            h = self.lstm.hidden_size
            last = torch.cat([lstm_out[:, -1, :h], lstm_out[:, 0, h:]], dim=-1)
        else:
            last = lstm_out[:, -1, :]   # (B, hidden_dim)
        return self.classifier(last)
