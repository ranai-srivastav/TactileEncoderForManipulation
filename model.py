from itertools import chain
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from encoders import CLIPRGBEncoder, T3TactileEncoder


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

    # Option B: per-modality projection for balanced fusion
    V_EMB, T_EMB = 512, 512
    FT_EMB, G_EMB, GF_EMB = 128, 64, 32
    PRE_LSTM_DIM = V_EMB + T_EMB + FT_EMB + G_EMB + GF_EMB  # 1248

    def __init__(
        self,
        frames_per_sec: int = 1,   # F1 — image frames sampled per second
        ft_dim: int = 6,           # FT_DIM = F2 * 6
        gripper_dim: int = 2,      # GR_DIM = F2 * 2
        hidden_dim: int = 256,
        lstm_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
        freeze_resnet: bool = True,
        modalities=None,           # collection of {'V','T','FT','G','GF'}; None = all
        use_ogm: bool = False
    ):
        super().__init__()
        self.use_ogm = use_ogm
        self.frames_per_sec = frames_per_sec
        self.ft_dim         = ft_dim
        self.gripper_dim    = gripper_dim
        self.modalities     = set(modalities or ['V', 'T', 'FT', 'G', 'GF'])
        self.bidirectional  = bidirectional

        # --- vision encoders (ResNet50, FC stripped → 2048-d) ---
        self.rgb_encoder        = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.rgb_encoder.fc     = nn.Identity()  # type: ignore[assignment]
        self.tactile_encoder    = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.tactile_encoder.fc = nn.Identity()  # type: ignore[assignment]

        if freeze_resnet:
            for p in chain(self.rgb_encoder.parameters(),
                           self.tactile_encoder.parameters()):
                p.requires_grad = False

        # --- per-modality projectors (Option B: balanced fusion) ---
        resnet_out = frames_per_sec * self.RESNET_EMB  # F1*2048
        self.v_proj   = nn.Linear(resnet_out, self.V_EMB)    # 2048 → 512
        self.t_proj   = nn.Linear(resnet_out, self.T_EMB)    # 2048 → 512
        self.ft_proj  = nn.Linear(ft_dim, self.FT_EMB)       # 6 → 128
        self.g_proj   = nn.Linear(gripper_dim, self.G_EMB)   # 2 → 64
        self.gf_proj  = nn.Linear(1, self.GF_EMB)            # 1 → 32

        # --- fusion projection (1248 → hidden_dim) ---
        self.projection = nn.Sequential(
            nn.Linear(self.PRE_LSTM_DIM, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # --- per-modality heads for OGM confidence score monitoring ---
        tac_in = frames_per_sec * self.RESNET_EMB
        rgb_in = frames_per_sec * self.RESNET_EMB
        prop_in = ft_dim + gripper_dim + 1
        
        self.proj_tac = nn.Linear(tac_in, hidden_dim)
        self.proj_rgb = nn.Linear(rgb_in, hidden_dim)
        self.proj_prop = nn.Linear(prop_in, hidden_dim)
        
        self.head_tac = nn.Linear(hidden_dim, 1)
        self.head_rgb = nn.Linear(hidden_dim, 1)
        self.head_prop = nn.Linear(hidden_dim, 1)

        # --- 2-layer LSTM/GRU (bidirectional or unidirectional) ---
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # --- classifier (hidden_dim * 2 if bidirectional else hidden_dim) ---
        classifier_in = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def train(self, mode=True):
        super().train(mode)
        self.rgb_encoder.eval()
        self.tactile_encoder.eval()
        return self

    def forward(self, tactile, rgb, ft, gripper, gripper_force, lengths):
        """
        Args:
            tactile:       (B, T, F1, 3, H, W)
            rgb:           (B, T, F1, 3, H, W)
            ft:            (B, T, FT_DIM)
            gripper:       (B, T, GR_DIM)
            gripper_force: (B, 1)
            lengths:       (B,) or list — actual sequence length per sample (for pack_padded_sequence)

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
        
        # --- per-modality confidence scores for OGM diagnostic ---
        prop = torch.cat([ft, gripper, gf], dim=-1)
        
        if self.use_ogm:            
            logit_tac = self.head_tac(
                torch.relu(self.proj_tac(tac_emb)).mean(dim=1)
            ) if 'T' in self.modalities else None
            logit_rgb = self.head_rgb(
                torch.relu(self.proj_rgb(rgb_emb)).mean(dim=1)
            ) if 'V' in self.modalities else None
            logit_prop = self.head_prop(
                torch.relu(self.proj_prop(prop)).mean(dim=1)
            ) if any(m in self.modalities for m in ['FT', 'G', 'GF']) else None
        else:
            with torch.no_grad():
                logit_tac = self.head_tac(
                    torch.relu(self.proj_tac(tac_emb)).mean(dim=1)
                ) if 'T' in self.modalities else None
                logit_rgb = self.head_rgb(
                    torch.relu(self.proj_rgb(rgb_emb)).mean(dim=1)
                ) if 'V' in self.modalities else None
                logit_prop = self.head_prop(
                    torch.relu(self.proj_prop(prop)).mean(dim=1)
                ) if any(m in self.modalities for m in ['FT', 'G', 'GF']) else None
        # --- per-modality projection (Option B: balanced fusion) ---
        v_emb  = self.v_proj(rgb_emb)                              # (B, T, 512)
        t_emb  = self.t_proj(tac_emb)                             # (B, T, 512)
        ft_emb = self.ft_proj(ft)                                 # (B, T, 128)
        g_emb  = self.g_proj(gripper)                             # (B, T, 64)
        gf     = gripper_force.unsqueeze(1).expand(B, T, 1)
        gf_emb = self.gf_proj(gf)                                 # (B, T, 32)

        # Zero out disabled modalities (after projection)
        if 'V'  not in self.modalities: v_emb  = v_emb  * 0.0
        if 'T'  not in self.modalities: t_emb  = t_emb  * 0.0
        if 'FT' not in self.modalities: ft_emb = ft_emb * 0.0
        if 'G'  not in self.modalities: g_emb  = g_emb  * 0.0
        if 'GF' not in self.modalities: gf_emb = gf_emb * 0.0

        # --- fuse projected modalities, then project to hidden_dim ---
        fused     = torch.cat([v_emb, t_emb, ft_emb, g_emb, gf_emb], dim=-1)  # (B, T, 1248)
        projected = self.projection(fused)                                       # (B, T, hidden_dim)

        # --- GRU over T seconds with pack_padded_sequence (handles variable length) ---
        lengths_cpu = lengths.cpu() if isinstance(lengths, torch.Tensor) else torch.tensor(lengths, dtype=torch.long)
        packed = nn.utils.rnn.pack_padded_sequence(
            projected, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)  # (B, T_max, hidden*2)

        # Index last valid timestep per sequence
        idx = (lengths_cpu - 1).to(projected.device)
        if self.bidirectional:
            h = self.lstm.hidden_size
            last_fwd = lstm_out[torch.arange(B, device=lstm_out.device), idx, :h]
            last_bwd = lstm_out[:, 0, h:]
            last = torch.cat([last_fwd, last_bwd], dim=-1)
        else:
            last = lstm_out[torch.arange(B, device=lstm_out.device), idx, :]  # (B, hidden_dim)
            
        return self.classifier(last), logit_tac, logit_rgb, logit_prop
    
    def apply_ogm(self, k_tac, k_rgb, k_prop):
        """
        Scale gradients of each modality's projection head by k^u.
        Called after loss.backward(), before optimizer.step().
        """
        if k_tac != 1.0:
            for p in self.proj_tac.parameters():
                if p.grad is not None:
                    p.grad *= k_tac
            for p in self.head_tac.parameters():
                if p.grad is not None:
                    p.grad *= k_tac

        if k_rgb != 1.0:
            for p in self.proj_rgb.parameters():
                if p.grad is not None:
                    p.grad *= k_rgb
            for p in self.head_rgb.parameters():
                if p.grad is not None:
                    p.grad *= k_rgb

        if k_prop != 1.0:
            for p in self.proj_prop.parameters():
                if p.grad is not None:
                    p.grad *= k_prop
            for p in self.head_prop.parameters():
                if p.grad is not None:
                    p.grad *= k_prop

class GraspStabilityLSTM_CLIP_T3(nn.Module):
    """
    Same as GraspStabilityLSTM but uses CLIP (ViT-L/14) for RGB and T3 large for tactile.
    CLIP outputs 768-d per image (``CLIP_EMB``). Tactile width is ``tactile_encoder.embed_dim``
    (from the T3 checkpoint; large is typically 1024-d).
    """

    CLIP_EMB = 768
    V_EMB, T_EMB = 512, 512
    FT_EMB, G_EMB, GF_EMB = 128, 64, 32
    PRE_LSTM_DIM = V_EMB + T_EMB + FT_EMB + G_EMB + GF_EMB  # 1248

    def __init__(
        self,
        frames_per_sec: int = 1,
        ft_dim: int = 6,
        gripper_dim: int = 2,
        hidden_dim: int = 256,
        lstm_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
        freeze_encoders: bool = True,
        modalities=None,
        pretrained_dir: Optional[str] = None,
        t3_encoder_domain: str = "gs_black",
    ):
        super().__init__()
        self.frames_per_sec = frames_per_sec
        self.ft_dim = ft_dim
        self.gripper_dim = gripper_dim
        self.modalities = set(modalities or ['V', 'T', 'FT', 'G', 'GF'])
        self.bidirectional = bidirectional

        self.rgb_encoder = CLIPRGBEncoder(freeze=freeze_encoders)
        self.tactile_encoder = T3TactileEncoder(
            pretrained_dir=pretrained_dir,
            encoder_domain=t3_encoder_domain,
            freeze=freeze_encoders,
        )
        t3_out = self.tactile_encoder.embed_dim

        v_in = frames_per_sec * self.CLIP_EMB
        t_in = frames_per_sec * t3_out
        self.v_proj = nn.Linear(v_in, self.V_EMB)
        self.t_proj = nn.Linear(t_in, self.T_EMB)
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

        self.lstm = nn.LSTM(
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

    def train(self, mode=True):
        super().train(mode)
        self.rgb_encoder.train(mode)
        self.tactile_encoder.train(mode)
        return self

    def forward(self, tactile, rgb, ft, gripper, gripper_force, lengths):
        if 'T' not in self.modalities:
            tactile = tactile * 0.0
        if 'V' not in self.modalities:
            rgb = rgb * 0.0
        if 'FT' not in self.modalities:
            ft = ft * 0.0
        if 'G' not in self.modalities:
            gripper = gripper * 0.0
        if 'GF' not in self.modalities:
            gripper_force = gripper_force * 0.0

        B, T, F1 = tactile.shape[:3]
        S = T * F1

        tac_emb = self.tactile_encoder(
            tactile.reshape(B * S, *tactile.shape[3:])
        ).reshape(B, T, F1 * self.tactile_encoder.embed_dim)
        rgb_emb = self.rgb_encoder(
            rgb.reshape(B * S, *rgb.shape[3:])
        ).reshape(B, T, F1 * self.CLIP_EMB)

        v_emb = self.v_proj(rgb_emb)
        t_emb = self.t_proj(tac_emb)
        ft_emb = self.ft_proj(ft)
        g_emb = self.g_proj(gripper)
        gf = gripper_force.unsqueeze(1).expand(B, T, 1)
        gf_emb = self.gf_proj(gf)

        if 'V' not in self.modalities:
            v_emb = v_emb * 0.0
        if 'T' not in self.modalities:
            t_emb = t_emb * 0.0
        if 'FT' not in self.modalities:
            ft_emb = ft_emb * 0.0
        if 'G' not in self.modalities:
            g_emb = g_emb * 0.0
        if 'GF' not in self.modalities:
            gf_emb = gf_emb * 0.0

        fused = torch.cat([v_emb, t_emb, ft_emb, g_emb, gf_emb], dim=-1)
        projected = self.projection(fused)

        lengths_cpu = lengths.cpu() if isinstance(lengths, torch.Tensor) else torch.tensor(lengths, dtype=torch.long)
        packed = nn.utils.rnn.pack_padded_sequence(
            projected, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        idx = (lengths_cpu - 1).to(projected.device)
        if self.bidirectional:
            h = self.lstm.hidden_size
            last_fwd = lstm_out[torch.arange(B, device=lstm_out.device), idx, :h]
            last_bwd = lstm_out[:, 0, h:]
            last = torch.cat([last_fwd, last_bwd], dim=-1)
        else:
            last = lstm_out[torch.arange(B, device=lstm_out.device), idx, :]
        return self.classifier(last)
