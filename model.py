"""
PoseIt baseline model: frozen ResNet50 for vision/tactile,
per-timestep feature concat, 2-layer bidirectional LSTM, binary classifier.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from dataloader import F1, F2, FT_DIM, GR_DIM

RESNET_EMB = 2048  # ResNet50 backbone output dim (before fc)


class BaselineTactileEncoder(nn.Module):
    def __init__(
        self,
        vision_resnet_emb_size: int = 2048,
        tactile_resnet_emb_size: int = 2048,
        model_emb_size: int = 512,
        hidden_dim: int = 500,
        dropout: float = 0.1,
        modalities=None,
    ):
        super(BaselineTactileEncoder, self).__init__()
        self.vision_resnet_emb_size = vision_resnet_emb_size
        self.tactile_resnet_emb_size = tactile_resnet_emb_size
        self.model_emb_size = model_emb_size
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.modalities = set(modalities or ['V', 'T', 'FT', 'GF', 'G'])

        # ResNet50 has: conv layers -> avgpool -> fc(2048 -> 1000 classes)
        # replace fc with Identity to get 2048-dim feature vectors per image.
        self.vision_encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.vision_encoder.fc = nn.Identity()
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        self.tactile_encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.tactile_encoder.fc = nn.Identity()
        for p in self.tactile_encoder.parameters():
            p.requires_grad = False

        # Projection: per-timestep concat -> model_emb_size
        # Concat dim = F1*vision_emb + F1*tactile_emb + FT_DIM + GR_DIM + 1
        pre_lstm_dim = (
            F1 * vision_resnet_emb_size
            + F1 * tactile_resnet_emb_size
            + FT_DIM
            + GR_DIM
            + 1
        )
        self.proj = nn.Linear(pre_lstm_dim, model_emb_size)

        # 2-layer bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=model_emb_size,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Classifier: last LSTM hidden -> 2 classes (stable / not stable)
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden_dim, 2)

    def forward(self, tac, rgb, ft, grip, gf):
        """
        tac:  (B, T, F1, 3, H, W)
        rgb:  (B, T, F1, 3, H, W)
        ft:   (B, T, FT_DIM)
        grip: (B, T, GR_DIM)
        gf:   (B, 1)
        """
        # Modality masking: zero out disabled inputs so they don't contribute to the concat
        if 'T' not in self.modalities:
            tac = tac * 0.0
        if 'V' not in self.modalities:
            rgb = rgb * 0.0
        if 'FT' not in self.modalities:
            ft = ft * 0.0
        if 'G' not in self.modalities:
            grip = grip * 0.0
        if 'GF' not in self.modalities:
            gf = gf * 0.0

        B, T, F1_img, C, H, W = tac.shape

        # Per-timestep: encode tactile and rgb, concat with ft, grip, gf
        seq = []
        for t in range(T):
            tac_t = tac[:, t]
            tac_flat = tac_t.reshape(B * F1_img, C, H, W)
            tac_emb = self.tactile_encoder(tac_flat)
            tac_emb = tac_emb.reshape(B, F1_img * RESNET_EMB)

            rgb_t = rgb[:, t]
            rgb_flat = rgb_t.reshape(B * F1_img, C, H, W)
            rgb_emb = self.vision_encoder(rgb_flat)
            rgb_emb = rgb_emb.reshape(B, F1_img * RESNET_EMB)

            ft_t = ft[:, t]
            grip_t = grip[:, t]
            gf_bc = gf.expand(B)

            x_t = torch.cat([tac_emb, rgb_emb, ft_t, grip_t, gf_bc.unsqueeze(1)], dim=1)
            seq.append(x_t)

        x = torch.stack(seq, dim=1)
        x = self.proj(x)
        lstm_out, _ = self.lstm(x)
        last_h = lstm_out[:, -1]
        last_h = self.dropout_layer(last_h)
        logits = self.classifier(last_h)
        return logits
