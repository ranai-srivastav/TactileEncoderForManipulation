import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class GraspClassifier(nn.Module):
    """
    ResNet-50 extracts per-frame features from RGB images,
    then mean-pool over valid timesteps (masking padding),
    then MLP classifies grasp outcome (pass vs fail).
    """

    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.feat_dim = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Linear(self.feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, imgs, lengths):
        """
        imgs:    (B, T, F1, 3, H, W)  — rgb or tactile, from collate_variable_length
        lengths: (B,)                  — true (unpadded) sequence lengths

        Since F1=1, we squeeze it out.
        """
        B, T, F1, C, H, W = imgs.shape
        # Flatten F1 into T (F1=1 so this is just a squeeze)
        imgs = imgs.view(B, T * F1, C, H, W)
        T_total = T * F1

        # Extract features for all frames at once
        features = self.backbone(imgs.view(B * T_total, C, H, W))  # (B*T_total, 512)
        features = features.view(B, T_total, self.feat_dim)         # (B, T, 512)

        # Masked mean pooling — ignore padded timesteps
        mask = torch.arange(T_total, device=imgs.device).unsqueeze(0) < (lengths * F1).unsqueeze(1)
        mask = mask.unsqueeze(-1).float()                            # (B, T, 1)
        pooled = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, 512)

        return self.mlp(pooled)  # (B, num_classes)