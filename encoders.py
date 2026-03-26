"""
CLIP and T3 encoders for RGB and tactile modalities.

CLIPRGBEncoder: CLIP ViT-L/14 (LAION-2B) for RGB, outputs 768-d per image.
T3TactileEncoder: T3 large (304M) encoder + trunk for tactile.
"""

import os
from functools import partial
from typing import Optional

import torch
import torch.nn as nn

# T3 model: "large" (304M) is best; encoder config inferred from checkpoint
T3_MODEL = "large"
T3_BASE_CONFIG = {
    "patch_size": 16,
    "embed_dim": 1024,  # t3_large; overridden by checkpoint inference
    "depth": 3,
    "num_heads": 16,
    "mlp_ratio": 4.0,
    "trunk_depth": 9,
    "img_size": 224,
    "in_chans": 3,
}


def _get_pretrained_dir() -> str:
    """Default pretrained directory (shared)."""
    default = "/ocean/projects/cis260031p/shared/pretrained"
    return os.environ.get("TEMU_PRETRAINED_DIR", default)


def _torch_load_checkpoint(path: str, map_location="cpu"):
    """Load a checkpoint file; prefer ``weights_only=True`` on PyTorch 2.x."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location)


# ---------------------------------------------------------------------------
# CLIP RGB Encoder
# ---------------------------------------------------------------------------


class CLIPRGBEncoder(nn.Module):
    """
    CLIP ViT-L/14 (LAION-2B) image encoder for RGB. Outputs 768-d per image.
    Uses open_clip; freezes by default.
    """

    EMB_DIM = 768

    def __init__(self, freeze: bool = True, pretrained: str = "laion2b_s32b_b82k"):
        super().__init__()
        try:
            import open_clip
        except ImportError:
            raise ImportError("Install open_clip: pip install open_clip_torch")

        # Prefer create_model: same weights as create_model_and_transforms, no unused preprocess objects.
        if hasattr(open_clip, "create_model"):
            model = open_clip.create_model("ViT-L-14", pretrained=pretrained)
        else:
            model, _, _ = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained=pretrained
            )
        self.visual = model.visual
        if freeze:
            for p in self.visual.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) — RGB images after CLIP preprocessing (224×224).
        Returns:
            (B, EMB_DIM) image features — ViT-L/14 uses 768-D (must match ``GraspStabilityLSTM_CLIP_T3.CLIP_EMB``).
        """
        out = self.visual(x)
        assert out.shape[-1] == self.EMB_DIM, (
            f"CLIP visual dim {out.shape[-1]} != EMB_DIM {self.EMB_DIM}; "
            "update CLIP_EMB / EMB_DIM if using a different backbone."
        )
        return out

    def train(self, mode: bool = True):
        super().train(mode)
        if any(not p.requires_grad for p in self.visual.parameters()):
            self.visual.eval()
        return self


def get_clip_preprocess():
    """Return CLIP-style preprocessing aligned with OpenCLIP ViT-L/14 (no model download)."""
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    return transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])


# ---------------------------------------------------------------------------
# T3 Tactile Encoder (standalone using timm)
# ---------------------------------------------------------------------------


class T3TactileEncoder(nn.Module):
    """
    T3 large (304M) encoder + trunk for tactile (GelSight).
    Uses timm for ViT components; loads pretrained gs_black + trunk from HF.
    Config inferred from checkpoint.
    """

    def __init__(
        self,
        pretrained_dir: Optional[str] = None,
        encoder_domain: str = "gs_black",
        freeze: bool = True,
    ):
        super().__init__()
        try:
            import timm.models.vision_transformer as timm_vit
        except ImportError:
            raise ImportError("Install timm: pip install timm")

        pretrained_dir = pretrained_dir or _get_pretrained_dir()
        t3_subdir = f"models/{T3_MODEL}"
        encoder_path = os.path.join(
            pretrained_dir, t3_subdir, "encoders", f"{encoder_domain}.pth"
        )
        trunk_path = os.path.join(pretrained_dir, t3_subdir, "trunk.pth")

        # Try alternate path (pretrained/t3_large directly)
        if not os.path.exists(encoder_path):
            alt = os.path.join(pretrained_dir, T3_MODEL, "encoders", f"{encoder_domain}.pth")
            if os.path.exists(alt):
                encoder_path = alt
        if not os.path.exists(trunk_path):
            alt = os.path.join(pretrained_dir, T3_MODEL, "trunk.pth")
            if os.path.exists(alt):
                trunk_path = alt

        # Infer config from encoder checkpoint
        enc_ckpt = None
        if os.path.exists(encoder_path):
            enc_ckpt = _torch_load_checkpoint(encoder_path, map_location="cpu")
            embed_dim = enc_ckpt["patch_embed.proj.weight"].shape[0]
            encoder_depth = max(
                (int(k.split(".")[1]) for k in enc_ckpt if k.startswith("blocks.")),
                default=2,
            ) + 1
            num_heads = max(1, embed_dim // 64)
        else:
            fallback = "gs_tag" if encoder_domain == "gs_black" else "gs_black"
            fallback_path = encoder_path.replace(encoder_domain, fallback)
            if os.path.exists(fallback_path):
                print(f"[T3] {encoder_domain} not found, using {fallback}")
                encoder_path = fallback_path
                enc_ckpt = _torch_load_checkpoint(encoder_path, map_location="cpu")
                embed_dim = enc_ckpt["patch_embed.proj.weight"].shape[0]
                encoder_depth = max(
                    (int(k.split(".")[1]) for k in enc_ckpt if k.startswith("blocks.")),
                    default=2,
                ) + 1
                num_heads = max(1, embed_dim // 64)
            else:
                embed_dim = T3_BASE_CONFIG["embed_dim"]
                encoder_depth = T3_BASE_CONFIG["depth"]
                num_heads = T3_BASE_CONFIG["num_heads"]

        cfg = {
            **T3_BASE_CONFIG,
            "embed_dim": embed_dim,
            "depth": encoder_depth,
            "num_heads": num_heads,
        }
        self.embed_dim = embed_dim
        self.encoder = _build_t3_encoder(cfg, timm_vit)
        self.trunk = _build_t3_trunk({**cfg, "trunk_depth": 6}, timm_vit)  # temp, updated below

        if enc_ckpt is not None:
            _load_t3_encoder(self.encoder, enc_ckpt, cfg)
        else:
            print(f"[T3] No encoder weights at {encoder_path}, using random init")

        if os.path.exists(trunk_path):
            trunk_ckpt = _torch_load_checkpoint(trunk_path, map_location="cpu")
            trunk_depth = max(
                (int(k.split(".")[1]) for k in trunk_ckpt if k.startswith("blocks.")),
                default=5,
            ) + 1
            cfg["trunk_depth"] = trunk_depth
            self.trunk = _build_t3_trunk(cfg, timm_vit)
            self.trunk.load_state_dict(trunk_ckpt, strict=True)
        else:
            print(f"[T3] No trunk weights at {trunk_path}, using random init")

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.trunk.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) — tactile images, ImageNet normalize, 224×224.
        Returns:
            (B, embed_dim) tactile features (``embed_dim`` is set from the checkpoint; T3 large is typically 1024).
        """
        enc = self.encoder(x)  # (B, 1+N, D)
        out = self.trunk(enc)  # (B, D)
        return out

    def train(self, mode: bool = True):
        super().train(mode)
        if any(not p.requires_grad for p in self.parameters()):
            self.encoder.eval()
            self.trunk.eval()
        return self


def _build_t3_encoder(cfg: dict, timm_vit):
    """Build T3 ViT encoder (no head, no final norm)."""
    embed_dim = cfg["embed_dim"]
    depth = cfg["depth"]
    num_heads = cfg["num_heads"]
    mlp_ratio = cfg["mlp_ratio"]
    patch_size = cfg["patch_size"]
    img_size = cfg["img_size"]
    in_chans = cfg["in_chans"]

    patch_embed = timm_vit.PatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
    )
    num_patches = patch_embed.num_patches
    cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
    pos_drop = nn.Dropout(p=0.0)
    norm_layer = partial(nn.LayerNorm, eps=1e-6)
    blocks = nn.ModuleList([
        timm_vit.Block(
            embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer
        )
        for _ in range(depth)
    ])

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = patch_embed
            self.cls_token = cls_token
            self.pos_embed = pos_embed
            self.pos_drop = pos_drop
            self.blocks = blocks

        def forward(self, x):
            B = x.shape[0]
            x = self.patch_embed(x)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)
            for blk in self.blocks:
                x = blk(x)
            return x

    return Encoder()


def _build_t3_trunk(cfg: dict, timm_vit):
    """Build T3 transformer trunk with cls pooling."""
    embed_dim = cfg["embed_dim"]
    depth = cfg["trunk_depth"]
    num_heads = cfg["num_heads"]
    mlp_ratio = cfg["mlp_ratio"]
    norm_layer = partial(nn.LayerNorm, eps=1e-6)

    blocks = nn.ModuleList([
        timm_vit.Block(
            embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer
        )
        for _ in range(depth)
    ])
    norm = norm_layer(embed_dim)

    class Trunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = blocks
            self.norm = norm

        def forward(self, x):
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            return x[:, 0]  # cls token

    return Trunk()


def _load_t3_encoder(encoder, path_or_ckpt, cfg: dict):
    """Load T3 encoder with optional pos_embed interpolation."""
    if isinstance(path_or_ckpt, dict):
        ckpt = path_or_ckpt
    elif os.path.exists(path_or_ckpt):
        ckpt = _torch_load_checkpoint(path_or_ckpt, map_location="cpu")
    else:
        return
    patch_embed = encoder.patch_embed
    num_patches = patch_embed.num_patches
    if "pos_embed" in ckpt:
        pos_embed = ckpt["pos_embed"]
        emb_dim = pos_embed.shape[-1]
        num_extra = pos_embed.shape[-2] - int((pos_embed.shape[-2] - 1) ** 0.5) ** 2
        if num_extra <= 0:
            num_extra = 1
        orig_size = int((pos_embed.shape[-2] - num_extra) ** 0.5)
        new_size = int(num_patches ** 0.5)
        if orig_size != new_size:
            extra = ckpt["pos_embed"][:, :num_extra]
            pos_tokens = ckpt["pos_embed"][:, num_extra:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, emb_dim
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            ckpt["pos_embed"] = torch.cat((extra, pos_tokens), dim=1)
    incomp = encoder.load_state_dict(ckpt, strict=False)
    if incomp.missing_keys or incomp.unexpected_keys:
        print(
            f"[T3] encoder load_state_dict (strict=False): "
            f"{len(incomp.missing_keys)} missing, {len(incomp.unexpected_keys)} unexpected keys"
        )
        if incomp.missing_keys:
            print(f"[T3]   missing (first 5): {incomp.missing_keys[:5]}")
        if incomp.unexpected_keys:
            print(f"[T3]   unexpected (first 5): {incomp.unexpected_keys[:5]}")
