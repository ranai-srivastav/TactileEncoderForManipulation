"""
MBTGraspStability: Multimodal Bottleneck Transformer for Grasp Stability Prediction

Architecture based on "Attention Bottlenecks for Multimodal Fusion" (Nagrani et al.,
NeurIPS 2021)

Architecture overview

  Modality streams:
    Visual (high-dim, many tokens):
      'T'  — Tactile (GelSight) frames   →  ViT-Base patch tokens
      'V'  — RGB camera frames            →  ViT-Base patch tokens
    Temporal (low-dim, few tokens):
      'FT' — Force-torque readings        →  projected to 768-d, one token per timestep
      'G'  — Gripper state                →  projected to 768-d, one token per timestep
    Static (single value):
      'GF' — Gripper force                →  projected to a single 768-d token

  Processing stages:
    1. Tokenisation
       Visual streams: patch-embed frames with factored spatial + temporal positional
       embeddings (ViViT-style). Each frame reuses the same 196-patch spatial positions
       from ViT, and a separate learned temporal embedding distinguishes frame order.
       Temporal streams: linear projection to 768-d per timestep, with CLS token and
       learned positional embeddings.
       Static stream: linear projection of the scalar to 768-d.

    2. Unimodal processing (layers 0 … Lf-1)
       Each stream processes its own tokens independently to build modality-specific
       representations before fusion begins.
       Visual streams use frozen ViT-Base blocks with optional AdaptFormer adapters.
       Temporal streams use lightweight trainable transformer blocks (4 heads, 2× MLP).

    3. Bottleneck fusion (layers Lf … 11)
       All 5 streams participate in cross-modal fusion via shared bottleneck tokens.

    4. Classification
       Each modality's CLS token is passed through its own linear classifier head.
       The final prediction is the average of all per-modality pre-softmax logits.
       This allows each modality to contribute independently and makes it easy to 
       inspect per-modality contributions.

Input shapes:
    tactile:       (B, T, F1, 3, 224, 224)
    rgb:           (B, T, F1, 3, 224, 224)
    ft:            (B, T, FT_DIM)
    gripper:       (B, T, GR_DIM)
    gripper_force: (B, 1)

Output: (B, num_classes) raw logits — use BCEWithLogitsLoss for binary classification
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class QuickGELU(nn.Module):
    """Fast approximation of GELU used by AdaptFormer."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


def _make_adapter(dim: int, adapter_dim: int):
    """
    Create an AdaptFormer adapter: a trainable down→GELU→up bottleneck that runs
    in parallel with a frozen FFN. Zero-initialised output so the adapter starts
    as a no-op and gradually learns to adjust the frozen representations.

    Returns: (down_proj, up_proj, learnable_scale) or None if adapter_dim <= 0.
    """
    if adapter_dim <= 0:
        return None
    down = nn.Linear(dim, adapter_dim)
    up   = nn.Linear(adapter_dim, dim)
    nn.init.xavier_uniform_(down.weight); nn.init.zeros_(down.bias)
    nn.init.zeros_(up.weight);            nn.init.zeros_(up.bias)
    scale = nn.Parameter(torch.ones(1))
    return down, up, scale


# ─────────────────────────────────────────────────────────────────────────────
# Unimodal transformer block (frozen ViT layer + optional AdaptFormer)
# ─────────────────────────────────────────────────────────────────────────────

class UnimodalBlock(nn.Module):
    """
    Wraps one frozen ViT transformer layer, optionally augmented with a trainable
    AdaptFormer adapter in parallel with the frozen FFN.

    Used for visual streams (Tactile and RGB) during unimodal processing.
    """
    VIT_DIM = 768

    def __init__(self, vit_block, adapter_dim: int = 64):
        super().__init__()
        self.norm1 = vit_block.norm1
        self.attn  = vit_block.attn
        self.norm2 = vit_block.norm2
        self.mlp   = vit_block.mlp

        self.adapter = None
        if adapter_dim > 0:
            res = _make_adapter(self.VIT_DIM, adapter_dim)
            self.down, self.up, self.scale = res
            self.act     = QuickGELU()
            self.dropout = nn.Dropout(0.1)
            self.adapter = True

    def _adapt(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.dropout(self.act(self.down(x)))) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        n = self.norm2(x)
        x = x + self.mlp(n) + (self._adapt(n) if self.adapter else 0)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight transformer block for low-dimensional modalities
# ─────────────────────────────────────────────────────────────────────────────

class LightweightTransformerBlock(nn.Module):
    """
    A small, fully trainable transformer block for 1D modalities
    (force-torque and gripper state).

    Standard pre-norm transformer with 4 attention heads and 2× MLP.
    """
    def __init__(self, dim: int = 768, num_heads: int = 4, mlp_ratio: float = 2.0,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nx = self.norm1(x)
        x = x + self.attn(nx, nx, nx, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Stream parameters container (used by the N-modality fusion block)
# ─────────────────────────────────────────────────────────────────────────────

class StreamParams(nn.Module):
    """
    Holds the transformer layer components for a single modality stream within
    a bottleneck fusion block. Provides a uniform interface for the fusion block
    to process any stream type.

    Two modes:
      - Frozen ViT block (visual streams): extracts norm/attn/mlp components and
        optionally adds an AdaptFormer adapter. Forward uses manual residuals.
      - Lightweight block (non-visual streams): wraps the block directly.
        Already fully trainable, no adapter needed.
    """
    def __init__(self, block, dim: int = 768, adapter_dim: int = 64,
                 is_lightweight: bool = False):
        super().__init__()
        self.is_lightweight = is_lightweight

        if is_lightweight:
            self.block = block
            self.has_adapter = False
        else:
            self.norm1 = block.norm1
            self.attn  = block.attn
            self.norm2 = block.norm2
            self.mlp   = block.mlp
            self.has_adapter = False
            if adapter_dim > 0:
                res = _make_adapter(dim, adapter_dim)
                self.down, self.up, self.scale = res
                self.act     = QuickGELU()
                self.dropout = nn.Dropout(0.1)
                self.has_adapter = True

    def _adapt(self, x):
        return self.up(self.dropout(self.act(self.down(x)))) * self.scale

    def forward_with_bottleneck(self, tokens, bn, Nb):
        """
        Concatenate stream tokens with shared bottleneck tokens, run one
        transformer layer, then split them back apart.

        This implements half of Eq. 8: the stream-specific update. The caller
        is responsible for collecting temporary bottleneck outputs from all
        streams and averaging them (Eq. 9).

        Args:
            tokens: (B, N_stream, D) — this stream's token sequence
            bn:     (B, Nb, D)       — shared bottleneck tokens (same for all streams)
            Nb:     int              — number of bottleneck tokens

        Returns:
            updated_tokens: (B, N_stream, D)
            bn_temp:        (B, Nb, D) — this stream's bottleneck contribution
        """
        if self.is_lightweight:
            cat = torch.cat([tokens, bn], dim=1)
            cat = self.block(cat)
            return cat[:, :-Nb], cat[:, -Nb:]
        else:
            cat = torch.cat([tokens, bn], dim=1)
            cat = cat + self.attn(self.norm1(cat))
            n   = self.norm2(cat)
            cat = cat + self.mlp(n) + (self._adapt(n) if self.has_adapter else 0)
            return cat[:, :-Nb], cat[:, -Nb:]


# ─────────────────────────────────────────────────────────────────────────────
# N-modality bottleneck fusion block
# ─────────────────────────────────────────────────────────────────────────────

class BottleneckFusionBlock(nn.Module):
    """
    One layer of N-modality bottleneck fusion, generalising Eq. 8-9 of the paper
    from 2 modalities to an arbitrary number.

    Within each fusion layer, every modality stream independently attends to the
    shared bottleneck tokens (Eq. 8). Each stream produces a temporary updated
    version of the bottleneck tokens. These are then averaged across all streams
    to produce the new shared bottleneck state (Eq. 9, symmetric update):

        For each stream i ∈ {T, V, FT, G, GF}:
          [z_i^{l+1} | ẑ_bn_i^{l+1}] = Transformer([z_i^l | z_bn^l]; θ_i)
        z_bn^{l+1} = (1/N) Σ_i ẑ_bn_i^{l+1}

    Each stream has its own parameters (θ_i), so modality-specific patterns are
    preserved while cross-modal information flows exclusively through the bottleneck.
    """

    def __init__(self, stream_blocks: dict, num_bottlenecks: int = 4):
        """
        Args:
            stream_blocks: dict mapping modality key → StreamParams module,
                           e.g. {'T': StreamParams(...), 'V': StreamParams(...), ...}
            num_bottlenecks: number of shared bottleneck tokens (B in the paper)
        """
        super().__init__()
        self.streams = nn.ModuleDict(stream_blocks)
        self.Nb = num_bottlenecks

    def forward(self, token_dict: dict, bn: torch.Tensor):
        """
        Args:
            token_dict: {modality_key: (B, N_i, D)} — token sequences per active modality
            bn:         (B, Nb, D) — shared bottleneck tokens from previous layer
        Returns:
            updated_token_dict: same structure as input, with updated token sequences
            bn_new:             (B, Nb, D) — updated shared bottleneck tokens
        """
        bn_accum = torch.zeros_like(bn)
        out_dict = {}
        n_streams = 0

        for key, stream_params in self.streams.items():
            if key not in token_dict:
                continue
            tokens = token_dict[key]
            tokens_new, bn_tmp = stream_params.forward_with_bottleneck(tokens, bn, self.Nb)
            out_dict[key] = tokens_new
            bn_accum = bn_accum + bn_tmp
            n_streams += 1

        bn_new = bn_accum / max(n_streams, 1)
        return out_dict, bn_new


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class MBTGraspStability(nn.Module):
    """
    Fuses all five modalities through attention bottlenecks at multiple layers
    of a 12-layer transformer, following the MBT architecture from Nagrani et al.
    (NeurIPS 2021).

    Constructor args
    ────────────────
    frames_per_sec    int   = 1     F1: image frames captured per second of grasp
    ft_dim            int   = 6     force-torque input dimension (e.g. F2 × 6)
    gripper_dim       int   = 2     gripper state input dimension (e.g. F2 × 2)
    max_timesteps     int   = 20    maximum T for temporal modalities (FT, Gripper).
                                    Determines positional embedding table size.
    num_bottlenecks   int   = 4     B: number of shared fusion tokens. The paper
                                    finds performance robust across 4–256.
    fusion_layer      int   = 8     Lf: layers 0…Lf-1 are unimodal, layers Lf…11
                                    perform bottleneck fusion. Paper optimal: Lf=8.
    max_visual_frames int   = 8     frames subsampled from T×F1 for ViT tokenisation.
                                    8 frames × 196 patches = 1568 tokens ≈ paper setup.
    adapter_dim       int   = 64    AdaptFormer bottleneck dimension for visual streams.
                                    Set to 0 to disable adapters (frozen backbone only).
    dropout           float = 0.1
    freeze_vit        bool  = True  freeze ViT backbone weights (only adapters train)
    modalities        set   = None  active subset of {'V','T','FT','G','GF'}; None = all.
                                    Inactive modalities are zeroed out.
    num_classes       int   = 1     output dimension (1 for binary BCEWithLogitsLoss)

    Design notes
    ────────────
    • ViT-Base/16 backbone: 12 layers, 768-d, 12 heads, 196 patches per 224×224 frame
    • Visual streams use factored spatial + temporal positional embeddings (ViViT-style):
      each frame gets ViT's original 196-patch spatial positions plus a learned temporal
      embedding per frame index, cleanly separating "where in the image" from "which frame".
    • Non-visual temporal streams (FT, Gripper) are projected to 768-d and processed by
      lightweight 4-head transformer blocks, which is sufficient for short sequences.
    • GF (static scalar) becomes a single 768-d token refined by an MLP before fusion.
    • Bottleneck tokens (B × 768) are learnable parameters initialised N(0, 0.02),
      matching ViT positional embedding initialisation. They flow through all fusion layers
      and are the sole conduit for cross-modal information exchange.
    • Consider using different learning rates: higher for lightweight blocks and classifiers,
      lower for AdaptFormer adapters.
    """

    VIT_DIM    = 768
    VIT_LAYERS = 12

    def __init__(
        self,
        frames_per_sec:    int   = 1,
        ft_dim:            int   = 6,
        gripper_dim:       int   = 2,
        max_timesteps:     int   = 20,
        num_bottlenecks:   int   = 4,
        fusion_layer:      int   = 8,
        max_visual_frames: int   = 8,
        adapter_dim:       int   = 64,
        dropout:           float = 0.1,
        freeze_vit:        bool  = True,
        modalities                = None,
        num_classes:       int   = 1,
    ):
        super().__init__()

        assert 0 <= fusion_layer <= self.VIT_LAYERS
        assert num_bottlenecks >= 1

        self.frames_per_sec    = frames_per_sec
        self.ft_dim            = ft_dim
        self.gripper_dim       = gripper_dim
        self.max_timesteps     = max_timesteps
        self.Nb                = num_bottlenecks
        self.fusion_layer      = fusion_layer
        self.max_visual_frames = max_visual_frames
        self.modalities        = set(modalities or ['V', 'T', 'FT', 'G', 'GF'])
        self.num_classes       = num_classes
        D = self.VIT_DIM

        # ── ViT-Base/16 backbones for visual streams ────────────────────────
        vit_tac = timm.create_model('vit_base_patch16_224', pretrained=True)
        vit_rgb = timm.create_model('vit_base_patch16_224', pretrained=True)
        vit_tac.head = vit_rgb.head = nn.Identity()

        if freeze_vit:
            for p in list(vit_tac.parameters()) + list(vit_rgb.parameters()):
                p.requires_grad = False

        # Patch embedding, CLS token, spatial positional embedding, final LayerNorm
        self.tac_patch_embed = vit_tac.patch_embed
        self.rgb_patch_embed = vit_rgb.patch_embed
        self.tac_cls_token   = vit_tac.cls_token            # (1, 1, 768)
        self.rgb_cls_token   = vit_rgb.cls_token
        self.tac_spatial_pos = vit_tac.pos_embed             # (1, 197, 768)
        self.rgb_spatial_pos = vit_rgb.pos_embed
        self.tac_norm        = vit_tac.norm
        self.rgb_norm        = vit_rgb.norm

        # ── Factored temporal positional embeddings for visual streams ──────
        # Separate from the spatial positions so the model can distinguish
        # "top-left of frame 3" from "top-left of frame 7".
        # Shape: (1, max_frames, 1, D) — broadcasts over all patches in a frame.
        self.tac_temporal_embed = nn.Parameter(
            torch.zeros(1, max_visual_frames, 1, D))
        self.rgb_temporal_embed = nn.Parameter(
            torch.zeros(1, max_visual_frames, 1, D))
        nn.init.trunc_normal_(self.tac_temporal_embed, std=0.02)
        nn.init.trunc_normal_(self.rgb_temporal_embed, std=0.02)

        # ── Tokenisation layers for non-visual modalities ───────────────────
        # Each timestep is projected to 768-d with a CLS token prepended.

        self.ft_proj = nn.Sequential(nn.Linear(ft_dim, D), nn.LayerNorm(D))
        self.ft_cls_token = nn.Parameter(torch.zeros(1, 1, D))
        self.ft_pos_embed = nn.Parameter(torch.zeros(1, max_timesteps + 1, D))
        nn.init.trunc_normal_(self.ft_cls_token, std=0.02)
        nn.init.trunc_normal_(self.ft_pos_embed, std=0.02)

        self.grip_proj = nn.Sequential(nn.Linear(gripper_dim, D), nn.LayerNorm(D))
        self.grip_cls_token = nn.Parameter(torch.zeros(1, 1, D))
        self.grip_pos_embed = nn.Parameter(torch.zeros(1, max_timesteps + 1, D))
        nn.init.trunc_normal_(self.grip_cls_token, std=0.02)
        nn.init.trunc_normal_(self.grip_pos_embed, std=0.02)

        # GF is a single scalar — projected to one 768-d token (acts as its own CLS)
        self.gf_proj = nn.Sequential(nn.Linear(1, D), nn.LayerNorm(D))

        # Final LayerNorms for non-visual streams (matching ViT's post-norm)
        self.ft_norm   = nn.LayerNorm(D)
        self.grip_norm = nn.LayerNorm(D)
        self.gf_norm   = nn.LayerNorm(D)

        # ── Unimodal layers (layers 0 … Lf-1) ──────────────────────────────
        # Each stream processes its tokens independently before fusion begins.

        # Visual: frozen ViT blocks with optional AdaptFormer adapters
        self.tac_unimodal = nn.ModuleList([
            UnimodalBlock(vit_tac.blocks[i], adapter_dim)
            for i in range(fusion_layer)
        ])
        self.rgb_unimodal = nn.ModuleList([
            UnimodalBlock(vit_rgb.blocks[i], adapter_dim)
            for i in range(fusion_layer)
        ])

        # Temporal: lightweight trainable transformer blocks
        n_unimodal = max(fusion_layer, 1)  # at least 1 layer for temporal streams
        self.ft_unimodal = nn.ModuleList([
            LightweightTransformerBlock(D, num_heads=4, dropout=dropout)
            for _ in range(n_unimodal)
        ])
        self.grip_unimodal = nn.ModuleList([
            LightweightTransformerBlock(D, num_heads=4, dropout=dropout)
            for _ in range(n_unimodal)
        ])

        # Static: MLP refinement (self-attention on a single token is a no-op)
        self.gf_unimodal = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D, D),
        )

        # ── Bottleneck fusion layers (layers Lf … 11) ──────────────────────
        # Each layer contains stream-specific parameters for all 5 modalities.
        # Visual streams use frozen ViT blocks + adapters; others use lightweight blocks.
        self.fusion_blocks = nn.ModuleList()
        for i in range(fusion_layer, self.VIT_LAYERS):
            stream_blocks = {
                'T':  StreamParams(vit_tac.blocks[i], D, adapter_dim, is_lightweight=False),
                'V':  StreamParams(vit_rgb.blocks[i], D, adapter_dim, is_lightweight=False),
                'FT': StreamParams(LightweightTransformerBlock(D, num_heads=4, dropout=dropout),
                                   D, adapter_dim=0, is_lightweight=True),
                'G':  StreamParams(LightweightTransformerBlock(D, num_heads=4, dropout=dropout),
                                   D, adapter_dim=0, is_lightweight=True),
                'GF': StreamParams(LightweightTransformerBlock(D, num_heads=4, dropout=dropout),
                                   D, adapter_dim=0, is_lightweight=True),
            }
            self.fusion_blocks.append(
                BottleneckFusionBlock(stream_blocks, num_bottlenecks=num_bottlenecks)
            )

        # Shared bottleneck tokens — the sole conduit for cross-modal information.
        # Initialised N(0, 0.02) to match ViT positional embedding initialisation.
        # These flow through all fusion layers and are NOT per-layer parameters.
        self.fusion_bottlenecks = nn.Parameter(
            torch.empty(1, num_bottlenecks, D).normal_(std=0.02)
        )

        # ── Per-modality classification heads ───────────────────────────────
        # Each modality's CLS token → its own linear head → logit(s).
        # Final prediction = average of all per-modality pre-softmax logits.
        self.classifier_T  = nn.Linear(D, num_classes)
        self.classifier_V  = nn.Linear(D, num_classes)
        self.classifier_FT = nn.Linear(D, num_classes)
        self.classifier_G  = nn.Linear(D, num_classes)
        self.classifier_GF = nn.Linear(D, num_classes)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _subsample_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Uniformly subsample to self.max_visual_frames to keep the attention
        budget manageable (8 frames × 196 patches = 1568 tokens ≈ paper setup).

        Args:   frames: (B, N_frames, 3, H, W)
        Returns:        (B, min(N_frames, max_visual_frames), 3, H, W)
        """
        N = frames.shape[1]
        M = self.max_visual_frames
        if N <= M:
            return frames
        idx = torch.linspace(0, N - 1, M, dtype=torch.long, device=frames.device)
        return frames[:, idx]

    def _tokenize_visual(
        self,
        frames:          torch.Tensor,   # (B, F, 3, H, W) — already subsampled
        patch_embed,
        cls_token,                        # (1, 1, D)
        spatial_pos,                      # (1, 197, D)
        temporal_embed,                   # (1, max_F, 1, D)
    ) -> torch.Tensor:
        """
        Convert video frames to a sequence of 768-d tokens with factored positional
        embeddings (ViViT-style).

        Each frame is independently patch-embedded (16×16 patches → 196 tokens per frame).
        Spatial positions from ViT's pretrained embedding are reused for every frame.
        A separate learned temporal embedding is added per frame index to encode ordering.

        Returns: (B, 1 + F×196, D) — CLS token followed by all frame patch tokens
        """
        B, F, C, H, W = frames.shape
        D = cls_token.shape[-1]

        # Patch embed all frames: (B×F, 196, D) → (B, F, 196, D)
        x = patch_embed(frames.reshape(B * F, C, H, W))
        _, L, _ = x.shape  # L = 196
        x = x.reshape(B, F, L, D)

        # Spatial positional embedding: shared across all frames
        # spatial_pos[:, 0] is the CLS position; spatial_pos[:, 1:] are patch positions
        x = x + spatial_pos[:, 1:, :].unsqueeze(1)  # (1, 1, 196, D) broadcasts over F

        # Temporal positional embedding: unique per frame index
        F_actual = min(F, temporal_embed.shape[1])
        x[:, :F_actual] = x[:, :F_actual] + temporal_embed[:, :F_actual]

        # Flatten to a single sequence: (B, F×196, D)
        x = x.reshape(B, F * L, D)

        # Prepend CLS token (with its spatial position from ViT)
        cls = cls_token.expand(B, -1, -1) + spatial_pos[:, :1, :]
        x = torch.cat([cls, x], dim=1)  # (B, 1 + F×196, D)

        return x

    def _tokenize_temporal(
        self,
        sequence:   torch.Tensor,   # (B, T, raw_dim)
        proj:       nn.Module,       # raw_dim → 768
        cls_token:  torch.Tensor,   # (1, 1, D)
        pos_embed:  torch.Tensor,   # (1, max_T+1, D)
    ) -> torch.Tensor:
        """
        Project a low-dimensional temporal sequence to 768-d tokens, prepend a
        CLS token, and add learned positional embeddings.

        Returns: (B, 1 + T, D)
        """
        B, T, _ = sequence.shape

        x = proj(sequence)                         # (B, T, D)
        cls = cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)             # (B, 1+T, D)

        # Positional embedding (truncate if T fits, interpolate if T > max_timesteps)
        seq_len = x.shape[1]
        if seq_len <= pos_embed.shape[1]:
            x = x + pos_embed[:, :seq_len]
        else:
            x = x + F.interpolate(
                pos_embed.permute(0, 2, 1),
                size=seq_len, mode='linear', align_corners=False
            ).permute(0, 2, 1)

        return x

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, tactile, rgb, ft, gripper, gripper_force):
        """
        Args:
            tactile:       (B, T, F1, 3, H, W)  — GelSight tactile images
            rgb:           (B, T, F1, 3, H, W)  — RGB camera images
            ft:            (B, T, FT_DIM)        — force-torque readings per timestep
            gripper:       (B, T, GR_DIM)        — gripper state per timestep
            gripper_force: (B, 1)                — scalar gripper force
        Returns:
            (B, num_classes) raw logits
        """
        mods = self.modalities
        B, T, F1 = tactile.shape[:3]

        # Zero out inactive modalities (preserves tensor shapes for uniform processing)
        if 'T'  not in mods: tactile       = tactile       * 0.0
        if 'V'  not in mods: rgb           = rgb           * 0.0
        if 'FT' not in mods: ft            = ft            * 0.0
        if 'G'  not in mods: gripper       = gripper       * 0.0
        if 'GF' not in mods: gripper_force = gripper_force * 0.0

        # ── Stage 1: Tokenisation ───────────────────────────────────────────

        # Visual streams: subsample frames, patch-embed, add positional embeddings
        tac_frames = self._subsample_frames(tactile.reshape(B, T * F1, *tactile.shape[3:]))
        rgb_frames = self._subsample_frames(rgb.reshape(B, T * F1, *rgb.shape[3:]))

        tac_tok = self._tokenize_visual(
            tac_frames, self.tac_patch_embed,
            self.tac_cls_token, self.tac_spatial_pos, self.tac_temporal_embed)
        rgb_tok = self._tokenize_visual(
            rgb_frames, self.rgb_patch_embed,
            self.rgb_cls_token, self.rgb_spatial_pos, self.rgb_temporal_embed)

        # Temporal streams: project to 768-d, add CLS token and positional embeddings
        ft_tok   = self._tokenize_temporal(ft, self.ft_proj, self.ft_cls_token, self.ft_pos_embed)
        grip_tok = self._tokenize_temporal(gripper, self.grip_proj, self.grip_cls_token, self.grip_pos_embed)

        # Static stream: project scalar to a single 768-d token
        gf_tok = self.gf_proj(gripper_force).unsqueeze(1)   # (B, 1, D)

        # ── Stage 2: Unimodal processing (layers 0 … Lf-1) ─────────────────

        for blk in self.tac_unimodal:
            tac_tok = blk(tac_tok)
        for blk in self.rgb_unimodal:
            rgb_tok = blk(rgb_tok)
        for blk in self.ft_unimodal:
            ft_tok = blk(ft_tok)
        for blk in self.grip_unimodal:
            grip_tok = blk(grip_tok)
        gf_tok = gf_tok + self.gf_unimodal(gf_tok)          # residual MLP refinement

        # ── Stage 3: Bottleneck fusion (layers Lf … 11) ────────────────────
        # All active modalities exchange information through shared bottleneck tokens.

        token_dict = {'T': tac_tok, 'V': rgb_tok, 'FT': ft_tok, 'G': grip_tok, 'GF': gf_tok}
        token_dict = {k: v for k, v in token_dict.items() if k in mods}

        bn = self.fusion_bottlenecks.expand(B, -1, -1)
        for blk in self.fusion_blocks:
            token_dict, bn = blk(token_dict, bn)

        # ── Stage 4: Classification ─────────────────────────────────────────
        # Each modality's CLS token → per-modality classifier → logits.
        # Final prediction = mean of all per-modality pre-softmax logits.

        logits_list = []

        if 'T' in token_dict:
            logits_list.append(self.classifier_T(self.tac_norm(token_dict['T'])[:, 0]))
        if 'V' in token_dict:
            logits_list.append(self.classifier_V(self.rgb_norm(token_dict['V'])[:, 0]))
        if 'FT' in token_dict:
            logits_list.append(self.classifier_FT(self.ft_norm(token_dict['FT'])[:, 0]))
        if 'G' in token_dict:
            logits_list.append(self.classifier_G(self.grip_norm(token_dict['G'])[:, 0]))
        if 'GF' in token_dict:
            logits_list.append(self.classifier_GF(self.gf_norm(token_dict['GF'])[:, 0]))

        logits = torch.stack(logits_list, dim=0).mean(dim=0)  # (B, num_classes)
        return logits