"""
MBTGraspStability: Multimodal Bottleneck Transformer for grasp stability prediction.
Based on "Attention Bottlenecks for Multimodal Fusion" (Nagrani et al., NeurIPS 2021).

Streams: T (GelSight/T3-large), V (RGB/ViT-Base), FT (force-torque), G (gripper), GF (gripper force).
Layers 0..Lf-1 are unimodal; layers Lf..11 fuse via shared bottleneck tokens.
Tactile runs at T3's native 1024-d; a Linear(1024→768) projects to fusion dim.

Input:  tactile, rgb (B, T, F1, 3, 224, 224) | ft (B, T, FT_DIM) | gripper (B, T, GR_DIM) | gripper_force (B, 1)
Output: (B, num_classes) raw logits
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# encoders.py lives in the repo root, one level up from MBT/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Adapter(nn.Module):
    """AdaptFormer bottleneck: down→GELU→up, zero-init so it starts as a no-op."""

    def __init__(self, dim: int, adapter_dim: int):
        super().__init__()
        self.down  = nn.Linear(dim, adapter_dim)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(0.1)
        self.up    = nn.Linear(adapter_dim, dim)
        self.scale = nn.Parameter(torch.ones(1))
        nn.init.xavier_uniform_(self.down.weight); nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight);            nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.drop(self.act(self.down(x)))) * self.scale


class UnimodalBlock(nn.Module):
    """
    Frozen ViT block with optional AdaptFormer adapter in parallel with the FFN.
    Used for the RGB stream in both unimodal and fusion stages.
    """

    def __init__(self, vit_block, adapter_dim: int = 64):
        super().__init__()
        self.norm1   = vit_block.norm1  # ViT LayerNorm
        self.attn    = vit_block.attn
        self.norm2   = vit_block.norm2
        self.mlp     = vit_block.mlp
        self.adapter = Adapter(768, adapter_dim) if adapter_dim > 0 else None  # QUESTION: Why hardcoded to 768?

    def _forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        n = self.norm2(x)
        return x + self.mlp(n) + (self.adapter(n) if self.adapter else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_sequence(x)

    def forward_with_bottleneck(self, tokens: torch.Tensor, bn: torch.Tensor, Nb: int):         
        """Concatenate stream tokens with bottleneck, run the block, then split them apart.
        tokens: TODO
        bn: TODO
        Nb: TODO
        """
        out = self._forward_sequence(torch.cat([tokens, bn], dim=1))
        return out[:, :-Nb], out[:, -Nb:]


class LightweightTransformerBlock(nn.Module):
    """
    Fully trainable pre-norm transformer block (4 heads, 2× MLP).
    Used for FT, G, GF, and the tactile fusion stream.
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
        return x + self.mlp(self.norm2(x))

    def forward_with_bottleneck(self, tokens: torch.Tensor, bn: torch.Tensor, Nb: int):
        """Concatenate stream tokens with bottleneck, run the block, then split them apart."""
        out = self.forward(torch.cat([tokens, bn], dim=1))
        return out[:, :-Nb], out[:, -Nb:]


class BottleneckFusionBlock(nn.Module):
    """
    One N-modality bottleneck fusion layer (Eq. 8–9, Nagrani et al. 2021).
    Each stream attends to the shared bottleneck with its own parameters;
    per-stream bottleneck outputs are averaged to form the new shared state.
    """

    def __init__(self, streams: dict, num_bottlenecks: int = 4):
        super().__init__()
        self.streams = nn.ModuleDict(streams)
        self.Nb = num_bottlenecks

    def forward(self, token_dict: dict, bn: torch.Tensor):
        bn_accum = torch.zeros_like(bn)
        out_dict = {}
        n_active = 0
        for key, block in self.streams.items():
            if key not in token_dict:
                continue
            out_dict[key], bn_tmp = block.forward_with_bottleneck(token_dict[key], bn, self.Nb)
            bn_accum = bn_accum + bn_tmp
            n_active += 1
        return out_dict, bn_accum / max(n_active, 1)


class MBTGraspStability(nn.Module):
    """
    5-modality bottleneck transformer for grasp stability (Nagrani et al., NeurIPS 2021).

    Visual streams use ViViT-style factored spatial+temporal positional embeddings.
    Tactile backbone is T3-large (GelSight-pretrained, 1024-d); a trainable
    Linear(1024→768) projects to fusion dim so bottleneck tokens are uniform across streams.
    Non-visual streams (FT, G, GF) use lightweight transformer blocks.
    Per-modality classifier heads; final logit is their mean.
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
        freeze_rgb:        bool  = True,
        freeze_t3:         bool  = True,
        modalities                = None,
        num_classes:       int   = 1,
        pretrained_dir:    str   = None,
        t3_encoder_domain: str   = 'gs_black',
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

        # ── RGB backbone: ViT-Base/16 ────────────────────────────────────────
        vit_rgb = timm.create_model('vit_base_patch16_224', pretrained=True) # TODO replace with pre-trained CLiP?
        vit_rgb.head = nn.Identity()
        if freeze_rgb:
            for p in vit_rgb.parameters():
                p.requires_grad = False

        self.rgb_patch_embed    = vit_rgb.patch_embed
        self.rgb_cls_token      = vit_rgb.cls_token
        self.rgb_spatial_pos    = vit_rgb.pos_embed
        self.rgb_temporal_embed = nn.Parameter(torch.zeros(1, max_visual_frames, 1, D))
        nn.init.trunc_normal_(self.rgb_temporal_embed, std=0.02)

        # ── Tactile backbone: T3-large (3 encoder + 9 trunk = 12 blocks, 1024-d) ──
        # Unimodal runs at 1024-d; tac_to_fusion projects to 768 at the fusion boundary.
        # T3's fusion-range blocks are 1024-d and can't share 768-d bottleneck tokens,
        # so we use lightweight trainable blocks at the fusion stage instead.
        from encoders import T3TactileEncoder
        t3 = T3TactileEncoder(
            pretrained_dir=pretrained_dir,
            encoder_domain=t3_encoder_domain,
            freeze=freeze_t3,
        )
        t3_dim        = t3.embed_dim
        all_t3_blocks = list(t3.encoder.blocks) + list(t3.trunk.blocks)
        assert len(all_t3_blocks) >= self.VIT_LAYERS, (
            f"T3 has {len(all_t3_blocks)} blocks, need {self.VIT_LAYERS}")

        self.tac_dim            = t3_dim
        self.tac_patch_embed    = t3.encoder.patch_embed
        self.tac_cls_token      = t3.encoder.cls_token      # (1, 1, t3_dim)
        self.tac_spatial_pos    = t3.encoder.pos_embed       # (1, 197, t3_dim)
        self.tac_temporal_embed = nn.Parameter(torch.zeros(1, max_visual_frames, 1, t3_dim))
        nn.init.trunc_normal_(self.tac_temporal_embed, std=0.02)
        self.tac_to_fusion      = nn.Linear(t3_dim, D)

        # ── Temporal stream projections (FT, gripper) ────────────────────────
        self.ft_proj        = nn.Sequential(nn.Linear(ft_dim, D), nn.LayerNorm(D))
        self.ft_cls_token   = nn.Parameter(torch.zeros(1, 1, D))
        self.ft_pos_embed   = nn.Parameter(torch.zeros(1, max_timesteps + 1, D))
        nn.init.trunc_normal_(self.ft_cls_token, std=0.02)
        nn.init.trunc_normal_(self.ft_pos_embed, std=0.02)

        self.grip_proj      = nn.Sequential(nn.Linear(gripper_dim, D), nn.LayerNorm(D))
        self.grip_cls_token = nn.Parameter(torch.zeros(1, 1, D))
        self.grip_pos_embed = nn.Parameter(torch.zeros(1, max_timesteps + 1, D))
        nn.init.trunc_normal_(self.grip_cls_token, std=0.02)
        nn.init.trunc_normal_(self.grip_pos_embed, std=0.02)

        # Static stream: single scalar → single 768-d token
        self.gf_proj = nn.Sequential(nn.Linear(1, D), nn.LayerNorm(D))

        # ── Unimodal stacks (layers 0 … Lf-1) ───────────────────────────────
        self.tac_unimodal = nn.ModuleList(all_t3_blocks[:fusion_layer])
        self.rgb_unimodal = nn.ModuleList([
            UnimodalBlock(vit_rgb.blocks[i], adapter_dim) for i in range(fusion_layer)
        ])
        n_unimodal = max(fusion_layer, 1)
        self.ft_unimodal = nn.ModuleList([
            LightweightTransformerBlock(D, num_heads=4, dropout=dropout) for _ in range(n_unimodal)
        ])
        self.grip_unimodal = nn.ModuleList([
            LightweightTransformerBlock(D, num_heads=4, dropout=dropout) for _ in range(n_unimodal)
        ])
        # GF is a single static token — self-attention on one token is a no-op, use MLP instead
        self.gf_unimodal = nn.Sequential(
            nn.LayerNorm(D), nn.Linear(D, D), nn.GELU(), nn.Dropout(dropout), nn.Linear(D, D),
        )

        # ── Bottleneck fusion layers (layers Lf … 11) ────────────────────────
        # Shared bottleneck tokens are the sole conduit for cross-modal information.
        self.fusion_bottlenecks = nn.Parameter(
            torch.empty(1, num_bottlenecks, D).normal_(std=0.02)
        )
        self.fusion_blocks = nn.ModuleList([
            BottleneckFusionBlock({
                'T':  LightweightTransformerBlock(D, num_heads=4, dropout=dropout),
                'V':  UnimodalBlock(vit_rgb.blocks[i], adapter_dim),
                'FT': LightweightTransformerBlock(D, num_heads=4, dropout=dropout),
                'G':  LightweightTransformerBlock(D, num_heads=4, dropout=dropout),
                'GF': LightweightTransformerBlock(D, num_heads=4, dropout=dropout),
            }, num_bottlenecks=num_bottlenecks)
            for i in range(fusion_layer, self.VIT_LAYERS)
        ])

        # ── Per-modality classification heads ─────────────────────────────────
        # Each modality's CLS token → its own head → logit; prediction = mean of all logits.
        self.norms = nn.ModuleDict({
            'T':  nn.LayerNorm(D),
            'V':  vit_rgb.norm,     # reuse pretrained ViT norm
            'FT': nn.LayerNorm(D),
            'G':  nn.LayerNorm(D),
            'GF': nn.LayerNorm(D),
        })
        self.classifiers = nn.ModuleDict({
            k: nn.Linear(D, num_classes) for k in ['T', 'V', 'FT', 'G', 'GF']
        })

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _subsample_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Uniformly subsample to max_visual_frames (8 frames × 196 patches ≈ paper setup)."""
        N, M = frames.shape[1], self.max_visual_frames
        if N <= M:
            return frames
        idx = torch.linspace(0, N - 1, M, dtype=torch.long, device=frames.device)
        return frames[:, idx]

    def _tokenize_visual(self, frames, patch_embed, cls_token, spatial_pos, temporal_embed):
        """
        Patch-embed video frames and add factored spatial+temporal positional embeddings (ViViT-style).

        Args:   frames: (B, F, 3, H, W)
        Returns:        (B, 1 + F×196, D) — CLS token followed by all frame patch tokens
        """
        B, F, C, H, W = frames.shape
        D = cls_token.shape[-1]

        x = patch_embed(frames.reshape(B * F, C, H, W))    # (B×F, 196, D)
        _, L, _ = x.shape
        x = x.reshape(B, F, L, D)
        x = x + spatial_pos[:, 1:, :].unsqueeze(1)          # spatial pos, broadcast over F

        F_actual = min(F, temporal_embed.shape[1])
        x[:, :F_actual] = x[:, :F_actual] + temporal_embed[:, :F_actual]  # temporal pos

        x = x.reshape(B, F * L, D)
        cls = cls_token.expand(B, -1, -1) + spatial_pos[:, :1, :]
        return torch.cat([cls, x], dim=1)                    # (B, 1 + F×196, D)

    def _tokenize_temporal(self, sequence, proj, cls_token, pos_embed):
        """
        Project a temporal sequence to 768-d tokens, prepend a CLS token, add positional embeddings.

        Args:   sequence: (B, T, raw_dim)
        Returns:          (B, 1 + T, D)
        """
        B = sequence.shape[0]
        x = proj(sequence)
        x = torch.cat([cls_token.expand(B, -1, -1), x], dim=1)  # (B, 1+T, D)

        seq_len = x.shape[1]
        if seq_len <= pos_embed.shape[1]:
            x = x + pos_embed[:, :seq_len]
        else:
            x = x + F.interpolate(
                pos_embed.permute(0, 2, 1), size=seq_len, mode='linear', align_corners=False
            ).permute(0, 2, 1)
        return x

    # ── Forward ───────────────────────────────────────────────────────────────

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

        # ── Stage 1: Tokenisation ────────────────────────────────────────────

        tac_frames = self._subsample_frames(tactile.reshape(B, T * F1, *tactile.shape[3:]))
        rgb_frames = self._subsample_frames(rgb.reshape(B, T * F1, *rgb.shape[3:]))

        tac_tok  = self._tokenize_visual(tac_frames, self.tac_patch_embed,
                                         self.tac_cls_token, self.tac_spatial_pos,
                                         self.tac_temporal_embed)
        rgb_tok  = self._tokenize_visual(rgb_frames, self.rgb_patch_embed,
                                         self.rgb_cls_token, self.rgb_spatial_pos,
                                         self.rgb_temporal_embed)
        ft_tok   = self._tokenize_temporal(ft, self.ft_proj, self.ft_cls_token, self.ft_pos_embed)
        grip_tok = self._tokenize_temporal(gripper, self.grip_proj,
                                           self.grip_cls_token, self.grip_pos_embed)
        gf_tok   = self.gf_proj(gripper_force).unsqueeze(1)  # (B, 1, D)

        # ── Stage 2: Unimodal processing (layers 0 … Lf-1) ──────────────────

        for blk in self.tac_unimodal:
            tac_tok = blk(tac_tok)
        tac_tok = self.tac_to_fusion(tac_tok)  # 1024-d → 768-d at fusion boundary

        for blk in self.rgb_unimodal:
            rgb_tok = blk(rgb_tok)
        for blk in self.ft_unimodal:
            ft_tok = blk(ft_tok)
        for blk in self.grip_unimodal:
            grip_tok = blk(grip_tok)
        gf_tok = gf_tok + self.gf_unimodal(gf_tok)  # residual MLP refinement

        # ── Stage 3: Bottleneck fusion (layers Lf … 11) ──────────────────────

        token_dict = {k: v for k, v in
                      [('T', tac_tok), ('V', rgb_tok), ('FT', ft_tok),
                       ('G', grip_tok), ('GF', gf_tok)]
                      if k in mods}

        bn = self.fusion_bottlenecks.expand(B, -1, -1)
        for blk in self.fusion_blocks:
            token_dict, bn = blk(token_dict, bn)

        # ── Stage 4: Classification ───────────────────────────────────────────
        # Each modality's CLS token → per-modality head → logit; final = mean.

        logits_list = [
            self.classifiers[k](self.norms[k](token_dict[k])[:, 0])
            for k in token_dict
        ]
        return torch.stack(logits_list, dim=0).mean(dim=0)  # (B, num_classes)
