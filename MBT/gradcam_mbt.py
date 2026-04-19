"""
GradCAM-based Pixel Fidelity Score (Insertion AUC) for MBTGraspStability.

Reports per-modality Insertion AUC for RGB (ViT-Base/16) and tactile (T3, ViT-like).
Directly comparable to the LSTM PFS results from R3 (same insertion protocol).

Usage:
    python MBT/gradcam_mbt.py \\
        --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \\
        --checkpoint trained_models/best_mbt_model.pt \\
        --modalities V T FT G GF \\
        --gradcam_mode both \\
        --L 9 --max_visual_frames 8 \\
        --steps 10 --n_samples 50 --n_vis 8 \\
        --vis_dir mbt_pfs_vis

Outputs saved to --vis_dir:
    gradcam_grid.png  — overlay grid (tac frame | tac GradCAM | rgb frame | rgb GradCAM)
    curves.png        — mean ± std insertion confidence curves
    metrics.json      — {tac_ins_auc, rgb_ins_auc, n_samples}

Target layers:
    RGB      → model.norms['V']  (pretrained vit_rgb.norm, LayerNorm over 768-d)
    Tactile  → model.norms['T']  (nn.LayerNorm(768), applied after tac_to_fusion projection)

Both produce activations of shape (B, 1 + F*196, 768).
We pick the patches for a single frame and reshape to (B, 768, 14, 14) for GradCAM.
"""

import argparse
import json
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Repo root is one level up from MBT/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataloader as _dl
from dataloader import PoseItDataset, uniform_random_split, FT_DIM, GR_DIM
from mbt_model import MBTGraspStability


# ImageNet stats used by the dataloader — needed to de-normalise for display
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

NUM_PATCHES = 14   # ViT-Base/16 with 224×224 input → 14×14 patch grid
PATCHES_PER_FRAME = NUM_PATCHES * NUM_PATCHES   # 196


# ---------------------------------------------------------------------------
# Helper: tensor → display-ready uint8 numpy (H, W, 3) ∈ [0,1]
# ---------------------------------------------------------------------------

def tensor_to_display(t: torch.Tensor) -> np.ndarray:
    """
    Convert a (3, H, W) ImageNet-normalised float tensor to (H, W, 3) float32 in [0,1].
    Safe for tactile frames which are baseline-subtracted and may be negative.
    """
    img = t.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    img = img * _IMAGENET_STD + _IMAGENET_MEAN   # denormalise
    return np.clip(img, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Frame wrapper: holds full 5-modal context, swaps one frame for GradCAM
# ---------------------------------------------------------------------------

class MBTFrameWrapper(nn.Module):
    """
    Holds a full multi-timestep batch; substitutes a single frame at timestep t
    so GradCAM can treat the model as f(single_frame) → logit.
    """

    def __init__(
        self,
        model: MBTGraspStability,
        tac: torch.Tensor,    # (1, T, F1, 3, H, W)
        rgb: torch.Tensor,    # (1, T, F1, 3, H, W)
        ft: torch.Tensor,     # (1, T, FT_DIM)
        gripper: torch.Tensor,     # (1, T, GR_DIM)
        gripper_force: torch.Tensor,  # (1, 1)
        t: int,
        modality: str,        # 'rgb' or 'tactile'
    ):
        super().__init__()
        self.model         = model
        self.tac           = tac
        self.rgb           = rgb
        self.ft            = ft
        self.gripper       = gripper
        self.gripper_force = gripper_force
        self.t             = t
        self.modality      = modality

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: (1, 3, H, W) — the single frame being probed by GradCAM."""
        tac = self.tac.clone()
        rgb = self.rgb.clone()
        if self.modality == 'tactile':
            tac[:, self.t, 0] = img
        else:
            rgb[:, self.t, 0] = img
        out = self.model(tac, rgb, self.ft, self.gripper, self.gripper_force)  # (1, 1)
        return out


# ---------------------------------------------------------------------------
# Reshape transform: pick frame t's patches from ViT token sequence
# ---------------------------------------------------------------------------

def make_reshape_transform(t: int, num_frames: int):
    """
    Returns a callable that converts (B, 1 + F*196, D) activations at the
    target LayerNorm into (B, D, 14, 14) — the format pytorch-grad-cam expects.

    t          : frame index within the subsampled sequence (0-based)
    num_frames : F (actual number of frames passed after subsampling)
    """
    def _reshape(tensor: torch.Tensor) -> torch.Tensor:
        # tensor: (B, 1 + num_frames*196, D)
        # frame t occupies positions [1 + t*196 : 1 + (t+1)*196]
        s = 1 + t * PATCHES_PER_FRAME
        e = s + PATCHES_PER_FRAME
        # Guard against tensor being shorter than expected (e.g. if T < max_visual_frames)
        if e > tensor.shape[1]:
            # Fall back to last available frame
            available = (tensor.shape[1] - 1) // PATCHES_PER_FRAME
            t_clamp = max(available - 1, 0)
            s = 1 + t_clamp * PATCHES_PER_FRAME
            e = s + PATCHES_PER_FRAME
        patches = tensor[:, s:e, :]                                    # (B, 196, D)
        B, _, D = patches.shape
        return patches.reshape(B, NUM_PATCHES, NUM_PATCHES, D).permute(0, 3, 1, 2)
        # → (B, D, 14, 14)
    return _reshape


# ---------------------------------------------------------------------------
# Confidence helper
# ---------------------------------------------------------------------------

def _conf(logit: torch.Tensor, target_class: int) -> float:
    p = torch.sigmoid(logit).flatten()[0].item()
    return p if target_class == 1 else 1.0 - p


# ---------------------------------------------------------------------------
# GradCAM heatmap for one frame
# ---------------------------------------------------------------------------

def gradcam_heatmap(
    wrapper: MBTFrameWrapper,
    frame_tensor: torch.Tensor,   # (1, 3, H, W) float32 on device
    target_layer: nn.Module,
    reshape_transform,
    target_class: int,
) -> np.ndarray:
    """Returns (H, W) float32 heatmap in [0, 1]."""
    sign = 1.0 if target_class == 1 else -1.0

    def _target(out):
        return sign * out.flatten()[0]

    # pytorch-grad-cam needs requires_grad on the input tensor
    frame_tensor = frame_tensor.requires_grad_(True)

    with GradCAM(
        model=wrapper,
        target_layers=[target_layer],
        reshape_transform=reshape_transform,
    ) as cam:
        heatmap = cam(
            input_tensor=frame_tensor,
            targets=[_target],
        )
    return heatmap[0]   # (H, W)


# ---------------------------------------------------------------------------
# Insertion AUC for one modality / one frame
# ---------------------------------------------------------------------------

def insertion_auc_single(
    model: MBTGraspStability,
    tac: torch.Tensor,
    rgb: torch.Tensor,
    ft: torch.Tensor,
    gripper: torch.Tensor,
    gripper_force: torch.Tensor,
    t: int,
    heatmap: np.ndarray,
    frame_np: np.ndarray,    # (H, W, 3) float32 in [0,1]
    modality: str,
    steps: int,
    target_class: int,
) -> tuple:
    """
    Blur the chosen frame, then reveal pixels in decreasing-importance order.
    Returns (auc: float, confidences: list[float] of length steps+1).
    """
    H, W = heatmap.shape
    order = heatmap.flatten().argsort()[::-1]
    n_pixels = order.size
    device = tac.device

    blurred_np = cv2.GaussianBlur(frame_np, (51, 51), 0)  # (H,W,3) float32 in [0,1]

    # Re-normalise the blurred frame back to ImageNet space so it matches the
    # tensor space the model expects (frame_np was denormalised for display).
    blurred_norm = (blurred_np - _IMAGENET_MEAN) / _IMAGENET_STD   # (H,W,3)

    # Replace frame t with the blurred version in the chosen modality
    tac_seq = tac.clone()
    rgb_seq = rgb.clone()
    blurred_t = torch.from_numpy(blurred_norm).permute(2, 0, 1).float().to(device)

    if modality == 'tactile':
        tac_seq[:, t, 0] = blurred_t
        orig_pixels = tac[:, t, 0]          # (1, 3, H, W)
    else:
        rgb_seq[:, t, 0] = blurred_t
        orig_pixels = rgb[:, t, 0]

    confidences = []
    with torch.no_grad():
        for step in range(steps + 1):
            k = int(step / steps * n_pixels)
            cur_tac = tac_seq.clone()
            cur_rgb = rgb_seq.clone()
            if k > 0:
                mask = np.zeros(n_pixels, dtype=bool)
                mask[order[:k]] = True
                mask2d = torch.from_numpy(mask.reshape(H, W)).to(device)   # (H, W)
                if modality == 'tactile':
                    cur_tac[:, t, 0, :, mask2d] = orig_pixels[:, :, mask2d]
                else:
                    cur_rgb[:, t, 0, :, mask2d] = orig_pixels[:, :, mask2d]
            logit = model(cur_tac, cur_rgb, ft, gripper, gripper_force)
            confidences.append(_conf(logit, target_class))

    auc = float(np.trapezoid(confidences, dx=1.0 / steps))
    return auc, confidences


# ---------------------------------------------------------------------------
# Visualisation helpers (ported from mlee/gradcam)
# ---------------------------------------------------------------------------

def save_overlay_grid(vis_items: list, gradcam_mode: str, save_path: str):
    n = len(vis_items)
    if gradcam_mode == 'both':
        ncols = 4
    else:
        ncols = 2

    _, axes = plt.subplots(n, ncols, figsize=(3 * ncols, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, item in enumerate(vis_items):
        label_str = 'slip/drop' if item['label'] == 1 else 'pass'

        if gradcam_mode == 'both':
            tac_overlay = show_cam_on_image(item['tac_np'], item['tac_hm'], use_rgb=True)
            rgb_overlay = show_cam_on_image(item['rgb_np'], item['rgb_hm'], use_rgb=True)

            axes[row, 0].imshow(item['tac_np'])
            axes[row, 0].set_title(f"Tactile [{item['idx']}] {label_str}", fontsize=8)
            axes[row, 0].axis('off')
            axes[row, 1].imshow(tac_overlay)
            axes[row, 1].set_title(f"Tac GradCAM  AUC={item['tac_ins']:.3f}", fontsize=8)
            axes[row, 1].axis('off')
            axes[row, 2].imshow(item['rgb_np'])
            axes[row, 2].set_title(f"RGB [{item['idx']}] {label_str}", fontsize=8)
            axes[row, 2].axis('off')
            axes[row, 3].imshow(rgb_overlay)
            axes[row, 3].set_title(f"RGB GradCAM  AUC={item['rgb_ins']:.3f}", fontsize=8)
            axes[row, 3].axis('off')

        elif gradcam_mode == 'tactile':
            overlay = show_cam_on_image(item['tac_np'], item['tac_hm'], use_rgb=True)
            axes[row, 0].imshow(item['tac_np'])
            axes[row, 0].set_title(f"Tactile [{item['idx']}] {label_str}", fontsize=8)
            axes[row, 0].axis('off')
            axes[row, 1].imshow(overlay)
            axes[row, 1].set_title(f"Tac GradCAM  AUC={item['tac_ins']:.3f}", fontsize=8)
            axes[row, 1].axis('off')

        else:  # rgb
            overlay = show_cam_on_image(item['rgb_np'], item['rgb_hm'], use_rgb=True)
            axes[row, 0].imshow(item['rgb_np'])
            axes[row, 0].set_title(f"RGB [{item['idx']}] {label_str}", fontsize=8)
            axes[row, 0].axis('off')
            axes[row, 1].imshow(overlay)
            axes[row, 1].set_title(f"RGB GradCAM  AUC={item['rgb_ins']:.3f}", fontsize=8)
            axes[row, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved overlay grid → {save_path}")


def save_curves_plot(curves_by_label: dict, steps: int, save_path: str):
    xs = np.linspace(0, 1, steps + 1)
    colors = {'Tactile': 'darkorange', 'RGB': 'steelblue'}

    _, ax = plt.subplots(figsize=(6, 4))
    for label, curves in curves_by_label.items():
        arr = np.array(curves)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        c = colors.get(label, 'gray')
        ax.plot(xs, mean, color=c, label=f'{label} Insertion', linewidth=2)
        ax.fill_between(xs, mean - std, mean + std, color=c, alpha=0.2)

    ax.set_xlabel('Fraction of pixels revealed')
    ax.set_ylabel('Model confidence (target class)')
    ax.set_title('Insertion AUC curves (MBT — mean ± std)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved curves plot   → {save_path}")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_gradcam(
    model: MBTGraspStability,
    dataset: PoseItDataset,
    indices,
    gradcam_mode: str,
    steps: int,
    n_samples: int,
    n_vis: int,
    vis_dir: str,
    device: str,
):
    """
    gradcam_mode: 'rgb' | 'tactile' | 'both'
    Returns dict with keys tac_ins_auc / rgb_ins_auc (whichever are active).
    """
    model.eval()

    do_tac = gradcam_mode in ('tactile', 'both')
    do_rgb = gradcam_mode in ('rgb', 'both')

    # Target LayerNorms — these receive the post-fusion CLS + patch tokens.
    # They have requires_grad=True by default (they are not frozen backbone layers).
    tac_target = model.norms['T'] if do_tac else None
    rgb_target = model.norms['V'] if do_rgb else None

    rng = np.random.default_rng(42)
    chosen = rng.choice(indices, size=min(n_samples, len(indices)), replace=False)

    tac_ins_list, rgb_ins_list = [], []
    tac_curves_all, rgb_curves_all = [], []
    vis_items = []

    for idx in chosen:
        sample = dataset[idx]
        tac_s, rgb_s, ft_s, grip_s, gf_s, label_s, _ = sample

        target_class = int(label_s.item())

        tac = tac_s.unsqueeze(0).to(device)      # (1, T, F1, 3, H, W)
        rgb = rgb_s.unsqueeze(0).to(device)
        ft  = ft_s.unsqueeze(0).to(device)
        grip = grip_s.unsqueeze(0).to(device)
        gf   = gf_s.unsqueeze(0).to(device)

        T   = tac.shape[1]
        F1s = tac.shape[2]   # F1 (= 1 in training)
        total_frames = T * F1s
        M = model.max_visual_frames

        # Replicate the model's _subsample_frames logic to know which original
        # frames survive subsampling.  t_sub  = position in the token sequence
        # t_orig = index into the flattened (T*F1) tensor before subsampling
        if total_frames <= M:
            sub_indices = list(range(total_frames))
        else:
            sub_indices = torch.linspace(
                0, total_frames - 1, M, dtype=torch.long).tolist()
            sub_indices = [int(i) for i in sub_indices]

        num_frames = len(sub_indices)
        mid_t_sub  = num_frames // 2

        sample_tac_ins, sample_rgb_ins = [], []
        mid_vis = {}

        for t_sub, t_flat in enumerate(sub_indices):
            # t_flat : index in the flattened (T*F1) dimension
            # t_orig : index in the T dimension of the (B, T, F1, 3, H, W) tensor
            t_orig = t_flat // F1s

            reshape_fn = make_reshape_transform(t_sub, num_frames)

            if do_tac:
                frame_np = tensor_to_display(tac[0, t_orig, 0])
                frame_t  = tac[0, t_orig, 0].unsqueeze(0).to(device)   # (1,3,H,W)

                wrapper = MBTFrameWrapper(model, tac, rgb, ft, grip, gf, t_orig, 'tactile')
                wrapper.eval()

                hm = gradcam_heatmap(wrapper, frame_t, tac_target, reshape_fn, target_class)
                auc, curve = insertion_auc_single(
                    model, tac, rgb, ft, grip, gf,
                    t_orig, hm, frame_np, 'tactile', steps, target_class)
                sample_tac_ins.append(auc)
                tac_curves_all.append(curve)
                if t_sub == mid_t_sub:
                    mid_vis['tac_np'] = frame_np
                    mid_vis['tac_hm'] = hm

            if do_rgb:
                frame_np = tensor_to_display(rgb[0, t_orig, 0])
                frame_t  = rgb[0, t_orig, 0].unsqueeze(0).to(device)

                wrapper = MBTFrameWrapper(model, tac, rgb, ft, grip, gf, t_orig, 'rgb')
                wrapper.eval()

                hm = gradcam_heatmap(wrapper, frame_t, rgb_target, reshape_fn, target_class)
                auc, curve = insertion_auc_single(
                    model, tac, rgb, ft, grip, gf,
                    t_orig, hm, frame_np, 'rgb', steps, target_class)
                sample_rgb_ins.append(auc)
                rgb_curves_all.append(curve)
                if t_sub == mid_t_sub:
                    mid_vis['rgb_np'] = frame_np
                    mid_vis['rgb_hm'] = hm

        s_tac = float(np.mean(sample_tac_ins)) if sample_tac_ins else 0.0
        s_rgb = float(np.mean(sample_rgb_ins)) if sample_rgb_ins else 0.0
        if do_tac: tac_ins_list.append(s_tac)
        if do_rgb: rgb_ins_list.append(s_rgb)

        parts = [f"sample {idx:4d} | label={target_class}"]
        if do_tac: parts.append(f"tac_ins={s_tac:.4f}")
        if do_rgb: parts.append(f"rgb_ins={s_rgb:.4f}")
        print("  " + "  ".join(parts))

        if len(vis_items) < n_vis:
            entry = {'idx': idx, 'label': target_class,
                     'tac_ins': s_tac, 'rgb_ins': s_rgb}
            entry.update(mid_vis)
            vis_items.append(entry)

    results = {}
    if do_tac: results['tac_ins_auc'] = float(np.mean(tac_ins_list))
    if do_rgb: results['rgb_ins_auc'] = float(np.mean(rgb_ins_list))
    results['n_samples'] = len(chosen)

    if vis_dir and vis_items:
        os.makedirs(vis_dir, exist_ok=True)

        save_overlay_grid(vis_items, gradcam_mode,
                          os.path.join(vis_dir, 'gradcam_grid.png'))

        curves_dict = {}
        if tac_curves_all: curves_dict['Tactile'] = tac_curves_all
        if rgb_curves_all:  curves_dict['RGB']     = rgb_curves_all
        if curves_dict:
            save_curves_plot(curves_dict, steps,
                             os.path.join(vis_dir, 'curves.png'))

        with open(os.path.join(vis_dir, 'metrics.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved metrics.json  → {os.path.join(vis_dir, 'metrics.json')}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='GradCAM + Insertion AUC (PFS) for MBTGraspStability')
    p.add_argument('--root_dir',         required=True)
    p.add_argument('--checkpoint',       required=True,
                   help='Path to best_mbt_model.pt')
    p.add_argument('--gradcam_mode',     default='both',
                   choices=['rgb', 'tactile', 'both'])
    p.add_argument('--split',            default='random',
                   choices=['random'])
    p.add_argument('--L',                type=int, default=None,
                   help='Max seconds per episode for the dataloader. '
                        'None = variable-length (default). Set to an int to clip sequences.')
    p.add_argument('--max_timesteps',   type=int, default=9,
                   help='Must match the value of --L used during training '
                        '(controls pos_embed size in the model). '
                        'Default 9 matches checkpoint from run z92x33dg.')
    p.add_argument('--F1',               type=int, default=1)
    p.add_argument('--F2',               type=int, default=1)
    p.add_argument('--modalities',       nargs='+',
                   default=['V', 'T', 'FT', 'G', 'GF'])
    p.add_argument('--num_bottlenecks',  type=int, default=4)
    p.add_argument('--fusion_layer',     type=int, default=8)
    p.add_argument('--max_visual_frames', type=int, default=8)
    p.add_argument('--adapter_dim',      type=int, default=128,
                   help='Must match training --adapter_dim (default 128 for run z92x33dg)')
    p.add_argument('--dropout',          type=float, default=0.1)
    p.add_argument('--pretrained_dir',   type=str,
                   default='/ocean/projects/cis260031p/shared/pretrained')
    p.add_argument('--t3_encoder_domain', type=str, default='gs_black')
    p.add_argument('--steps',            type=int, default=10,
                   help='Insertion masking steps (higher = smoother AUC curve)')
    p.add_argument('--n_samples',        type=int, default=50,
                   help='Number of test samples to evaluate')
    p.add_argument('--n_vis',            type=int, default=8,
                   help='Number of samples to include in overlay grid')
    p.add_argument('--vis_dir',          type=str, default='mbt_pfs_vis',
                   help='Output directory for visualisations (empty string to skip)')
    return p.parse_args()


def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}  gradcam_mode: {args.gradcam_mode}  steps: {args.steps}")

    # Set global dataloader params before constructing dataset
    _dl.L  = args.L
    _dl.F1 = args.F1
    _dl.F2 = args.F2

    ds = PoseItDataset(root_dir=args.root_dir)
    _, _, test_set = uniform_random_split(ds)
    print(f"Test set: {len(test_set)} samples  "
          f"(evaluating {min(args.n_samples, len(test_set.indices))})")

    model = MBTGraspStability(
        frames_per_sec=_dl.F1,
        ft_dim=FT_DIM,
        gripper_dim=GR_DIM,
        max_timesteps=args.max_timesteps,
        num_bottlenecks=args.num_bottlenecks,
        fusion_layer=args.fusion_layer,
        max_visual_frames=args.max_visual_frames,
        adapter_dim=args.adapter_dim,
        dropout=args.dropout,
        modalities=args.modalities,
        pretrained_dir=args.pretrained_dir,
        t3_encoder_domain=args.t3_encoder_domain,
    ).to(device)

    assert os.path.exists(args.checkpoint), f"Checkpoint not found: {args.checkpoint}"
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True))
    print(f"Loaded checkpoint: {args.checkpoint}")

    results = evaluate_gradcam(
        model, ds, test_set.indices,
        gradcam_mode=args.gradcam_mode,
        steps=args.steps,
        n_samples=args.n_samples,
        n_vis=args.n_vis,
        vis_dir=args.vis_dir or None,
        device=device,
    )

    print(f"\n{'='*55}")
    if 'tac_ins_auc' in results:
        print(f"Tactile Insertion AUC (↑ better): {results['tac_ins_auc']:.4f}")
    if 'rgb_ins_auc' in results:
        print(f"RGB     Insertion AUC (↑ better): {results['rgb_ins_auc']:.4f}")
    print(f"n_samples evaluated: {results['n_samples']}")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()
