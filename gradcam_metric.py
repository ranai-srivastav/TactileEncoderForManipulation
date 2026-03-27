"""
GradCAM-based Insertion AUC evaluation for GraspStabilityLSTM (model.py).

Reports per-modality Insertion AUC (tactile, RGB, or both separately).

Usage:
    # Both modalities
    python gradcam_metric.py \
        --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
        --checkpoint trained_models/best_model.pt \
        --gradcam_mode both --L 5 --modalities V T

    # Single modality
    python gradcam_metric.py \
        --root_dir ... --checkpoint ... --gradcam_mode rgb --L 5 --modalities V

Metrics:
    Insertion AUC — higher is better (revealing important pixels restores confidence)

Outputs (saved to --vis_dir):
    gradcam_grid.png   — overlay grid per sample
    curves.png         — mean ± std insertion confidence curves
"""

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import dataloader as _dl
from dataloader import PoseItDataset, split_by_object
from model import GraspStabilityLSTM


# ---------------------------------------------------------------------------
# Wrapper: lets GradCAM feed (1, 3, H, W) into a model expecting full sequences
# ---------------------------------------------------------------------------

class LSTMFrameWrapper(nn.Module):
    """
    Holds a full sequence context; substitutes a single frame at timestep t
    so GradCAM can treat the model as a function of that one frame.
    """
    def __init__(self, model: GraspStabilityLSTM,
                 full_tactile, full_rgb, ft, gripper, gripper_force,
                 t: int, modality: str = 'tactile'):
        super().__init__()
        self.model           = model
        self.full_tactile    = full_tactile    # (1, T, F1, 3, H, W)
        self.full_rgb        = full_rgb        # (1, T, F1, 3, H, W)
        self.ft              = ft              # (1, T, FT_DIM)
        self.gripper         = gripper         # (1, T, GR_DIM)
        self.gripper_force   = gripper_force   # (1, 1)
        self.t               = t
        self.modality        = modality

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img: (1, 3, H, W) — the frame being probed by GradCAM
        tac = self.full_tactile.clone()
        rgb = self.full_rgb.clone()
        if self.modality == 'tactile':
            tac[:, self.t, 0] = img
        else:
            rgb[:, self.t, 0] = img
        # cuDNN RNN backward requires training mode
        self.model.lstm.train()
        out = self.model(tac, rgb, self.ft, self.gripper, self.gripper_force)  # (1, 1)
        self.model.lstm.eval()
        return out


# ---------------------------------------------------------------------------
# Confidence helper (BCEWithLogitsLoss: single logit output)
# ---------------------------------------------------------------------------

def _conf(logit: torch.Tensor, target_class: int) -> float:
    """Convert raw BCEWithLogits logit (1,1) to P(target_class)."""
    p = torch.sigmoid(logit).flatten()[0].item()
    return p if target_class == 1 else 1.0 - p


# ---------------------------------------------------------------------------
# GradCAM heatmap for a single frame
# ---------------------------------------------------------------------------

def gradcam_heatmap(wrapper: LSTMFrameWrapper, frame_tensor: torch.Tensor,
                    target_layer, target_class: int) -> np.ndarray:
    """
    frame_tensor: (1, 3, H, W) float32 on the correct device
    Returns: (H, W) numpy array in [0, 1]
    """
    sign = 1.0 if target_class == 1 else -1.0
    targets = [lambda out, s=sign: s * out.flatten()[0]]
    with GradCAM(model=wrapper, target_layers=[target_layer]) as cam:
        heatmap = cam(input_tensor=frame_tensor, targets=targets)  # (1, H, W)
    return heatmap[0]                                               # (H, W)


# ---------------------------------------------------------------------------
# Single-modality Insertion AUC
# ---------------------------------------------------------------------------

def insertion_auc_single(model: GraspStabilityLSTM,
                         tac: torch.Tensor, rgb: torch.Tensor,
                         ft: torch.Tensor, gripper: torch.Tensor, gripper_force: torch.Tensor,
                         t: int, heatmap: np.ndarray, frame_np: np.ndarray,
                         modality: str, steps: int, target_class: int):
    """
    Blur→reveal for a single modality while keeping the other unchanged.
    Returns: (auc float, confidences list of length steps+1)
    """
    H, W = heatmap.shape
    order = heatmap.flatten().argsort()[::-1]
    n_pixels = order.size
    device = tac.device

    blurred_np = cv2.GaussianBlur(frame_np, (51, 51), 0)

    # Clone full sequences; replace frame t with blurred for the target modality
    tac_seq = tac.clone()
    rgb_seq = rgb.clone()
    if modality == 'tactile':
        tac_seq[:, t, 0] = torch.from_numpy(blurred_np).permute(2, 0, 1).to(device)
        orig = tac[:, t, 0]  # (1, 3, H, W)
    else:
        rgb_seq[:, t, 0] = torch.from_numpy(blurred_np).permute(2, 0, 1).to(device)
        orig = rgb[:, t, 0]

    confidences = []
    with torch.no_grad():
        for step in range(steps + 1):
            k = int(step / steps * n_pixels)
            rev_tac = tac_seq.clone()
            rev_rgb = rgb_seq.clone()
            if k > 0:
                mask = np.zeros(n_pixels, dtype=np.float32)
                mask[order[:k]] = 1.0
                mask2d = torch.from_numpy(mask.reshape(H, W)).to(device)
                if modality == 'tactile':
                    rev_tac[:, t, 0, :, mask2d == 1] = orig[:, :, mask2d == 1]
                else:
                    rev_rgb[:, t, 0, :, mask2d == 1] = orig[:, :, mask2d == 1]
            logit = model(rev_tac, rev_rgb, ft, gripper, gripper_force)
            confidences.append(_conf(logit, target_class))

    return float(np.trapezoid(confidences, dx=1.0 / steps)), confidences


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def save_overlay_grid(vis_items: list, gradcam_mode: str, save_path: str):
    """
    Saves a grid of GradCAM overlays. Columns adapt to gradcam_mode.
    """
    n = len(vis_items)
    if gradcam_mode == 'both':
        ncols = 4  # tac | tac overlay | rgb | rgb overlay
    else:
        ncols = 2  # original | overlay

    _, axes = plt.subplots(n, ncols, figsize=(3 * ncols, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, item in enumerate(vis_items):
        label_str = 'slip/drop' if item['label'] == 1 else 'pass'

        if gradcam_mode == 'both':
            tac_overlay = show_cam_on_image(item['tac_frame_np'], item['tac_heatmap'], use_rgb=True)
            rgb_overlay = show_cam_on_image(item['rgb_frame_np'], item['rgb_heatmap'], use_rgb=True)

            axes[row, 0].imshow(item['tac_frame_np'])
            axes[row, 0].set_title(f"Tactile | {item['idx']} | {label_str}", fontsize=8)
            axes[row, 0].axis('off')
            axes[row, 1].imshow(tac_overlay)
            axes[row, 1].set_title(f"Tac ins={item['tac_ins']:.3f}", fontsize=8)
            axes[row, 1].axis('off')
            axes[row, 2].imshow(item['rgb_frame_np'])
            axes[row, 2].set_title(f"RGB | {item['idx']} | {label_str}", fontsize=8)
            axes[row, 2].axis('off')
            axes[row, 3].imshow(rgb_overlay)
            axes[row, 3].set_title(f"RGB ins={item['rgb_ins']:.3f}", fontsize=8)
            axes[row, 3].axis('off')

        elif gradcam_mode == 'tactile':
            overlay = show_cam_on_image(item['tac_frame_np'], item['tac_heatmap'], use_rgb=True)
            axes[row, 0].imshow(item['tac_frame_np'])
            axes[row, 0].set_title(f"Tactile | {item['idx']} | {label_str}", fontsize=8)
            axes[row, 0].axis('off')
            axes[row, 1].imshow(overlay)
            axes[row, 1].set_title(f"Tac ins={item['tac_ins']:.3f}", fontsize=8)
            axes[row, 1].axis('off')

        else:  # rgb
            overlay = show_cam_on_image(item['rgb_frame_np'], item['rgb_heatmap'], use_rgb=True)
            axes[row, 0].imshow(item['rgb_frame_np'])
            axes[row, 0].set_title(f"RGB | {item['idx']} | {label_str}", fontsize=8)
            axes[row, 0].axis('off')
            axes[row, 1].imshow(overlay)
            axes[row, 1].set_title(f"RGB ins={item['rgb_ins']:.3f}", fontsize=8)
            axes[row, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved overlay grid → {save_path}")


def save_curves_plot(curves_by_label: dict, steps: int, save_path: str):
    """
    curves_by_label: {'Tactile': [...], 'RGB': [...]} — each value is list of curves
    """
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
    ax.set_title('Insertion AUC curves (mean ± std)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved curves plot    → {save_path}")


# ---------------------------------------------------------------------------
# Evaluate over a dataset split
# ---------------------------------------------------------------------------

def evaluate_gradcam(model: GraspStabilityLSTM, dataset: PoseItDataset,
                     indices, gradcam_mode: str, steps: int,
                     n_samples: int, n_vis: int, vis_dir: str, device: str):
    """
    Runs per-modality Insertion AUC on n_samples from the given index set.
    gradcam_mode: 'rgb', 'tactile', or 'both'
    Returns dict with keys like 'tac_ins', 'rgb_ins' (whichever are active).
    """
    model.eval()

    do_tac = gradcam_mode in ('tactile', 'both')
    do_rgb = gradcam_mode in ('rgb', 'both')

    # Unfreeze target layers for GradCAM gradient computation
    if do_tac:
        tac_target_layer = model.tactile_encoder.layer4[-1]
        for p in tac_target_layer.parameters():
            p.requires_grad_(True)
    if do_rgb:
        rgb_target_layer = model.rgb_encoder.layer4[-1]
        for p in rgb_target_layer.parameters():
            p.requires_grad_(True)

    rng = np.random.default_rng(42)
    chosen = rng.choice(indices, size=min(n_samples, len(indices)), replace=False)

    tac_ins_aucs, rgb_ins_aucs = [], []
    tac_curves, rgb_curves = [], []
    vis_items = []

    for idx in chosen:
        sample = dataset[idx]
        tactile, rgb, ft, gripper, gripper_force, label, _ = sample
        target_class = int(label.item())

        tac = tactile.unsqueeze(0).to(device)
        rgb_ = rgb.unsqueeze(0).to(device)
        ft_  = ft.unsqueeze(0).to(device)
        grip = gripper.unsqueeze(0).to(device)
        gf   = gripper_force.unsqueeze(0).to(device)

        T = tac.shape[1]
        mid_t = T // 2
        sample_tac_ins, sample_rgb_ins = [], []
        mid_vis = {}

        for t in range(T):
            if do_tac:
                tac_frame_np = tac[0, t, 0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
                tac_frame_np = np.clip(tac_frame_np, 0.0, 1.0)
                tac_frame_tensor = torch.from_numpy(tac_frame_np).permute(2, 0, 1).unsqueeze(0).to(device)

                tac_wrapper = LSTMFrameWrapper(model, tac, rgb_, ft_, grip, gf, t, 'tactile')
                tac_wrapper.eval()
                tac_heatmap = gradcam_heatmap(tac_wrapper, tac_frame_tensor, tac_target_layer, target_class)

                ti, ti_curve = insertion_auc_single(
                    model, tac, rgb_, ft_, grip, gf, t,
                    tac_heatmap, tac_frame_np, 'tactile', steps, target_class)
                sample_tac_ins.append(ti)
                tac_curves.append(ti_curve)

                if t == mid_t:
                    mid_vis['tac_frame_np'] = tac_frame_np
                    mid_vis['tac_heatmap']  = tac_heatmap

            if do_rgb:
                rgb_frame_np = rgb_[0, t, 0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
                rgb_frame_np = np.clip(rgb_frame_np, 0.0, 1.0)
                rgb_frame_tensor = torch.from_numpy(rgb_frame_np).permute(2, 0, 1).unsqueeze(0).to(device)

                rgb_wrapper = LSTMFrameWrapper(model, tac, rgb_, ft_, grip, gf, t, 'rgb')
                rgb_wrapper.eval()
                rgb_heatmap = gradcam_heatmap(rgb_wrapper, rgb_frame_tensor, rgb_target_layer, target_class)

                ri, ri_curve = insertion_auc_single(
                    model, tac, rgb_, ft_, grip, gf, t,
                    rgb_heatmap, rgb_frame_np, 'rgb', steps, target_class)
                sample_rgb_ins.append(ri)
                rgb_curves.append(ri_curve)

                if t == mid_t:
                    mid_vis['rgb_frame_np'] = rgb_frame_np
                    mid_vis['rgb_heatmap']  = rgb_heatmap

        # Per-sample means
        s_tac = float(np.mean(sample_tac_ins)) if sample_tac_ins else 0.0
        s_rgb = float(np.mean(sample_rgb_ins)) if sample_rgb_ins else 0.0
        if do_tac: tac_ins_aucs.append(s_tac)
        if do_rgb: rgb_ins_aucs.append(s_rgb)

        parts = [f"sample {idx:4d} | label={target_class}"]
        if do_tac: parts.append(f"tac_ins={s_tac:.3f}")
        if do_rgb: parts.append(f"rgb_ins={s_rgb:.3f}")
        print("  " + "  ".join(parts))

        if len(vis_items) < n_vis:
            vis_item = {'idx': idx, 'label': target_class,
                        'tac_ins': s_tac, 'rgb_ins': s_rgb}
            vis_item.update(mid_vis)
            vis_items.append(vis_item)

    # Save visualizations
    if vis_dir and vis_items:
        os.makedirs(vis_dir, exist_ok=True)
        save_overlay_grid(vis_items, gradcam_mode,
                          os.path.join(vis_dir, 'gradcam_grid.png'))
        curves_dict = {}
        if tac_curves: curves_dict['Tactile'] = tac_curves
        if rgb_curves: curves_dict['RGB'] = rgb_curves
        if curves_dict:
            save_curves_plot(curves_dict, steps,
                             os.path.join(vis_dir, 'curves.png'))

    results = {}
    if do_tac: results['tac_ins'] = float(np.mean(tac_ins_aucs))
    if do_rgb: results['rgb_ins'] = float(np.mean(rgb_ins_aucs))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root_dir',    required=True)
    p.add_argument('--checkpoint',  required=True,
                   help='Path to best_model.pt checkpoint')
    p.add_argument('--gradcam_mode', default='both',
                   choices=['rgb', 'tactile', 'both'],
                   help='Which modality to compute GradCAM for')
    p.add_argument('--split',       default='random',
                   choices=['object', 'pose', 'random'])
    p.add_argument('--test_objects', nargs='+',
                   default=['mug', 'bowl', 'flashlight'])
    p.add_argument('--L',           type=int, default=20)
    p.add_argument('--F1',          type=int, default=1)
    p.add_argument('--F2',          type=int, default=1)
    p.add_argument('--hidden_dim',  type=int, default=256)
    p.add_argument('--lstm_layers', type=int, default=2)
    p.add_argument('--dropout',     type=float, default=0.1)
    p.add_argument('--modalities',  nargs='+', default=None,
                   help='Active modalities, e.g. --modalities V T FT (default: all)')
    p.add_argument('--steps',       type=int, default=10,
                   help='Number of masking steps for insertion curves')
    p.add_argument('--n_samples',   type=int, default=50,
                   help='Number of test samples to evaluate')
    p.add_argument('--n_vis',       type=int, default=8,
                   help='Number of samples to include in the overlay grid')
    p.add_argument('--vis_dir',     type=str, default='gradcam_vis',
                   help='Directory to save visualizations (set to "" to skip)')
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}  steps: {args.steps}  gradcam_mode: {args.gradcam_mode}")

    _dl.L  = args.L
    _dl.F1 = args.F1
    _dl.F2 = args.F2

    ds = PoseItDataset(root_dir=args.root_dir)

    if args.split == 'object':
        _, _, test_set = split_by_object(ds, test_objects=args.test_objects)
    elif args.split == 'pose':
        from dataloader import split_by_pose
        _, _, test_set = split_by_pose(ds)
    else:
        from dataloader import uniform_random_split
        _, _, test_set = uniform_random_split(ds)

    model = GraspStabilityLSTM(
        frames_per_sec=args.F1,
        ft_dim=_dl.FT_DIM,
        gripper_dim=_dl.GR_DIM,
        hidden_dim=args.hidden_dim,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        freeze_resnet=True,
        modalities=args.modalities,
    ).to(device)

    assert os.path.exists(args.checkpoint), f"Checkpoint not found: {args.checkpoint}"
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Evaluating {min(args.n_samples, len(test_set.indices))} test samples...\n")

    results = evaluate_gradcam(
        model, ds, test_set.indices, args.gradcam_mode,
        args.steps, args.n_samples, args.n_vis,
        args.vis_dir or None, device)

    print(f"\n{'='*50}")
    if 'tac_ins' in results:
        print(f"Tactile Insertion AUC (↑ better): {results['tac_ins']:.4f}")
    if 'rgb_ins' in results:
        print(f"RGB     Insertion AUC (↑ better): {results['rgb_ins']:.4f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
