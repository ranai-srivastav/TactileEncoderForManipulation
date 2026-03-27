"""
GradCAM-based Deletion/Insertion AUC evaluation for GraspStabilityLSTM (model.py).

Computes GradCAM heatmaps from BOTH vision encoders (tactile + RGB) and performs
deletion/insertion by masking both modalities simultaneously.

Usage:
    python gradcam_metric.py \
        --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
        --checkpoint trained_models/best_model.pt \
        --steps 10 \
        --n_samples 50 \
        --vis_dir gradcam_vis

Metrics:
    Deletion AUC  — lower is better  (masking important pixels collapses confidence)
    Insertion AUC — higher is better (revealing important pixels restores confidence)
    Delta = Insertion - Deletion     — higher is better, primary reported number

Outputs (saved to --vis_dir):
    gradcam_grid.png   — grid of [tactile | tac overlay | rgb | rgb overlay] per sample
    curves.png         — mean ± std deletion/insertion confidence curves
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
        return self.model(tac, rgb, self.ft, self.gripper, self.gripper_force)  # (1, 1)


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

    For class 1 (slip/drop): maximize logit.
    For class 0 (pass): maximize −logit (= minimize logit).
    """
    sign = 1.0 if target_class == 1 else -1.0
    targets = [lambda out, s=sign: s * out.flatten()[0]]
    with GradCAM(model=wrapper, target_layers=[target_layer]) as cam:
        heatmap = cam(input_tensor=frame_tensor, targets=targets)  # (1, H, W)
    return heatmap[0]                                               # (H, W)


# ---------------------------------------------------------------------------
# Multimodal Deletion AUC
# ---------------------------------------------------------------------------

def deletion_auc(model: GraspStabilityLSTM,
                 tac: torch.Tensor, rgb: torch.Tensor,
                 ft: torch.Tensor, gripper: torch.Tensor, gripper_force: torch.Tensor,
                 t: int,
                 tac_heatmap: np.ndarray, rgb_heatmap: np.ndarray,
                 steps: int, target_class: int):
    """
    Progressively zero out the most important pixels in BOTH modalities
    (each ranked by its own heatmap) simultaneously.
    Returns: (auc float, confidences list of length steps+1)
    """
    H, W = tac_heatmap.shape
    tac_order = tac_heatmap.flatten().argsort()[::-1]
    rgb_order = rgb_heatmap.flatten().argsort()[::-1]
    n_pixels = tac_order.size

    confidences = []
    with torch.no_grad():
        for step in range(steps + 1):
            k = int(step / steps * n_pixels)
            masked_tac = tac.clone()
            masked_rgb = rgb.clone()
            if k > 0:
                # Tactile mask
                tac_mask = np.zeros(n_pixels, dtype=np.float32)
                tac_mask[tac_order[:k]] = 1.0
                tac_mask2d = torch.from_numpy(tac_mask.reshape(H, W)).to(tac.device)
                masked_tac[:, t, 0, :, tac_mask2d == 1] = 0.0

                # RGB mask
                rgb_mask = np.zeros(n_pixels, dtype=np.float32)
                rgb_mask[rgb_order[:k]] = 1.0
                rgb_mask2d = torch.from_numpy(rgb_mask.reshape(H, W)).to(rgb.device)
                masked_rgb[:, t, 0, :, rgb_mask2d == 1] = 0.0

            logit = model(masked_tac, masked_rgb, ft, gripper, gripper_force)
            confidences.append(_conf(logit, target_class))

    return float(np.trapezoid(confidences, dx=1.0 / steps)), confidences


# ---------------------------------------------------------------------------
# Multimodal Insertion AUC
# ---------------------------------------------------------------------------

def insertion_auc(model: GraspStabilityLSTM,
                  tac: torch.Tensor, rgb: torch.Tensor,
                  ft: torch.Tensor, gripper: torch.Tensor, gripper_force: torch.Tensor,
                  t: int,
                  tac_heatmap: np.ndarray, rgb_heatmap: np.ndarray,
                  tac_frame_np: np.ndarray, rgb_frame_np: np.ndarray,
                  steps: int, target_class: int):
    """
    Start from blurred baselines for both modalities; progressively reveal
    the most important pixels in each (ranked by respective heatmaps).
    Returns: (auc float, confidences list of length steps+1)
    """
    H, W = tac_heatmap.shape
    tac_order = tac_heatmap.flatten().argsort()[::-1]
    rgb_order = rgb_heatmap.flatten().argsort()[::-1]
    n_pixels = tac_order.size
    device = tac.device

    # Build blurred baselines
    tac_blurred_np = cv2.GaussianBlur(tac_frame_np, (51, 51), 0)
    rgb_blurred_np = cv2.GaussianBlur(rgb_frame_np, (51, 51), 0)

    # Create baseline sequences (clone full sequence, replace frame t with blurred)
    tac_baseline = tac.clone()
    tac_baseline[:, t, 0] = torch.from_numpy(tac_blurred_np).permute(2, 0, 1).to(device)
    rgb_baseline = rgb.clone()
    rgb_baseline[:, t, 0] = torch.from_numpy(rgb_blurred_np).permute(2, 0, 1).to(device)

    # Original frame tensors for revealing
    tac_orig = tac[:, t, 0]  # (1, 3, H, W)
    rgb_orig = rgb[:, t, 0]

    confidences = []
    with torch.no_grad():
        for step in range(steps + 1):
            k = int(step / steps * n_pixels)
            revealed_tac = tac_baseline.clone()
            revealed_rgb = rgb_baseline.clone()
            if k > 0:
                tac_mask = np.zeros(n_pixels, dtype=np.float32)
                tac_mask[tac_order[:k]] = 1.0
                tac_mask2d = torch.from_numpy(tac_mask.reshape(H, W)).to(device)
                revealed_tac[:, t, 0, :, tac_mask2d == 1] = tac_orig[:, :, tac_mask2d == 1]

                rgb_mask = np.zeros(n_pixels, dtype=np.float32)
                rgb_mask[rgb_order[:k]] = 1.0
                rgb_mask2d = torch.from_numpy(rgb_mask.reshape(H, W)).to(device)
                revealed_rgb[:, t, 0, :, rgb_mask2d == 1] = rgb_orig[:, :, rgb_mask2d == 1]

            logit = model(revealed_tac, revealed_rgb, ft, gripper, gripper_force)
            confidences.append(_conf(logit, target_class))

    return float(np.trapezoid(confidences, dx=1.0 / steps)), confidences


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def save_overlay_grid(vis_items: list, save_path: str):
    """
    vis_items: list of dicts with keys:
        idx, label, tac_frame_np, rgb_frame_np, tac_heatmap, rgb_heatmap, del_auc, ins_auc
    Saves a grid: each row = one sample, cols = [tactile | tac overlay | rgb | rgb overlay]
    """
    n = len(vis_items)
    _, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, item in enumerate(vis_items):
        tac_overlay = show_cam_on_image(item['tac_frame_np'], item['tac_heatmap'], use_rgb=True)
        rgb_overlay = show_cam_on_image(item['rgb_frame_np'], item['rgb_heatmap'], use_rgb=True)
        label_str = 'slip/drop' if item['label'] == 1 else 'pass'

        axes[row, 0].imshow(item['tac_frame_np'])
        axes[row, 0].set_title(f"Tactile | sample {item['idx']} | {label_str}", fontsize=8)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(tac_overlay)
        axes[row, 1].set_title(f"Tactile GradCAM", fontsize=8)
        axes[row, 1].axis('off')

        axes[row, 2].imshow(item['rgb_frame_np'])
        axes[row, 2].set_title(f"RGB | sample {item['idx']} | {label_str}", fontsize=8)
        axes[row, 2].axis('off')

        axes[row, 3].imshow(rgb_overlay)
        axes[row, 3].set_title(
            f"RGB GradCAM  del={item['del_auc']:.3f}  ins={item['ins_auc']:.3f}", fontsize=8)
        axes[row, 3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved overlay grid → {save_path}")


def save_curves_plot(all_del_curves: list, all_ins_curves: list,
                     steps: int, save_path: str):
    """
    all_del_curves / all_ins_curves: list of lists, each of length steps+1
    Plots mean ± std for deletion and insertion curves.
    """
    xs = np.linspace(0, 1, steps + 1)
    del_arr = np.array(all_del_curves)   # (N, steps+1)
    ins_arr = np.array(all_ins_curves)

    _, ax = plt.subplots(figsize=(6, 4))
    for arr, color, label in [
        (del_arr, 'tomato',    'Deletion (↓ better)'),
        (ins_arr, 'steelblue', 'Insertion (↑ better)'),
    ]:
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        ax.plot(xs, mean, color=color, label=label, linewidth=2)
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel('Fraction of pixels masked / revealed')
    ax.set_ylabel('Model confidence (target class)')
    ax.set_title('Multimodal Deletion / Insertion curves (mean ± std)')
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
                     indices, steps: int,
                     n_samples: int, n_vis: int, vis_dir: str, device: str):
    """
    Runs multimodal deletion/insertion AUC on n_samples from the given index set.
    Computes GradCAM heatmaps from both tactile and RGB encoders, then masks
    both modalities simultaneously for deletion/insertion.
    Returns mean deletion AUC, mean insertion AUC, mean delta.
    """
    model.eval()

    # Target layers: layer4[-1] of both ResNet encoders
    tac_target_layer = model.tactile_encoder.layer4[-1]
    rgb_target_layer = model.rgb_encoder.layer4[-1]
    # Unfreeze layer4 so GradCAM can compute gradients through them
    for p in tac_target_layer.parameters():
        p.requires_grad_(True)
    for p in rgb_target_layer.parameters():
        p.requires_grad_(True)

    rng = np.random.default_rng(42)
    chosen = rng.choice(indices, size=min(n_samples, len(indices)), replace=False)

    del_aucs, ins_aucs = [], []
    all_del_curves, all_ins_curves = [], []
    vis_items = []

    for idx in chosen:
        sample = dataset[idx]
        # 7-tuple: (tactile, rgb, ft, gripper, gripper_force, label, pose_label)
        tactile, rgb, ft, gripper, gripper_force, label, _ = sample
        target_class = int(label.item())

        # Add batch dim and move to device
        tac = tactile.unsqueeze(0).to(device)       # (1, T, F1, 3, H, W)
        rgb_ = rgb.unsqueeze(0).to(device)           # (1, T, F1, 3, H, W)
        ft_  = ft.unsqueeze(0).to(device)            # (1, T, FT_DIM)
        grip = gripper.unsqueeze(0).to(device)       # (1, T, GR_DIM)
        gf   = gripper_force.unsqueeze(0).to(device) # (1, 1)

        T = tac.shape[1]
        frame_del_aucs, frame_ins_aucs = [], []
        mid_t = T // 2
        mid_tac_heatmap = None
        mid_rgb_heatmap = None
        mid_tac_frame_np = None
        mid_rgb_frame_np = None

        for t in range(T):
            # Extract frame numpy arrays for both modalities
            tac_frame_np = tac[0, t, 0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
            tac_frame_np = np.clip(tac_frame_np, 0.0, 1.0)
            tac_frame_tensor = torch.from_numpy(tac_frame_np).permute(2, 0, 1).unsqueeze(0).to(device)

            rgb_frame_np = rgb_[0, t, 0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
            rgb_frame_np = np.clip(rgb_frame_np, 0.0, 1.0)
            rgb_frame_tensor = torch.from_numpy(rgb_frame_np).permute(2, 0, 1).unsqueeze(0).to(device)

            # GradCAM heatmap from tactile encoder
            tac_wrapper = LSTMFrameWrapper(model, tac, rgb_, ft_, grip, gf, t, 'tactile')
            tac_wrapper.eval()
            tac_heatmap = gradcam_heatmap(tac_wrapper, tac_frame_tensor, tac_target_layer, target_class)

            # GradCAM heatmap from RGB encoder
            rgb_wrapper = LSTMFrameWrapper(model, tac, rgb_, ft_, grip, gf, t, 'rgb')
            rgb_wrapper.eval()
            rgb_heatmap = gradcam_heatmap(rgb_wrapper, rgb_frame_tensor, rgb_target_layer, target_class)

            # Multimodal deletion/insertion: mask both modalities simultaneously
            d, d_curve = deletion_auc(
                model, tac, rgb_, ft_, grip, gf, t,
                tac_heatmap, rgb_heatmap,
                steps, target_class)
            i, i_curve = insertion_auc(
                model, tac, rgb_, ft_, grip, gf, t,
                tac_heatmap, rgb_heatmap,
                tac_frame_np, rgb_frame_np,
                steps, target_class)

            frame_del_aucs.append(d)
            frame_ins_aucs.append(i)
            all_del_curves.append(d_curve)
            all_ins_curves.append(i_curve)

            if t == mid_t:
                mid_tac_heatmap  = tac_heatmap
                mid_rgb_heatmap  = rgb_heatmap
                mid_tac_frame_np = tac_frame_np
                mid_rgb_frame_np = rgb_frame_np

        sample_del = float(np.mean(frame_del_aucs))
        sample_ins = float(np.mean(frame_ins_aucs))
        del_aucs.append(sample_del)
        ins_aucs.append(sample_ins)
        print(f"  sample {idx:4d} | label={target_class} | "
              f"del={sample_del:.3f}  ins={sample_ins:.3f}  "
              f"delta={sample_ins - sample_del:.3f}")

        if len(vis_items) < n_vis:
            vis_items.append({
                'idx':          idx,
                'label':        target_class,
                'tac_frame_np': mid_tac_frame_np,
                'rgb_frame_np': mid_rgb_frame_np,
                'tac_heatmap':  mid_tac_heatmap,
                'rgb_heatmap':  mid_rgb_heatmap,
                'del_auc':      sample_del,
                'ins_auc':      sample_ins,
            })

    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
        save_overlay_grid(vis_items,
                          os.path.join(vis_dir, 'gradcam_grid.png'))
        save_curves_plot(all_del_curves, all_ins_curves, steps,
                         os.path.join(vis_dir, 'curves.png'))

    mean_del   = float(np.mean(del_aucs))
    mean_ins   = float(np.mean(ins_aucs))
    mean_delta = mean_ins - mean_del
    return mean_del, mean_ins, mean_delta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root_dir',    required=True)
    p.add_argument('--checkpoint',  required=True,
                   help='Path to best_model.pt checkpoint')
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
                   help='Number of masking steps for deletion/insertion curves')
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
    print(f"Device: {device}  steps: {args.steps}  (multimodal: tactile + RGB)")

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

    mean_del, mean_ins, mean_delta = evaluate_gradcam(
        model, ds, test_set.indices,
        args.steps, args.n_samples, args.n_vis,
        args.vis_dir or None, device)

    print(f"\n{'='*50}")
    print(f"Deletion  AUC (↓ better): {mean_del:.4f}")
    print(f"Insertion AUC (↑ better): {mean_ins:.4f}")
    print(f"Delta = Ins - Del (↑ better): {mean_delta:.4f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
