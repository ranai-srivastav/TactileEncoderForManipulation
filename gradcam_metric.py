"""
GradCAM-based Deletion/Insertion AUC evaluation for GraspStabilityLSTM (model.py).

Usage:
    python gradcam_metric.py \
        --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
        --modality rgb \
        --checkpoint trained_models/best_model.pt \
        --steps 10 \
        --n_samples 50 \
        --vis_dir gradcam_vis

Metrics:
    Deletion AUC  — lower is better  (masking important pixels collapses confidence)
    Insertion AUC — higher is better (revealing important pixels restores confidence)
    Delta = Insertion - Deletion     — higher is better, primary reported number

Outputs (saved to --vis_dir):
    gradcam_grid.png   — grid of [original | GradCAM overlay] per sample (first --n_vis)
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
    p = torch.sigmoid(logit)[0, 0].item()
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
    targets = [lambda out, s=sign: s * out[0, 0]]
    with GradCAM(model=wrapper, target_layers=[target_layer]) as cam:
        heatmap = cam(input_tensor=frame_tensor, targets=targets)  # (1, H, W)
    return heatmap[0]                                               # (H, W)


# ---------------------------------------------------------------------------
# Deletion AUC
# ---------------------------------------------------------------------------

def deletion_auc(wrapper: LSTMFrameWrapper, frame_tensor: torch.Tensor,
                 heatmap: np.ndarray, steps: int, target_class: int):
    """
    Progressively zero out the most important pixels (by heatmap rank).
    Returns: (auc float, confidences list of length steps+1)
    """
    H, W = heatmap.shape
    flat_order = heatmap.flatten().argsort()[::-1]   # most → least important
    n_pixels = flat_order.size

    confidences = []
    with torch.no_grad():
        for step in range(steps + 1):
            k = int(step / steps * n_pixels)
            masked = frame_tensor.clone()
            if k > 0:
                mask = np.zeros(n_pixels, dtype=np.float32)
                mask[flat_order[:k]] = 1.0
                mask2d = torch.from_numpy(mask.reshape(H, W)).to(frame_tensor.device)
                masked[:, :, mask2d == 1] = 0.0
            logit = wrapper(masked)
            confidences.append(_conf(logit, target_class))

    return float(np.trapezoid(confidences, dx=1.0 / steps)), confidences


# ---------------------------------------------------------------------------
# Insertion AUC
# ---------------------------------------------------------------------------

def insertion_auc(wrapper: LSTMFrameWrapper, frame_tensor: torch.Tensor,
                  frame_np: np.ndarray, heatmap: np.ndarray,
                  steps: int, target_class: int):
    """
    Start from a blurred baseline; progressively reveal the most important pixels.
    Returns: (auc float, confidences list of length steps+1)
    """
    H, W = heatmap.shape
    flat_order = heatmap.flatten().argsort()[::-1]
    n_pixels = flat_order.size

    blurred_np = cv2.GaussianBlur(frame_np, (51, 51), 0)           # (H, W, 3)
    baseline = torch.from_numpy(blurred_np).permute(2, 0, 1).unsqueeze(0).to(frame_tensor.device)

    confidences = []
    with torch.no_grad():
        for step in range(steps + 1):
            k = int(step / steps * n_pixels)
            revealed = baseline.clone()
            if k > 0:
                mask = np.zeros(n_pixels, dtype=np.float32)
                mask[flat_order[:k]] = 1.0
                mask2d = torch.from_numpy(mask.reshape(H, W)).to(frame_tensor.device)
                revealed[:, :, mask2d == 1] = frame_tensor[:, :, mask2d == 1]
            logit = wrapper(revealed)
            confidences.append(_conf(logit, target_class))

    return float(np.trapezoid(confidences, dx=1.0 / steps)), confidences


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def save_overlay_grid(vis_items: list, save_path: str):
    """
    vis_items: list of dicts with keys:
        idx, label, frame_np (H,W,3 float32), heatmap (H,W float32), del_auc, ins_auc
    Saves a grid: each row = one sample, cols = [original, overlay]
    """
    n = len(vis_items)
    _, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, item in enumerate(vis_items):
        overlay = show_cam_on_image(item['frame_np'], item['heatmap'], use_rgb=True)
        label_str = 'slip/drop' if item['label'] == 1 else 'pass'
        axes[row, 0].imshow(item['frame_np'])
        axes[row, 0].set_title(f"sample {item['idx']} | {label_str}", fontsize=8)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(
            f"GradCAM  del={item['del_auc']:.3f}  ins={item['ins_auc']:.3f}", fontsize=8)
        axes[row, 1].axis('off')

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
    ax.set_title('Deletion / Insertion curves (mean ± std)')
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
                     indices, modality: str, steps: int,
                     n_samples: int, n_vis: int, vis_dir: str, device: str):
    """
    Runs deletion/insertion AUC on n_samples from the given index set.
    Saves GradCAM overlay grid and curve plot to vis_dir.
    Returns mean deletion AUC, mean insertion AUC, mean delta.
    """
    model.eval()

    # Target layer: layer4[-1] of the relevant ResNet encoder
    encoder = model.tactile_encoder if modality == 'tactile' else model.rgb_encoder
    target_layer = encoder.layer4[-1]
    # Unfreeze layer4 so GradCAM can compute gradients through it
    for p in target_layer.parameters():
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
        rgb_ = rgb.unsqueeze(0).to(device)          # (1, T, F1, 3, H, W)
        ft_  = ft.unsqueeze(0).to(device)           # (1, T, FT_DIM)
        grip = gripper.unsqueeze(0).to(device)      # (1, T, GR_DIM)
        gf   = gripper_force.unsqueeze(0).to(device)  # (1, 1)

        T = tac.shape[1]
        frame_del_aucs, frame_ins_aucs = [], []
        mid_t = T // 2
        mid_heatmap = None
        mid_frame_np = None

        imgs = tac if modality == 'tactile' else rgb_

        for t in range(T):
            frame_np = imgs[0, t, 0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
            frame_np = np.clip(frame_np, 0.0, 1.0)
            frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0).to(device)

            wrapper = LSTMFrameWrapper(model, tac, rgb_, ft_, grip, gf, t, modality)
            wrapper.eval()

            heatmap = gradcam_heatmap(wrapper, frame_tensor, target_layer, target_class)

            d, d_curve = deletion_auc(wrapper, frame_tensor, heatmap, steps, target_class)
            i, i_curve = insertion_auc(wrapper, frame_tensor, frame_np, heatmap, steps, target_class)
            frame_del_aucs.append(d)
            frame_ins_aucs.append(i)
            all_del_curves.append(d_curve)
            all_ins_curves.append(i_curve)

            if t == mid_t:
                mid_heatmap  = heatmap
                mid_frame_np = frame_np

        sample_del = float(np.mean(frame_del_aucs))
        sample_ins = float(np.mean(frame_ins_aucs))
        del_aucs.append(sample_del)
        ins_aucs.append(sample_ins)
        print(f"  sample {idx:4d} | label={target_class} | "
              f"del={sample_del:.3f}  ins={sample_ins:.3f}  "
              f"delta={sample_ins - sample_del:.3f}")

        if len(vis_items) < n_vis:
            vis_items.append({
                'idx':      idx,
                'label':    target_class,
                'frame_np': mid_frame_np,
                'heatmap':  mid_heatmap,
                'del_auc':  sample_del,
                'ins_auc':  sample_ins,
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
    p.add_argument('--modality',    default='tactile', choices=['rgb', 'tactile'])
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
    print(f"Device: {device}  modality: {args.modality}  steps: {args.steps}")

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
        model, ds, test_set.indices, args.modality,
        args.steps, args.n_samples, args.n_vis,
        args.vis_dir or None, device)

    print(f"\n{'='*50}")
    print(f"Deletion  AUC (↓ better): {mean_del:.4f}")
    print(f"Insertion AUC (↑ better): {mean_ins:.4f}")
    print(f"Delta = Ins - Del (↑ better): {mean_delta:.4f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
