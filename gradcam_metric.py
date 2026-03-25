"""
GradCAM-based Deletion/Insertion AUC evaluation for GraspClassifier (unimodal_2d.py).

Usage:
    python gradcam_metric.py \
        --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
        --modality rgb \
        --checkpoint trained_models/best_2d.pt \
        --steps 10 \
        --n_samples 50

Metrics:
    Deletion AUC  — lower is better  (masking important pixels collapses confidence)
    Insertion AUC — higher is better (revealing important pixels restores confidence)
    Delta = Insertion - Deletion     — higher is better, primary reported number
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'unimodal'))

import dataloader as _dl
from dataloader import PoseItDataset, split_by_object
from unimodal_2d import GraspClassifier


# ---------------------------------------------------------------------------
# Wrapper: lets GradCAM feed (B, 3, H, W) into a model expecting (B,T,F1,3,H,W)
# ---------------------------------------------------------------------------

class SingleFrameWrapper(nn.Module):
    def __init__(self, model: GraspClassifier):
        super().__init__()
        self.model = model

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img: (B, 3, H, W)
        B = img.shape[0]
        imgs = img.unsqueeze(1).unsqueeze(1)                        # (B, 1, 1, 3, H, W)
        lengths = torch.ones(B, dtype=torch.long, device=img.device)
        return self.model(imgs, lengths)                            # (B, 2)


# ---------------------------------------------------------------------------
# GradCAM heatmap for a single frame
# ---------------------------------------------------------------------------

def gradcam_heatmap(wrapper: SingleFrameWrapper, frame_tensor: torch.Tensor,
                    target_layer, target_class: int) -> np.ndarray:
    """
    frame_tensor: (1, 3, H, W) float32 on the correct device
    Returns: (H, W) numpy array in [0, 1]
    """
    with GradCAM(model=wrapper, target_layers=[target_layer]) as cam:
        targets = [ClassifierOutputTarget(target_class)]
        heatmap = cam(input_tensor=frame_tensor, targets=targets)   # (1, H, W)
    return heatmap[0]                                               # (H, W)


# ---------------------------------------------------------------------------
# Deletion AUC
# ---------------------------------------------------------------------------

def deletion_auc(wrapper: SingleFrameWrapper, frame_tensor: torch.Tensor,
                 heatmap: np.ndarray, steps: int, target_class: int) -> float:
    """
    Progressively zero out the most important pixels (by heatmap rank).
    Measures how fast model confidence drops.
    Lower AUC = model was truly attending to important pixels.
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
            conf = torch.softmax(logit, dim=1)[0, target_class].item()
            confidences.append(conf)

    return float(np.trapz(confidences, dx=1.0 / steps))


# ---------------------------------------------------------------------------
# Insertion AUC
# ---------------------------------------------------------------------------

def insertion_auc(wrapper: SingleFrameWrapper, frame_tensor: torch.Tensor,
                  frame_np: np.ndarray, heatmap: np.ndarray,
                  steps: int, target_class: int) -> float:
    """
    Start from a blurred baseline; progressively reveal the most important pixels.
    Measures how fast model confidence rises.
    Higher AUC = heatmap correctly identifies discriminative regions.
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
            conf = torch.softmax(logit, dim=1)[0, target_class].item()
            confidences.append(conf)

    return float(np.trapz(confidences, dx=1.0 / steps))


# ---------------------------------------------------------------------------
# Evaluate over a dataset split
# ---------------------------------------------------------------------------

def evaluate_gradcam(model: GraspClassifier, dataset: PoseItDataset,
                     indices, modality: str, steps: int,
                     n_samples: int, device: str):
    """
    Runs deletion/insertion AUC on n_samples from the given index set.
    Returns mean deletion AUC, mean insertion AUC, mean delta.
    """
    wrapper = SingleFrameWrapper(model).to(device)
    wrapper.eval()
    target_layer = wrapper.model.backbone.layer4[-1]
    # GradCAM needs gradients through the target layer even when backbone is frozen
    for p in wrapper.model.backbone.layer4.parameters():
        p.requires_grad_(True)

    rng = np.random.default_rng(42)
    chosen = rng.choice(indices, size=min(n_samples, len(indices)), replace=False)

    del_aucs, ins_aucs = [], []

    for idx in chosen:
        sample = dataset[idx]
        # 7-tuple: (tactile, rgb, ft, gripper, gripper_force, label, pose_label)
        tactile, rgb, _, _, _, label, _ = sample
        imgs = tactile if modality == 'tactile' else rgb  # (T, F1, 3, H, W)
        target_class = int(label.item())

        T = imgs.shape[0]
        frame_del_aucs, frame_ins_aucs = [], []

        for t in range(T):
            # frame_np: (H, W, 3) float32 in [0, 1]
            frame_np = imgs[t, 0].permute(1, 2, 0).numpy().astype(np.float32)
            frame_np = np.clip(frame_np, 0.0, 1.0)
            frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0).to(device)

            heatmap = gradcam_heatmap(wrapper, frame_tensor, target_layer, target_class)

            d = deletion_auc(wrapper, frame_tensor, heatmap, steps, target_class)
            i = insertion_auc(wrapper, frame_tensor, frame_np, heatmap, steps, target_class)
            frame_del_aucs.append(d)
            frame_ins_aucs.append(i)

        del_aucs.append(np.mean(frame_del_aucs))
        ins_aucs.append(np.mean(frame_ins_aucs))
        print(f"  sample {idx:4d} | label={target_class} | "
              f"del={del_aucs[-1]:.3f}  ins={ins_aucs[-1]:.3f}  "
              f"delta={ins_aucs[-1]-del_aucs[-1]:.3f}")

    mean_del = float(np.mean(del_aucs))
    mean_ins = float(np.mean(ins_aucs))
    mean_delta = mean_ins - mean_del
    return mean_del, mean_ins, mean_delta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root_dir',    required=True)
    p.add_argument('--modality',    default='rgb', choices=['rgb', 'tactile'])
    p.add_argument('--checkpoint',  required=True,
                   help='Path to best_2d.pt checkpoint')
    p.add_argument('--split',       default='object',
                   choices=['object', 'pose', 'random'])
    p.add_argument('--test_objects', nargs='+',
                   default=['mug', 'bowl', 'flashlight'])
    p.add_argument('--L',           type=int, default=3)
    p.add_argument('--F1',          type=int, default=1)
    p.add_argument('--F2',          type=int, default=1)
    p.add_argument('--steps',       type=int, default=10,
                   help='Number of masking steps for deletion/insertion curves')
    p.add_argument('--n_samples',   type=int, default=50,
                   help='Number of test samples to evaluate')
    p.add_argument('--freeze_backbone', action='store_true', default=True)
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
    else:
        from dataloader import uniform_random_split
        _, _, test_set = uniform_random_split(ds)

    model = GraspClassifier(freeze_backbone=args.freeze_backbone).to(device)
    assert os.path.exists(args.checkpoint), f"Checkpoint not found: {args.checkpoint}"
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Evaluating {min(args.n_samples, len(test_set.indices))} test samples...\n")

    mean_del, mean_ins, mean_delta = evaluate_gradcam(
        model, ds, test_set.indices, args.modality,
        args.steps, args.n_samples, device)

    print(f"\n{'='*50}")
    print(f"Deletion  AUC (↓ better): {mean_del:.4f}")
    print(f"Insertion AUC (↑ better): {mean_ins:.4f}")
    print(f"Delta = Ins - Del (↑ better): {mean_delta:.4f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
