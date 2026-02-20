"""
Check DRS sampler balancing on a small slice of real data.

Usage:
    python check_balance.py --root /shared/dataset/Gelsight/ --n_samples 20
"""

import argparse
import numpy as np
from torch.utils.data import DataLoader

from dataloader import PoseItDataset
from sampler import DRSSampler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root',      default='/shared/dataset/Gelsight/')
    p.add_argument('--n_samples', type=int, default=20)   # how many folders to load
    p.add_argument('--sigma',     type=float, default=1.0)
    p.add_argument('--batch_size',type=int,   default=10)
    p.add_argument('--n_batches', type=int,   default=100)  # batches to sample for ratio check
    return p.parse_args()


def main():
    args = parse_args()

    # Load only first n_samples folders for speed
    from pathlib import Path
    all_dirs = sorted(Path(args.root).iterdir())
    dirs = [str(d) for d in all_dirs if d.is_dir()][:args.n_samples]
    print(f"Loading {len(dirs)} sample folders...\n")

    ds = PoseItDataset(sample_dirs=dirs)

    if len(ds) == 0:
        print("No samples loaded — check your path or stages.csv keys.")
        return

    # ---- Print per-sample labels -------------------------------------------
    print(f"{'idx':<5} {'object':<30} {'label (shake)':<15} {'pose_label':<12} {'S= or S≠'}")
    print("-" * 75)
    s_eq_count = s_neq_count = 0
    for i, s in enumerate(ds.samples):
        label      = int(s['label'])
        pose_label = int(s['pose_label'])
        group      = 'S=' if label == pose_label else 'S≠'
        if group == 'S=': s_eq_count  += 1
        else:             s_neq_count += 1
        obj = s['object']
        print(f"{i:<5} {obj:<30} {label:<15} {pose_label:<12} {group}")

    print("-" * 75)
    print(f"Total: {len(ds.samples)}  |  S=: {s_eq_count}  |  S≠: {s_neq_count}")
    r = s_neq_count / max(s_eq_count, 1)
    print(f"r = |S≠|/|S=| = {r:.4f}\n")

    if s_neq_count == 0:
        print("All samples have matching pose/shake labels (S≠ is empty).")
        print("DRS has nothing to balance — try loading more samples.")
        return

    if args.sigma <= r:
        print(f"WARNING: sigma ({args.sigma}) <= r ({r:.4f}). Increase --sigma.")
        return

    # ---- Sampler ratio check -----------------------------------------------
    sampler = DRSSampler(ds, sigma=args.sigma, batch_size=args.batch_size, seed=42)
    sampler.activate()

    s_neq_set = set(sampler.s_neq.tolist())
    ratios, eq_counts, neq_counts = [], [], []

    for _ in range(args.n_batches):
        batch = sampler._sample_batch()
        n_neq = sum(1 for i in batch if i in s_neq_set)
        n_eq  = len(batch) - n_neq
        eq_counts.append(n_eq)
        neq_counts.append(n_neq)
        if n_eq > 0:
            ratios.append(n_neq / n_eq)

    print(f"Sampler stats over {args.n_batches} batches (sigma={args.sigma}):")
    print(f"  Avg batch size : {np.mean([e+n for e,n in zip(eq_counts,neq_counts)]):.1f}  "
          f"(pre-DRS: {args.batch_size})")
    print(f"  Avg S=  per batch : {np.mean(eq_counts):.1f}")
    print(f"  Avg S≠  per batch : {np.mean(neq_counts):.1f}")
    print(f"  Avg S≠/S= ratio   : {np.mean(ratios):.4f}  (target sigma={args.sigma})")
    print(f"  Std S≠/S= ratio   : {np.std(ratios):.4f}")


if __name__ == '__main__':
    main()