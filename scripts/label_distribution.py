#!/usr/bin/env python3
"""Print binary label counts for a PoseIt dataset."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import dataloader as _dl
from dataloader import PoseItDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Report stable/non-stable label counts for a PoseIt dataset."
    )
    parser.add_argument(
        "--root_dir",
        required=True,
        help="Dataset root containing PoseIt sample directories.",
    )
    parser.add_argument(
        "--L",
        type=int,
        default=9,
        help="Max seconds per episode passed to the dataset loader.",
    )
    parser.add_argument(
        "--load_images",
        action="store_true",
        help="Enable GelSight/RGB image loading. Disabled by default for speed.",
    )
    return parser.parse_args()


def _pct(count: int, total: int) -> float:
    return (100.0 * count / total) if total > 0 else 0.0


def main():
    args = parse_args()
    _dl.L = args.L

    dataset = PoseItDataset(
        root_dir=args.root_dir,
        load_tactile=args.load_images,
        load_rgb=args.load_images,
    )

    total = len(dataset.samples)
    stable = sum(int(sample["label"].item() == 0) for sample in dataset.samples)
    non_stable = total - stable

    print(f"image_loading={'enabled' if args.load_images else 'disabled'}")
    print(f"total loaded samples: {total}")
    print(f"stable count (label==0): {stable} ({_pct(stable, total):.2f}%)")
    print(f"non-stable count (label==1): {non_stable} ({_pct(non_stable, total):.2f}%)")


if __name__ == "__main__":
    main()
