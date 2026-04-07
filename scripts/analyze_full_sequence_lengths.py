from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataloader_full import _build_full_entry_index, _second_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze full-resolution sequence lengths for PoseIt.")
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--modalities", nargs="+", default=["tactile", "rgb"])
    parser.add_argument("--plot_path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir)
    sample_dirs = sorted(path for path in root_dir.iterdir() if path.is_dir())

    second_lengths: list[int] = []
    token_lengths: list[int] = []
    objects: list[str] = []

    for sample_dir in sample_dirs:
        entry = _build_full_entry_index(sample_dir, modalities=args.modalities)
        seconds = _second_grid(entry["entry_start_timestamp"], entry["entry_end_timestamp"])
        t = int(seconds.shape[0])
        second_lengths.append(t)
        token_lengths.append(1 + t * len(args.modalities))
        objects.append(str(entry["object"]))

    seconds_arr = np.asarray(second_lengths)
    tokens_arr = np.asarray(token_lengths)

    print(f"samples={len(second_lengths)}")
    print(f"modalities={args.modalities}")
    print(f"seconds_min={seconds_arr.min()} seconds_mean={seconds_arr.mean():.2f} seconds_median={np.median(seconds_arr):.2f} seconds_max={seconds_arr.max()}")
    print(f"tokens_min={tokens_arr.min()} tokens_mean={tokens_arr.mean():.2f} tokens_median={np.median(tokens_arr):.2f} tokens_max={tokens_arr.max()}")

    top_indices = np.argsort(seconds_arr)[-10:][::-1]
    print("top_10_longest_samples:")
    for idx in top_indices:
        print(f"  {sample_dirs[int(idx)].name}: object={objects[int(idx)]} seconds={seconds_arr[int(idx)]} tokens={tokens_arr[int(idx)]}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(seconds_arr, bins=40, color="#1f77b4", edgecolor="black")
    axes[0].axvline(seconds_arr.mean(), color="red", linestyle="--", linewidth=1.5, label=f"mean={seconds_arr.mean():.1f}")
    axes[0].axvline(np.median(seconds_arr), color="green", linestyle=":", linewidth=1.5, label=f"median={np.median(seconds_arr):.1f}")
    axes[0].set_title("Seconds Per Sample")
    axes[0].set_xlabel("seconds")
    axes[0].set_ylabel("count")
    axes[0].legend()

    axes[1].hist(tokens_arr, bins=40, color="#ff7f0e", edgecolor="black")
    axes[1].axvline(tokens_arr.mean(), color="red", linestyle="--", linewidth=1.5, label=f"mean={tokens_arr.mean():.1f}")
    axes[1].axvline(np.median(tokens_arr), color="green", linestyle=":", linewidth=1.5, label=f"median={np.median(tokens_arr):.1f}")
    axes[1].set_title("Transformer Tokens Per Sample")
    axes[1].set_xlabel("tokens")
    axes[1].set_ylabel("count")
    axes[1].legend()

    fig.suptitle(f"Sequence Length Distribution ({', '.join(args.modalities)})")
    fig.tight_layout()
    plot_path = Path(args.plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    print(f"plot_path={plot_path}")


if __name__ == "__main__":
    main()
