#!/usr/bin/env python3
"""
Download pretrained weights for CLIP + T3 encoders to shared directory.

T3 large (304M, gs_black encoder + trunk) from Hugging Face alanz-mit/FoundationTactile.
CLIP ViT-L/14 (LAION-2B) downloaded on first use via open_clip (cached to HF_HOME).

Usage:
    python scripts/download_pretrained.py [--shared_dir /path/to/shared]

Sets HF_HOME so CLIP and T3 downloads go to shared/.cache/huggingface.
"""

import argparse
import os
import sys

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    p = argparse.ArgumentParser(description="Download T3 and CLIP pretrained weights")
    p.add_argument(
        "--shared_dir",
        default="/ocean/projects/cis260031p/shared",
        help="Shared directory for pretrained weights and cache",
    )
    args = p.parse_args()

    shared = os.path.abspath(args.shared_dir)
    pretrained_dir = os.path.join(shared, "pretrained")
    cache_dir = os.path.join(shared, ".cache", "huggingface")
    t3_dir = os.path.join(pretrained_dir, "models", "t3_large")

    os.makedirs(pretrained_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Set HF cache so downloads go to shared
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir

    print(f"HF cache: {cache_dir}")
    print(f"T3 output: {t3_dir}")

    # Download T3 large from Hugging Face (dataset repo)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing huggingface_hub: pip install huggingface_hub")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import hf_hub_download

    repo_id = "alanz-mit/FoundationTactile"
    files = [
        "models/t3_large/encoders/gs_black.pth",
        "models/t3_large/encoders/gs_tag.pth",
        "models/t3_large/trunk.pth",
    ]

    for hf_path in files:
        local_path = os.path.join(pretrained_dir, hf_path)
        if os.path.exists(local_path):
            print(f"Already exists: {local_path}")
            continue
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"Downloading {hf_path}...")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=hf_path,
            repo_type="dataset",
            local_dir=pretrained_dir,
            local_dir_use_symlinks=False,
        )
        print(f"  -> {downloaded}")

    # Pre-download CLIP ViT-L/14 LAION-2B (optional - will also download on first model load)
    print("\nPre-downloading CLIP ViT-L/14 (LAION-2B)...")
    try:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="laion2b_s32b_b82k"
        )
        print("  CLIP ViT-L-14 cached successfully")
    except ImportError:
        print("  open_clip not installed. Run: pip install open_clip_torch")
    except Exception as e:
        print(f"  CLIP download warning: {e} (will retry on first use)")

    print(f"\nDone. Set HF_HOME={cache_dir} when running training.")


if __name__ == "__main__":
    main()
