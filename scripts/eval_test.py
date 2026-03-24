#!/usr/bin/env python3
"""
Load a checkpoint and evaluate on val and/or test with the same setup as train.py.

Must pass the same data/model flags you used during training. For --split random,
use --seed <int> if you trained with that seed (train.py must use the same seed for
the split to match).

Examples:
  python scripts/eval_test.py --checkpoint trained_models/clipt3_5k_best.pt \\
    --model clipt3 --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \\
    --variable_length --standardize_sensors --modalities V T FT G GF --split random

  python scripts/eval_test.py -c trained_models/best_model.pt --model resnet \\
    --root_dir ... --split object --test_objects mug bowl
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
# Project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import dataloader as _dl
from dataloader import (
    PoseItDataset,
    collate_variable_length,
    compute_sensor_stats,
    F2,
    FT_DIM,
    GR_DIM,
)
from model import GraspStabilityLSTM, GraspStabilityLSTM_CLIP_T3
from train import evaluate, make_loader, make_split, print_dataset_stats


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate checkpoint on val/test")
    p.add_argument("--checkpoint", "-c", required=True, help="Path to model state_dict .pt")
    p.add_argument("--root_dir", default="./data")
    p.add_argument("--split", default="object", choices=["object", "pose", "random"])
    p.add_argument("--test_objects", nargs="+", default=["mug", "bowl"])
    p.add_argument("--test_poses", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--lstm_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--F1", type=int, default=1)
    p.add_argument("--F2", type=int, default=1)
    p.add_argument("--L", type=int, default=20)
    p.add_argument("--variable_length", action="store_true")
    p.add_argument("--standardize_sensors", action="store_true")
    p.add_argument("--subsample", type=float, default=1.0)
    p.add_argument("--modalities", nargs="+", default=["V", "T", "FT", "G", "GF"])
    p.add_argument("--unidirectional", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--model", default="resnet", choices=["resnet", "clipt3"])
    p.add_argument("--t3_encoder_domain", default="gs_black")
    p.add_argument("--pretrained_dir", default="/ocean/projects/cis260031p/shared/pretrained")
    p.add_argument("--which", default="both", choices=["val", "test", "both"])
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="numpy seed before split (only affects --split random; must match training if used)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if args.seed is not None:
        np.random.seed(args.seed)

    _dl.L = None if args.variable_length else args.L
    _dl.F1 = args.F1
    _dl.F2 = args.F2

    if args.model == "clipt3":
        cache_dir = os.path.join(os.path.dirname(args.pretrained_dir), ".cache", "huggingface")
        os.environ.setdefault("HF_HOME", cache_dir)
        os.environ.setdefault("TEMU_PRETRAINED_DIR", args.pretrained_dir)

    rgb_preprocess = "clip" if args.model == "clipt3" else "imagenet"
    ds = PoseItDataset(root_dir=args.root_dir, rgb_preprocess=rgb_preprocess)
    if args.subsample < 1.0:
        import random

        k = max(4, int(len(ds.samples) * args.subsample))
        ds.samples = random.sample(ds.samples, k)
        print(f"Subsampled to {len(ds.samples)} samples")

    train_set, val_set, test_set = make_split(ds, args)
    print(f"Split ({args.split}): train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
    print_dataset_stats(ds, train_set, val_set, test_set)

    if args.standardize_sensors:
        stats = compute_sensor_stats(ds, train_set.indices)
        ds.set_sensor_stats(stats)

    collate_fn = collate_variable_length if args.variable_length else None
    val_loader = make_loader(
        val_set, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn
    )
    test_loader = make_loader(
        test_set, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn
    )

    train_labels = [ds.samples[i]["label"].item() for i in train_set.indices]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if args.model == "clipt3":
        model = GraspStabilityLSTM_CLIP_T3(
            frames_per_sec=F2,
            ft_dim=FT_DIM,
            gripper_dim=GR_DIM,
            hidden_dim=args.hidden_dim,
            lstm_layers=args.lstm_layers,
            bidirectional=not args.unidirectional,
            dropout=args.dropout,
            modalities=args.modalities,
            pretrained_dir=args.pretrained_dir,
            t3_encoder_domain=args.t3_encoder_domain,
        ).to(device)
    else:
        model = GraspStabilityLSTM(
            frames_per_sec=F2,
            ft_dim=FT_DIM,
            gripper_dim=GR_DIM,
            hidden_dim=args.hidden_dim,
            lstm_layers=args.lstm_layers,
            bidirectional=not args.unidirectional,
            dropout=args.dropout,
            modalities=args.modalities,
        ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    print(f"Loaded checkpoint: {args.checkpoint}")

    if args.which in ("val", "both"):
        v_loss, v_acc, v_p, v_r, v_f1 = evaluate(model, val_loader, criterion, device)
        print(
            f"Val   loss={v_loss:.4f}  acc={v_acc*100:.2f}%  "
            f"prec={v_p:.3f}  rec={v_r:.3f}  f1={v_f1:.3f}"
        )
    if args.which in ("test", "both"):
        t_loss, t_acc, t_p, t_r, t_f1 = evaluate(model, test_loader, criterion, device)
        print(
            f"Test  loss={t_loss:.4f}  acc={t_acc*100:.2f}%  "
            f"prec={t_p:.3f}  rec={t_r:.3f}  f1={t_f1:.3f}"
        )


if __name__ == "__main__":
    main()
