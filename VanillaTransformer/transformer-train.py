import argparse
import os
import random
import sys
from os import path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# repo root for dataloader.py
ROOT = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.insert(0, ROOT)

import dataloader as _dl
from dataloader import PoseItDataset, split_by_object, split_by_pose, uniform_random_split
from transformer_model import VanillaTransformer


def print_split_stats(dataset, train_set, val_set, test_set):
    print("Label mapping:")
    print("  0 = pass / no slip")
    print("  1 = slip or drop / fail")

    def _stats(name, subset):
        indices = subset.indices
        labels = [dataset.samples[i]["label"].item() for i in indices]
        n_total = len(labels)
        n_zero = sum(1 for x in labels if x == 0)
        n_one = sum(1 for x in labels if x == 1)
        zero_pct = 100.0 * n_zero / n_total if n_total else 0.0
        one_pct = 100.0 * n_one / n_total if n_total else 0.0
        print(
            f"{name}: total={n_total}  "
            f"0(pass/no slip)={n_zero} ({zero_pct:.1f}%)  "
            f"1(slip/drop)={n_one} ({one_pct:.1f}%)"
        )

    print("Split statistics:")
    _stats("Train", train_set)
    _stats("Val", val_set)
    _stats("Test", test_set)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", default="./data")
    p.add_argument("--split", default="object", choices=["object", "pose", "random"])
    p.add_argument("--test_object_ids", nargs="+", type=int, default=None)
    p.add_argument("--n_test_objects", type=int, default=None)
    p.add_argument("--test_pose_ids", nargs="+", type=int, default=None)
    p.add_argument("--n_test_poses", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--FRGB", type=int, default=1)
    p.add_argument("--FTactile", type=int, default=1)
    p.add_argument("--FFT", type=int, default=1)
    p.add_argument("--FGripper", type=int, default=1)
    p.add_argument("--L", type=int, default=20)
    p.add_argument("--overfit", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--hidden_dim", type=int, default=768)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--modalities", nargs="+", default=["V", "T", "FT", "G"])
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--model_save_path", default="trained_models/best_vanilla_transformer.pt")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    return p.parse_args()


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_split(dataset, args):
    if args.split == "object":
        objects = sorted(set(s["object"] for s in dataset.samples))
        print("Object index (sorted alphabetically):")
        for i, obj in enumerate(objects):
            print(f"  {i:>3}: {obj}")
        if args.test_object_ids is not None:
            test_objects = [objects[i] for i in args.test_object_ids]
        elif args.n_test_objects is not None:
            test_objects = sorted(random.sample(objects, args.n_test_objects))
        else:
            raise ValueError("--split object requires --test_object_ids or --n_test_objects")
        return split_by_object(dataset, test_objects)

    if args.split == "pose":
        poses = sorted(set(s["pose_idx"] for s in dataset.samples))
        print(f"Pose IDs in dataset: {poses}")
        if args.test_pose_ids is not None:
            test_poses = args.test_pose_ids
        elif args.n_test_poses is not None:
            test_poses = sorted(random.sample(poses, args.n_test_poses))
        else:
            raise ValueError("--split pose requires --test_pose_ids or --n_test_poses")
        return split_by_pose(dataset, test_poses)

    return uniform_random_split(dataset)


def make_loader(subset, batch_size, num_workers, shuffle=False):
    return DataLoader(subset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


def batch_to_device(batch, device):
    tac, rgb, ft, grip, gf, label, pose_label = batch
    return (
        tac.to(device),
        rgb.to(device),
        ft.to(device),
        grip.to(device),
        gf.to(device),
        label.to(device),
        pose_label.to(device),
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    tp = fp = fn = n = 0

    for batch in loader:
        tac, rgb, ft, grip, gf, label, _ = batch_to_device(batch, device)
        logits = model(tac, rgb, ft, grip, gf).squeeze(-1)
        total_loss += criterion(logits, label.float()).item() * len(label)

        preds = logits > 0
        actual = label.bool()
        tp += (preds & actual).sum().item()
        fp += (preds & ~actual).sum().item()
        fn += (~preds & actual).sum().item()
        n += len(label)

    if n == 0:
        return 0.0, 0.0

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return total_loss / n, f1


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    use_wandb = _WANDB_AVAILABLE and args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            entity=args.wandb_entity,
            config=vars(args),
        )
    elif args.wandb_project is not None:
        print("[WARN] wandb not installed — W&B logging disabled.")

    _dl.L = args.L
    _dl.FRGB = args.FRGB
    _dl.FTactile = args.FTactile
    _dl.FFT = args.FFT
    _dl.FGripper = args.FGripper
    _dl.refresh_sampling_dims()

    dataset = PoseItDataset(root_dir=args.root_dir)
    if len(dataset) == 0:
        raise ValueError(f"No samples found in {args.root_dir}")

    tac, rgb, ft, grip, gf, label, pose_label = dataset[0]
    print("Example sample shapes:")
    print(f"  tactile:       {tuple(tac.shape)}")
    print(f"  rgb:           {tuple(rgb.shape)}")
    print(f"  ft:            {tuple(ft.shape)}")
    print(f"  gripper:       {tuple(grip.shape)}")
    print(f"  gripper_force: {tuple(gf.shape)}")
    print(f"  label:         {label.item()}")
    print(f"  pose_label:    {pose_label.item()}")

    model = VanillaTransformer(
        frames_per_sec=args.FRGB,
        ft_dim=_dl.FT_DIM,
        gripper_dim=_dl.GR_DIM,
        max_timesteps=args.L if args.L > 0 else max(1, tac.shape[0]),
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        modalities=args.modalities,
    ).to(device)
    model.eval()

    with torch.no_grad():
        logits, debug = model(
            tac.unsqueeze(0).to(device),
            rgb.unsqueeze(0).to(device),
            ft.unsqueeze(0).to(device),
            grip.unsqueeze(0).to(device),
            gf.unsqueeze(0).to(device),
            return_debug=True,
        )

    print("Encoded output shapes:")
    for name, value in debug.items():
        print(f"  {name}: {tuple(value.shape)}")
    print(f"  logits:        {tuple(logits.shape)}")

    if args.overfit:
        dataset.samples = dataset.samples[:1]
        subset = Subset(dataset, [0])
        train_set = val_set = test_set = subset
    else:
        train_set, val_set, test_set = make_split(dataset, args)

    print_split_stats(dataset, train_set, val_set, test_set)

    train_loader = make_loader(train_set, args.batch_size, args.num_workers, shuffle=True)
    val_loader = make_loader(val_set, args.batch_size, args.num_workers)
    test_loader = make_loader(test_set, args.batch_size, args.num_workers)

    print(f"Loaded dataset with {len(dataset.samples)} samples")
    print(f"Train / Val / Test: {len(train_set)} / {len(val_set)} / {len(test_set)}")
    print(
        f"FRGB={args.FRGB}, FTactile={args.FTactile}, "
        f"FFT={args.FFT}, FGripper={args.FGripper}, L={args.L}"
    )
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Val loader batches:   {len(val_loader)}")
    print(f"Test loader batches:  {len(test_loader)}")

    train_labels = [dataset.samples[i]["label"].item() for i in train_set.indices]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    print(f"pos_weight={pos_weight.item():.3f} (n_pos={n_pos}, n_neg={n_neg})")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    save_dir = path.dirname(args.model_save_path) or "."
    path.isdir(save_dir) or os.makedirs(save_dir, exist_ok=True)
    best_val_f1 = -1.0
    train_iter = iter(train_loader)

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch in train_loader:
            tac, rgb, ft, grip, gf, label, _ = batch_to_device(batch, device)
            logits = model(tac, rgb, ft, grip, gf).squeeze(-1)
            loss = criterion(logits, label.float())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(label)
            train_count += len(label)

        avg_train_loss = train_loss_sum / max(train_count, 1)
        val_loss, val_f1 = evaluate(model, val_loader, criterion, device)
        print(
            f"[epoch {epoch + 1:3d}/{args.epochs:3d}] "
            f"train_loss={avg_train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_f1={val_f1:.3f}"
        )
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "val/loss": val_loss,
                "val/f1": val_f1,
            })
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), args.model_save_path)

    print("\nLoading best checkpoint for test evaluation...")
    model.load_state_dict(torch.load(args.model_save_path, map_location=device))
    test_loss, test_f1 = evaluate(model, test_loader, criterion, device)
    print(f"Test loss={test_loss:.4f}  f1={test_f1:.3f}")
    if use_wandb:
        wandb.log({
            "test/loss": test_loss,
            "test/f1": test_f1,
            "best_val_f1": best_val_f1,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
