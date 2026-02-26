"""
PoseIt 2D unimodal training script (ResNet-18 + MLP).

Usage:
    # Train with RGB
    python unimodal/train_2d.py --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
        --modality rgb --test_objects mug bowl flashlight --n_iters 6000

    # Train with tactile
    python unimodal/train_2d.py --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
        --modality tactile --test_objects mug bowl flashlight --n_iters 6000

    # Test only
    python unimodal/train_2d.py --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
        --modality rgb --test_objects mug bowl flashlight --test
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

import dataloader as _dl
from dataloader import (PoseItDataset, split_by_object, split_by_pose,
                        uniform_random_split, collate_variable_length)
from unimodal_2d import GraspClassifier


# ---------------------------------------------------------------------------
# Helpers (kept from LSTM train.py)
# ---------------------------------------------------------------------------

def print_dataset_stats(dataset, train_set, val_set, test_set) -> None:
    def _count(samples):
        c = {
            'grasp':     [0, 0, 0],
            'pose':      [0, 0, 0],
            'stability': [0, 0, 0],
        }
        for s in samples:
            g = s.get('grasp_label', -1)
            c['grasp'][0 if g == 0 else (1 if g == 1 else 2)] += 1
            p = s['pose_label'].item()
            c['pose'][0 if p == 0 else 1] += 1
            l = s['label'].item()
            c['stability'][0 if l == 0 else 1] += 1
        return c

    def _print_split(name, samples):
        c = _count(samples)
        print(f'  {name} — {len(samples)} samples')
        print(f'    {"Phase":<18} {"Pass":>5} {"Fail":>5} {"Unknown":>8}')
        print(f'    {"-"*38}')
        for key, display in [('grasp', 'Grasp'), ('pose', 'Pose'), ('stability', 'Stability/Retract')]:
            p, f, u = c[key]
            print(f'    {display:<18} {p:>5} {f:>5} {u:>8}')

    train_s = [dataset.samples[i] for i in train_set.indices]
    val_s   = [dataset.samples[i] for i in val_set.indices]
    test_s  = [dataset.samples[i] for i in test_set.indices]

    print()
    print('=' * 54)
    print(f'Dataset stats — {len(dataset)} total samples loaded')
    _print_split('All', dataset.samples)
    print()
    _print_split('Train', train_s)
    print()
    _print_split('Val',   val_s)
    print()
    _print_split('Test',  test_s)
    print('=' * 54)
    print()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root_dir',     default='./data')
    p.add_argument('--modality',     default='rgb', choices=['rgb', 'tactile'],
                   help='Which image stream to use')
    p.add_argument('--split',        default='object', choices=['object', 'pose', 'random'])
    p.add_argument('--test_objects', nargs='+', default=['mug', 'bowl', 'flashlight'])
    p.add_argument('--test_poses',   nargs='+', type=int, default=[1, 2, 3, 4, 5])
    p.add_argument('--batch_size',   type=int,   default=32)
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--n_iters',      type=int,   default=6000)
    p.add_argument('--F1',           type=int,   default=1)
    p.add_argument('--F2',           type=int,   default=1)
    p.add_argument('--num_workers',  type=int,   default=4)
    p.add_argument('--L',            type=int,   default=3,
                   help='Last L seconds of grasp')
    p.add_argument('--subsample',    type=float, default=1.0)
    p.add_argument('--freeze_backbone', action='store_true', default=True,
                   help='Freeze ResNet-18 backbone (default: True)')
    p.add_argument('--no_freeze_backbone', dest='freeze_backbone', action='store_false')
    p.add_argument('--wandb_project', type=str, default='TEMU')
    p.add_argument('--wandb_run',     type=str, default=None)
    p.add_argument('--wandb_entity',  type=str, default='mrsd-smores')
    p.add_argument('--overfit',       action='store_true')
    p.add_argument('--model_save_path', type=str, default='trained_models/best_2d.pt')
    p.add_argument('--test',          action='store_true',
                   help='Load checkpoint and evaluate on test set only')
    return p.parse_args()


def make_split(dataset, args):
    if args.split == 'object':
        return split_by_object(dataset, test_objects=args.test_objects)
    elif args.split == 'pose':
        return split_by_pose(dataset, test_pose_indices=args.test_poses)
    else:
        return uniform_random_split(dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, modality):
    model.eval()
    total_loss = 0.0
    tp, fp, fn, n = 0, 0, 0, 0

    for tactile, rgb, _, _, _, labels, _, lengths in loader:
        imgs    = (tactile if modality == 'tactile' else rgb).to(device)
        labels  = labels.to(device)
        lengths = lengths.to(device)

        logits = model(imgs, lengths)          # (B, 2)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)

        preds  = logits.argmax(dim=1)
        actual = labels
        tp += ((preds == 1) & (actual == 1)).sum().item()
        fp += ((preds == 1) & (actual == 0)).sum().item()
        fn += ((preds == 0) & (actual == 1)).sum().item()
        n  += labels.size(0)

    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    tn        = n - tp - fp - fn
    acc       = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return total_loss / n, acc, precision, recall, f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}  modality: {args.modality}")

    # W&B
    use_wandb = _WANDB_AVAILABLE and args.wandb_project is not None
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run,
                   entity=args.wandb_entity, config=vars(args))
    elif args.wandb_project is not None:
        print("[WARN] wandb not installed — W&B logging disabled.")

    # dataloader globals
    _dl.L  = args.L
    _dl.F1 = args.F1
    _dl.F2 = args.F2

    # dataset
    ds = PoseItDataset(root_dir=args.root_dir)
    if args.subsample < 1.0:
        import random
        k = max(4, int(len(ds.samples) * args.subsample))
        ds.samples = random.sample(ds.samples, k)
        print(f"Subsampled to {len(ds.samples)} samples ({args.subsample*100:.1f}%)")

    if args.overfit:
        ds.samples = ds.samples[:1]
        overfit_set = Subset(ds, [0])
        train_set = val_set = test_set = overfit_set
        print("Overfit mode: 1 sample for train/val/test")
    else:
        train_set, val_set, test_set = make_split(ds, args)
        print(f"Split ({args.split}): train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
        print_dataset_stats(ds, train_set, val_set, test_set)

    criterion = nn.CrossEntropyLoss()

    # ---- Test-only mode ----
    if args.test:
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_variable_length)
        model = GraspClassifier(freeze_backbone=args.freeze_backbone).to(device)
        ckpt  = args.model_save_path
        assert os.path.exists(ckpt), f"Checkpoint not found: {ckpt}"
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded checkpoint: {ckpt}")

        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(
            model, test_loader, criterion, device, args.modality)
        print(f"\nTest Results:")
        print(f"  loss={test_loss:.4f}  acc={test_acc*100:.2f}%  "
              f"prec={test_prec:.3f}  rec={test_rec:.3f}  f1={test_f1:.3f}")

        if use_wandb:
            wandb.log({'test/loss': test_loss, 'test/acc': test_acc,
                       'test/precision': test_prec, 'test/recall': test_rec,
                       'test/f1': test_f1})
            wandb.finish()
        return

    # ---- Training mode ----
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=collate_variable_length)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers,
                              collate_fn=collate_variable_length)
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers,
                              collate_fn=collate_variable_length)

    model = GraspClassifier(freeze_backbone=args.freeze_backbone).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)

    # checkpoint paths
    save_dir    = os.path.dirname(args.model_save_path) or '.'
    latest_path = os.path.join(save_dir, 'model_2d_latest.pt')
    os.makedirs(save_dir, exist_ok=True)

    if use_wandb:
        wandb.log({"modality": args.modality, "freeze_backbone": args.freeze_backbone,
                    "lr": args.lr, "weight_decay": args.weight_decay,
                    "batch_size": args.batch_size, "n_iters": args.n_iters,
                    "sequence_length": args.L})

    # iteration-based training loop (matching LSTM train.py style)
    best_val_f1 = 0.0
    iteration   = 0

    while iteration < args.n_iters:
        model.train()

        for tactile, rgb, _, _, _, labels, _, lengths in train_loader:
            if iteration >= args.n_iters:
                break

            imgs    = (tactile if args.modality == 'tactile' else rgb).to(device)
            labels  = labels.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            logits = model(imgs, lengths)      # (B, 2)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            if iteration % 10 == 0:
                val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
                    model, val_loader, criterion, device, args.modality)
                print(f"[iter {iteration:4d}] "
                      f"train_loss={loss.item():.4f}  "
                      f"val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%  "
                      f"prec={val_prec:.3f}  rec={val_rec:.3f}  f1={val_f1:.3f}")

                if use_wandb:
                    wandb.log({
                        'iter': iteration, 'train/loss': loss.item(),
                        'val/loss': val_loss, 'val/acc': val_acc,
                        'val/precision': val_prec, 'val/recall': val_rec,
                        'val/f1': val_f1,
                    }, step=iteration)

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    torch.save(model.state_dict(), args.model_save_path)

                # rolling latest checkpoint
                if os.path.exists(latest_path):
                    os.remove(latest_path)
                torch.save(model.state_dict(), latest_path)

                if use_wandb:
                    wandb.save(latest_path, base_path=save_dir)
                    if os.path.exists(args.model_save_path):
                        wandb.save(args.model_save_path, base_path=save_dir)

            model.train()
            iteration += 1

    # Final test evaluation
    print(f"\nBest val F1: {best_val_f1:.3f}")
    print("Loading best checkpoint for test evaluation...")
    model.load_state_dict(torch.load(args.model_save_path, map_location=device))
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(
        model, test_loader, criterion, device, args.modality)
    print(f"Test loss={test_loss:.4f}  acc={test_acc*100:.2f}%  "
          f"prec={test_prec:.3f}  rec={test_rec:.3f}  f1={test_f1:.3f}")

    if use_wandb:
        wandb.log({'test/loss': test_loss, 'test/acc': test_acc,
                   'test/precision': test_prec, 'test/recall': test_rec,
                   'test/f1': test_f1})
        wandb.finish()


if __name__ == '__main__':
    main()