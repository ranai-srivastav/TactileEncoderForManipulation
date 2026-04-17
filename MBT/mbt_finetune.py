"""
Fine-tuning script for a pretrained MBTGraspStability checkpoint on a new dataset.

Architecture config (modalities, L, F1/F2, bottlenecks, etc.) is read directly
from the checkpoint — no architecture flags needed.  Only dataset path, split,
and training hyper-parameters are required.

DRS is always active from the first iteration with sigma=1.0 (balanced classes).
W&B run name is auto-derived as finetune_<original_run_name>.

Examples:
# random split
python MBT/mbt_finetune.py \
    --checkpoint trained_models/best_mbt_model.pt \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
    --split random --n_iters 600 --anneal_iter 400

# object split
python MBT/mbt_finetune.py \
    --checkpoint trained_models/best_mbt_model.pt \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
    --split object --test_object_ids 0 5 10
"""

import argparse
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
                        uniform_random_split, FT_DIM, GR_DIM)
from sampler import DRSSampler
from mbt_model import MBTGraspStability


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
        labels = [('grasp', 'Grasp'), ('pose', 'Pose'), ('stability', 'Stability/Retract')]
        for key, display in labels:
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
    p.add_argument('--checkpoint',    required=True,
                   help='Path to pretrained .pt checkpoint (must contain state_dict + config).')
    p.add_argument('--root_dir',      default='./data')
    p.add_argument('--split',         default='object', choices=['object', 'pose', 'random'])
    p.add_argument('--test_object_ids', nargs='+', type=int, default=None,
                   help='Zero-based indices into the sorted alphabetical object list printed at '
                        'startup. Use with --split object. Takes precedence over --n_test_objects.')
    p.add_argument('--n_test_objects',  type=int, default=None,
                   help='Randomly pick N objects for the test set (--split object). '
                        'Ignored if --test_object_ids is given.')
    p.add_argument('--test_pose_ids',   nargs='+', type=int, default=None,
                   help='pose_idx integers (from folder names) to hold out for test. '
                        'Use with --split pose. Takes precedence over --n_test_poses.')
    p.add_argument('--n_test_poses',    type=int, default=None,
                   help='Randomly pick N pose IDs for the test set (--split pose). '
                        'Ignored if --test_pose_ids is given.')
    p.add_argument('--batch_size',    type=int,   default=4)
    p.add_argument('--grad_accum',    type=int,   default=8,
                   help='Gradient accumulation steps. Effective batch = batch_size × grad_accum')
    p.add_argument('--lr',            type=float, default=1e-4)
    p.add_argument('--weight_decay',  type=float, default=0.01)
    p.add_argument('--n_iters',       type=int,   default=600)
    p.add_argument('--anneal_iter',   type=int,   default=300,
                   help='Iteration to begin cosine LR decay (set > n_iters to disable)')
    p.add_argument('--num_workers',   type=int,   default=4)
    p.add_argument('--wandb_project', type=str,   default='TEMU')
    p.add_argument('--wandb_run',     type=str,   default=None,
                   help='Override W&B run name. Defaults to finetune_<original_run_name>.')
    p.add_argument('--wandb_entity',  type=str,   default='mrsd-smores')
    p.add_argument('--model_save_path', type=str, default='trained_models/best_mbt_finetune.pt')
    p.add_argument('--seed',          type=int,   default=None)
    return p.parse_args()


def make_split(dataset, args):
    if args.split == 'object':
        all_objects = sorted(set(s['object'] for s in dataset.samples))
        print("Object index (sorted alphabetically):")
        for i, obj in enumerate(all_objects):
            print(f"  {i:>3}: {obj}")

        if args.test_object_ids is not None:
            test_objects = [all_objects[i] for i in args.test_object_ids]
            print(f"Test objects (--test_object_ids {args.test_object_ids}): {test_objects}")
        elif args.n_test_objects is not None:
            test_objects = sorted(random.sample(all_objects, args.n_test_objects))
            print(f"Randomly selected {args.n_test_objects} test objects: {test_objects}")
        else:
            raise ValueError("--split object requires --test_object_ids or --n_test_objects")

        return split_by_object(dataset, test_objects=test_objects)

    elif args.split == 'pose':
        all_poses = sorted(set(s['pose_idx'] for s in dataset.samples))
        print(f"Pose IDs in dataset: {all_poses}")

        if args.test_pose_ids is not None:
            print(f"Test poses (--test_pose_ids): {args.test_pose_ids}")
            return split_by_pose(dataset, test_pose_indices=args.test_pose_ids)
        elif args.n_test_poses is not None:
            test_poses = sorted(random.sample(all_poses, args.n_test_poses))
            print(f"Randomly selected {args.n_test_poses} test poses: {test_poses}")
            return split_by_pose(dataset, test_pose_indices=test_poses)
        else:
            raise ValueError("--split pose requires --test_pose_ids or --n_test_poses")

    else:
        return uniform_random_split(dataset)


def make_loader(subset, sampler=None, batch_size=32, num_workers=4):
    if sampler is not None:
        return DataLoader(subset.dataset, batch_sampler=sampler, num_workers=num_workers)
    return DataLoader(subset, batch_size=batch_size, num_workers=num_workers)


def batch_to_device(batch, device):
    tac, rgb, ft, grip, gf, label, pose_label = batch
    lengths = [tac.shape[1]] * tac.shape[0]
    return (
        tac.to(device),
        rgb.to(device),
        ft.to(device),
        grip.to(device),
        gf.to(device),
        label.to(device),
        pose_label.to(device),
        lengths,
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    tp, fp, fn, n = 0, 0, 0, 0
    for batch in loader:
        tac, rgb, ft, grip, gf, label, _, lengths = batch_to_device(batch, device)
        with torch.autocast('cuda', enabled=torch.cuda.is_available()):
            logits = model(tac, rgb, ft, grip, gf).squeeze(-1)
            total_loss += criterion(logits, label.float()).item() * len(label)
        preds  = logits > 0
        actual = label.bool()
        tp += (preds &  actual).sum().item()
        fp += (preds & ~actual).sum().item()
        fn += (~preds & actual).sum().item()
        n  += len(label)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    acc       = (tp + (n - tp - fp - fn)) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return total_loss / n, acc, precision, recall, f1


def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Seed set to {args.seed}")

    # Load checkpoint first — config drives dataset shape and model construction
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg  = ckpt['config']
    print(f"  Original run: {cfg.get('run_name', 'unknown')}")
    print(f"  Modalities: {cfg['modalities']}  L={cfg['L']}  F1={cfg['F1']}  F2={cfg['F2']}")

    _dl.L  = cfg['L']
    _dl.F1 = cfg['F1']
    _dl.F2 = cfg['F2']

    effective_batch = args.batch_size * args.grad_accum
    print(f"Batch: {args.batch_size} micro × {args.grad_accum} accum = {effective_batch} effective")

    # W&B run name derived from original run
    finetune_run_name = args.wandb_run or f"finetune_{cfg.get('run_name', 'unknown')}"

    use_wandb = _WANDB_AVAILABLE and args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=finetune_run_name,
            entity=args.wandb_entity,
            config={**cfg, **vars(args)},
        )
        wandb.define_metric("iter")
        wandb.define_metric("train/*", step_metric="iter")
        wandb.define_metric("val/*", step_metric="iter")
        wandb.define_metric("lr", step_metric="iter")
    elif args.wandb_project is not None:
        print("[WARN] wandb not installed — W&B logging disabled.")

    # Dataset — shape driven by cfg
    ds = PoseItDataset(root_dir=args.root_dir)
    train_set, val_set, test_set = make_split(ds, args)
    print(f"Split ({args.split}): train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
    print_dataset_stats(ds, train_set, val_set, test_set)

    # DRS active immediately, sigma=1.0 (balanced)
    sampler = DRSSampler(
        dataset=ds,
        sigma=1.0,
        batch_size=args.batch_size,
        indices=train_set.indices,
    )

    train_loader = make_loader(train_set, sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader   = make_loader(val_set,   batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader  = make_loader(test_set,  batch_size=args.batch_size, num_workers=args.num_workers)

    # pos_weight from training labels
    train_labels = [ds.samples[i]['label'].item() for i in train_set.indices]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    print(f"pos_weight={pos_weight.item():.3f} (n_pos={n_pos}, n_neg={n_neg})")

    # Model — built from checkpoint config, weights loaded immediately
    model = MBTGraspStability(
        frames_per_sec=cfg['F1'],
        ft_dim=cfg['F2'] * 6,
        gripper_dim=cfg['F2'] * 2,
        max_timesteps=cfg['L'],
        num_bottlenecks=cfg['num_bottlenecks'],
        fusion_layer=cfg['fusion_layer'],
        max_visual_frames=cfg['max_visual_frames'],
        adapter_dim=cfg['adapter_dim'],
        dropout=cfg['dropout'],
        modalities=cfg['modalities'],
        pretrained_dir=cfg['pretrained_dir'],
        t3_encoder_domain=cfg['t3_encoder_domain'],
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    print(f"Loaded pretrained weights from {args.checkpoint}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total "
          f"({trainable/total*100:.1f}% trainable)")

    _model_config = {**cfg, 'run_name': finetune_run_name}

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Differential learning rates (same as pretraining)
    slow_params   = []
    other_params  = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_adapter    = any(k in name for k in ['_down.', '_up.', '_scale', '.down.', '.up.', '.scale'])
        is_tac_fusion = name.startswith('fusion_blocks.') and '.streams.T.' in name
        if is_adapter or is_tac_fusion:
            slow_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': args.lr},
        {'params': slow_params,  'lr': args.lr * 0.1},
    ], weight_decay=args.weight_decay)

    def lr_lambda(it):
        if it < args.anneal_iter:
            return 1.0
        progress = (it - args.anneal_iter) / max(args.n_iters - args.anneal_iter, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.GradScaler('cuda', enabled=torch.cuda.is_available())

    save_dir    = os.path.dirname(args.model_save_path) or '.'
    latest_path = os.path.join(save_dir, 'finetune_latest.pt')
    os.makedirs(save_dir, exist_ok=True)

    best_val_f1 = 0.0
    iteration   = 0
    accum_step  = 0
    accum_loss  = 0.0

    while iteration < args.n_iters:
        model.train()

        for batch in train_loader:
            if iteration >= args.n_iters:
                break

            tac, rgb, ft, grip, gf, label, _, lengths = batch_to_device(batch, device)

            with torch.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model(tac, rgb, ft, grip, gf).squeeze(-1)
                loss   = criterion(logits, label.float()) / args.grad_accum

            scaler.scale(loss).backward()
            accum_loss += loss.item() * args.grad_accum
            accum_step += 1

            if accum_step >= args.grad_accum:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                if iteration % 10 == 0 or iteration == args.n_iters - 1:
                    val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
                        model, val_loader, criterion, device)
                    current_lr = optimizer.param_groups[0]['lr']
                    avg_train_loss = accum_loss / args.grad_accum
                    print(f"[iter {iteration:4d}] "
                          f"train_loss={avg_train_loss:.4f}  "
                          f"val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%  "
                          f"prec={val_prec:.3f}  rec={val_rec:.3f}  f1={val_f1:.3f}  "
                          f"lr={current_lr:.2e}")

                    if use_wandb:
                        wandb.log({
                            'iter':           iteration,
                            'train/loss':     avg_train_loss,
                            'val/loss':       val_loss,
                            'val/acc':        val_acc,
                            'val/precision':  val_prec,
                            'val/recall':     val_rec,
                            'val/f1':         val_f1,
                            'lr':             current_lr,
                        }, step=iteration)

                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        torch.save({'state_dict': model.state_dict(), 'config': _model_config}, args.model_save_path)

                    if os.path.exists(latest_path):
                        os.remove(latest_path)
                    torch.save({'state_dict': model.state_dict(), 'config': _model_config}, latest_path)

                    if use_wandb:
                        wandb.save(latest_path, base_path=save_dir)
                        if os.path.exists(args.model_save_path):
                            wandb.save(args.model_save_path, base_path=save_dir)

                model.train()
                accum_step = 0
                accum_loss = 0.0
                iteration += 1

    print("\nLoading best checkpoint for test evaluation...")
    _ckpt = torch.load(args.model_save_path, map_location=device)
    model.load_state_dict(_ckpt['state_dict'] if isinstance(_ckpt, dict) else _ckpt)
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(
        model, test_loader, criterion, device)
    print(f"Test loss={test_loss:.4f}  acc={test_acc*100:.2f}%  "
          f"prec={test_prec:.3f}  rec={test_rec:.3f}  f1={test_f1:.3f}")

    if use_wandb:
        wandb.log({
            'test/loss':      test_loss,
            'test/acc':       test_acc,
            'test/precision': test_prec,
            'test/recall':    test_rec,
            'test/f1':        test_f1,
        }, step=args.n_iters - 1)
        wandb.run.summary.update({
            'test/loss': test_loss, 'test/acc': test_acc,
            'test/precision': test_prec, 'test/recall': test_rec, 'test/f1': test_f1,
        })
        wandb.finish()


if __name__ == '__main__':
    main()
