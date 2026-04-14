"""
Training script for Multimodal Bottleneck Transformer (MBT) on the PoseIt dataset.

Supports all three split modes (examples):
# by object
python MBT/mbt_train.py --split object --test_objects mug bowl --sigma 1.0

# by pose
python MBT/mbt_train.py --split pose --test_poses 1 2 3 4 5 --sigma 1.0

# random
python MBT/mbt_train.py --split random --anneal_iter 300 --n_iters 600

# overfit sanity check
python MBT/mbt_train.py --overfit --n_iters 100 --lr 0.001 --num_workers 0 \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight --wandb_project None
"""

import argparse
import math
import os
import sys

# dataloader.py and sampler.py live in the repo root, one level up from MBT/
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
    """Print per-phase label distribution for the loaded dataset and each split."""

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
    p.add_argument('--root_dir',     default='./data')
    p.add_argument('--split',        default='object', choices=['object', 'pose', 'random'])
    p.add_argument('--test_objects', nargs='+', default=['mug', 'bowl'])
    p.add_argument('--test_poses',   nargs='+', type=int, default=[1, 2, 3, 4, 5])
    p.add_argument('--sigma',        type=float, default=0.5,
                   help='DRS target S≠/S= ratio. 0.5 = gentler resampling')
    p.add_argument('--drs_iter',     type=int,   default=400,
                   help='Iteration at which DRS activates (separate from LR anneal)')
    p.add_argument('--batch_size',   type=int,   default=4,
                   help='Micro-batch size per GPU forward pass (keep small for V100)')
    p.add_argument('--grad_accum',   type=int,   default=8,
                   help='Gradient accumulation steps. Effective batch = batch_size × grad_accum')
    p.add_argument('--lr',           type=float, default=1e-4,
                   help='Peak learning rate (default: 1e-4 for AdamW with frozen backbone)')
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--dropout',           type=float, default=0.1)
    p.add_argument('--num_bottlenecks',   type=int,   default=4,
                   help='B bottleneck latent tokens (paper default: 4)')
    p.add_argument('--fusion_layer',      type=int,   default=8,
                   help='Lf: unimodal for layers 0..Lf-1, bottleneck fusion for Lf..11 (paper default: 8)')
    p.add_argument('--max_visual_frames', type=int,   default=8,
                   help='Frames subsampled from T*F1 for ViT tokenisation (default: 8, matches paper)')
    p.add_argument('--adapter_dim',       type=int,   default=64,
                   help='AdaptFormer hidden dim; 0 = frozen backbone only')
    p.add_argument('--t3_encoder_domain', type=str,   default='gs_black',
                   help='T3 sensor-specific encoder domain (gs_black | gs_tag)')
    p.add_argument('--pretrained_dir',    type=str,
                   default='/ocean/projects/cis260031p/shared/pretrained',
                   help='Directory containing T3 pretrained weights')
    p.add_argument('--n_iters',      type=int,   default=600)
    p.add_argument('--anneal_iter',  type=int,   default=300,
                   help='Iteration to begin cosine LR decay (set > n_iters to disable)')
    p.add_argument('--F1',          type=int,   default=1)
    p.add_argument('--F2',          type=int,   default=1)
    p.add_argument('--num_workers',  type=int,   default=4)
    p.add_argument('--modalities',   nargs='+',  default=['V', 'T', 'FT', 'G', 'GF'],
                   help='Active modalities: V T FT G GF')
    p.add_argument('--L',            type=int,   default=20,
                   help='Max seconds per episode (clips longer sequences)')
    p.add_argument('--subsample',    type=float, default=1.0,
                   help='Fraction of dataset to use (e.g. 0.01 for 1%%)')
    p.add_argument('--wandb_project', type=str, default="TEMU",
                   help='W&B project name. Set to None to disable W&B logging.')
    p.add_argument('--wandb_run',     type=str, default=None,
                   help='W&B run name (optional).')
    p.add_argument('--wandb_entity',  type=str, default="mrsd-smores",
                   help='W&B entity/team. Set to None to disable W&B logging.')
    p.add_argument('--overfit', action='store_true',
                   help='Use a single sample for train/val/test to sanity-check the model.')
    p.add_argument("--model_save_path", type=str, default="trained_models/best_mbt_model.pt")
    return p.parse_args()


def make_split(dataset, args):
    if args.split == 'object':
        return split_by_object(dataset, test_objects=args.test_objects)
    elif args.split == 'pose':
        return split_by_pose(dataset, test_pose_indices=args.test_poses)
    else:
        return uniform_random_split(dataset)


def make_loader(subset, sampler=None, batch_size=32, num_workers=4, shuffle=False):
    if sampler is not None:
        return DataLoader(subset.dataset, batch_sampler=sampler,
                          num_workers=num_workers)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)


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

    # Validate modality keys early
    valid_mods = {'V', 'T', 'FT', 'G', 'GF'}
    bad_mods = set(args.modalities) - valid_mods
    if bad_mods:
        raise ValueError(
            f"Invalid modality keys: {bad_mods}. "
            f"Valid keys are: {sorted(valid_mods)}")

    effective_batch = args.batch_size * args.grad_accum
    print(f"Batch: {args.batch_size} micro × {args.grad_accum} accum = {effective_batch} effective")

    # W&B initialisation
    use_wandb = _WANDB_AVAILABLE and args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            entity=args.wandb_entity,
            config=vars(args),
        )
        wandb.define_metric("iter")
        wandb.define_metric("train/*", step_metric="iter")
        wandb.define_metric("val/*", step_metric="iter")
        wandb.define_metric("lr", step_metric="iter")
        wandb.define_metric("drs_active", step_metric="iter")
    elif args.wandb_project is not None:
        print("[WARN] wandb not installed — W&B logging disabled.")

    # set episode length cap before dataset construction
    _dl.L = args.L
    _dl.F1 = args.F1
    _dl.F2 = args.F2

    # dataset
    ds = PoseItDataset(root_dir=args.root_dir)
    if args.subsample < 1.0:
        import random
        k = max(4, int(len(ds.samples) * args.subsample))
        ds.samples = random.sample(ds.samples, k)
        print(f"Subsampled to {len(ds.samples)} samples ({args.subsample*100:.1f}% of dataset)")
    if args.overfit:
        ds.samples = ds.samples[:1]
        overfit_set = Subset(ds, [0])
        train_set = val_set = test_set = overfit_set
        args.anneal_iter = args.n_iters + 1
        args.drs_iter = args.n_iters + 1
        print("Overfit mode: using 1 sample for train/val/test, DRS disabled")
    else:
        train_set, val_set, test_set = make_split(ds, args)
        print(f"Split ({args.split}): train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
        print_dataset_stats(ds, train_set, val_set, test_set)

    # DRS sampler uses the micro-batch size (what the DataLoader actually yields)
    sampler = DRSSampler(
        dataset=ds,
        sigma=args.sigma,
        batch_size=args.batch_size,
        indices=train_set.indices,
    )

    train_loader = make_loader(train_set, sampler=sampler, num_workers=args.num_workers)
    val_loader   = make_loader(val_set,   batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader  = make_loader(test_set,  batch_size=args.batch_size, num_workers=args.num_workers)

    # pos_weight
    train_labels = [ds.samples[i]['label'].item() for i in train_set.indices]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    print(f"pos_weight={pos_weight.item():.3f} (n_pos={n_pos}, n_neg={n_neg})")

    # Model
    model = MBTGraspStability(
        frames_per_sec=_dl.F1,
        ft_dim=FT_DIM,
        gripper_dim=GR_DIM,
        max_timesteps=args.L,
        num_bottlenecks=args.num_bottlenecks,
        fusion_layer=args.fusion_layer,
        max_visual_frames=args.max_visual_frames,
        adapter_dim=args.adapter_dim,
        dropout=args.dropout,
        modalities=args.modalities,
        pretrained_dir=args.pretrained_dir,
        t3_encoder_domain=args.t3_encoder_domain,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total "
          f"({trainable/total*100:.1f}% trainable)")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Differential learning rates
    slow_params   = []
    other_params  = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_adapter       = any(k in name for k in ['_down.', '_up.', '_scale', '.down.', '.up.', '.scale'])
        is_tac_fusion    = name.startswith('fusion_blocks.') and '.streams.T.' in name
        if is_adapter or is_tac_fusion:
            slow_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': other_params,   'lr': args.lr},
        {'params': slow_params,    'lr': args.lr * 0.1},
    ], weight_decay=args.weight_decay)

    # Cosine annealing: constant LR until anneal_iter, then cosine decay to 0
    def lr_lambda(it):
        if it < args.anneal_iter:
            return 1.0
        progress = (it - args.anneal_iter) / max(args.n_iters - args.anneal_iter, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision — cuts memory roughly in half for the frozen ViT forward passes
    scaler = torch.GradScaler('cuda', enabled=torch.cuda.is_available())

    # checkpoint paths
    save_dir    = os.path.dirname(args.model_save_path) or '.'
    latest_path = os.path.join(save_dir, 'model_latest.pt')
    os.makedirs(save_dir, exist_ok=True)

    # training loop
    best_val_f1 = 0.0
    iteration   = 0
    accum_step  = 0       # tracks micro-steps within one accumulation window
    accum_loss  = 0.0     # running loss across accumulation steps

    while iteration < args.n_iters:
        model.train()

        for batch in train_loader:
            if iteration >= args.n_iters:
                break

            # DRS activates at drs_iter
            if iteration == args.drs_iter and accum_step == 0:
                sampler.activate()
                print(f"[iter {iteration}] DRS activated")

            tac, rgb, ft, grip, gf, label, _, lengths = batch_to_device(batch, device)

            # Forward pass with mixed precision
            with torch.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model(tac, rgb, ft, grip, gf).squeeze(-1)
                loss   = criterion(logits, label.float()) / args.grad_accum

            # Backward (scaled for mixed precision)
            scaler.scale(loss).backward()
            accum_loss += loss.item() * args.grad_accum  # unscale for logging
            accum_step += 1

            # Optimizer step after grad_accum micro-batches
            if accum_step >= args.grad_accum:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                # Logging and evaluation
                if iteration % 10 == 0 or iteration == args.n_iters - 1:
                    val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
                        model, val_loader, criterion, device)
                    current_lr = optimizer.param_groups[0]['lr']
                    avg_train_loss = accum_loss / args.grad_accum
                    print(f"[iter {iteration:4d}] "
                          f"train_loss={avg_train_loss:.4f}  "
                          f"val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%  "
                          f"prec={val_prec:.3f}  rec={val_rec:.3f}  f1={val_f1:.3f}  "
                          f"lr={current_lr:.2e}  "
                          f"DRS={'on' if sampler.is_active else 'off'}")

                    if use_wandb:
                        wandb.log({
                            'iter':           iteration,
                            'train/loss':     avg_train_loss,
                            'val/loss':       val_loss,
                            'val/acc':        val_acc,
                            'val/precision':  val_prec,
                            'val/recall':     val_rec,
                            'val/f1':         val_f1,
                            'drs_active':     int(sampler.is_active),
                            'lr':             current_lr,
                        }, step=iteration)

                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        torch.save(model.state_dict(), args.model_save_path)

                    if os.path.exists(latest_path):
                        os.remove(latest_path)
                    torch.save(model.state_dict(), latest_path)

                    if use_wandb:
                        wandb.save(latest_path, base_path=save_dir)
                        if os.path.exists(args.model_save_path):
                            wandb.save(args.model_save_path, base_path=save_dir)

                model.train()
                accum_step = 0
                accum_loss = 0.0
                iteration += 1

    # test
    print("\nLoading best checkpoint for test evaluation...")
    model.load_state_dict(torch.load(args.model_save_path, map_location=device))
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