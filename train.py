"""
PoseIt training script.

Supports all three split modes (examples):
# by object
python train.py --split object --test_objects mug bowl --sigma 1.0

# by pose
python train.py --split pose --test_poses 1 2 3 4 5 --sigma 1.0

# random
python train.py --split random --anneal_iter 300 --n_iters 600
"""

import argparse
import os

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
                        uniform_random_split)
from sampler import DRSSampler
from model import GraspStabilityLSTM


def print_dataset_stats(dataset, train_set, val_set, test_set) -> None:
    """Print per-phase label distribution for the loaded dataset and each split.

    Phases:
      Grasp     — label for the grasping phase (stored as grasp_label; -1 = unknown)
      Pose      — label for the pose phase (pose_label)
      Stability — label for the stability/retract phase (label, used for training)
    """

    def _count(samples):
        c = {
            'grasp':     [0, 0, 0],   # [pass, fail, unknown]
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
    p.add_argument('--batch_size',   type=int,   default=32)
    p.add_argument('--lr',           type=float, default=0.01)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--dropout',      type=float, default=0.1)
    p.add_argument('--hidden_dim',   type=int,   default=256)
    p.add_argument('--lstm_layers',  type=int,   default=2,
                   help='Number of LSTM layers (default: 2)')
    p.add_argument('--n_iters',      type=int,   default=600)
    p.add_argument('--anneal_iter',  type=int,   default=300)
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
                   help='W&B project name. Default is `TEMU`. Set to None to disable W&B logging.')
    p.add_argument('--wandb_run',     type=str, default=None,
                   help='W&B run name (optional).')
    p.add_argument('--wandb_entity',  type=str, default="mrsd-smores",
                   help='W&B entity/team. Default is "mrsd-smores". Set to None to disable W&B logging.')
    p.add_argument('--unidirectional', action='store_true',
                   help='Use unidirectional LSTM (default: bidirectional)')
    p.add_argument('--overfit', action='store_true',
                   help='Use a single sample for train/val/test to sanity-check the model.')
    p.add_argument("--model_save_path", type=str, default="trained_models/best_model.pt")

    # --- optimizer / scheduler ---
    p.add_argument('--optimizer',     default='sgd', choices=['sgd', 'adamw'],
                   help='Optimizer: sgd (momentum=0.9) or adamw')
    p.add_argument('--lr_scheduler',  default='step', choices=['step', 'cosine_warm', 'none'],
                   help='LR schedule: step (StepLR at anneal_iter), cosine_warm '
                        '(CosineAnnealingWarmRestarts every iter), none')
    p.add_argument('--cosine_t0',     type=int, default=100,
                   help='T_0 for CosineAnnealingWarmRestarts (iters per first cycle)')
    p.add_argument('--cosine_t_mult', type=int, default=2,
                   help='T_mult for CosineAnnealingWarmRestarts (cycle length multiplier)')

    # --- architecture extras ---
    p.add_argument('--n_outputs',      type=int, default=1, choices=[1, 2],
                   help='Output head size: 1=BCEWithLogitsLoss, 2=CrossEntropyLoss')
    p.add_argument('--freeze',         nargs='*',
                   default=['resnet_rgb', 'resnet_tactile'],
                   help='Components to freeze. Default: both ResNets. '
                        'Pass --freeze with no args to train everything. '
                        'Choices: resnet_rgb, resnet_tactile, projection, gru, classifier. '
                        'Example: --freeze resnet_rgb resnet_tactile gru')
    p.add_argument('--clip_grad_norm', type=float, default=1.0,
                   help='Max gradient norm for clipping (0 = disabled)')
    p.add_argument('--tau', type=float, default=0.0,
                   help='Tau regularization: adds tau * ||W_majority||_2 to CE loss. '
                        'Only active with --n_outputs 2. tau=0 disables it. '
                        'Typical range: 0.001–0.1 (tau=1 adds the full weight norm, '
                        'which is usually 3–10x larger than CE loss).')
    # --- sweep-friendly iter fractions ---
    p.add_argument('--anneal_frac', type=float, default=None,
                   help='If set, anneal_iter = int(anneal_frac * n_iters). Overrides --anneal_iter. '
                        'Use in sweeps so the LR drop stays proportional to training length.')
    p.add_argument('--drs_frac', type=float, default=None,
                   help='If set, drs_iter = int(drs_frac * n_iters). Overrides --drs_iter. '
                        'Use in sweeps so DRS activation stays proportional to training length.')
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
        # batch_sampler controls both batching and shuffling — don't pass batch_size/shuffle
        return DataLoader(subset.dataset, batch_sampler=sampler,
                          num_workers=num_workers)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)


def batch_to_device(batch, device):
    tac, rgb, ft, grip, gf, label, pose_label = batch
    lengths = [tac.shape[1]] * tac.shape[0]  # uniform T since L is fixed
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
    """
    Returns 8-tuple:
        (loss, acc, precision, recall, f1, tpr, tnr, pos_pred_rate)

    tpr  = TP / (TP+FN)  — recall on positives (sensitivity)
    tnr  = TN / (TN+FP)  — recall on negatives (specificity)
    ppr  = (TP+FP) / N   — fraction of predictions that are positive
                           (near-0 means model collapsed to predicting all-negative)
    """
    model.eval()
    total_loss = 0.0
    tp, fp, fn, n = 0, 0, 0, 0
    for batch in loader:
        tac, rgb, ft, grip, gf, label, _, lengths = batch_to_device(batch, device)
        logits = model(tac, rgb, ft, grip, gf)
        if model.n_outputs == 1:
            logits = logits.squeeze(1)                 # (B,)
            total_loss += criterion(logits, label.float()).item() * len(label)
            preds = logits > 0
        else:
            total_loss += criterion(logits, label.long()).item() * len(label)
            preds = logits.argmax(1).bool()
        actual = label.bool()
        tp += (preds &  actual).sum().item()
        fp += (preds & ~actual).sum().item()
        fn += (~preds & actual).sum().item()
        n  += len(label)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    tn        = n - tp - fp - fn
    acc       = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    tpr       = recall                                 # same as recall
    tnr       = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppr       = (tp + fp) / n                         # positive prediction rate
    return total_loss / n, acc, precision, recall, f1, tpr, tnr, ppr


@torch.no_grad()
def _ablation_eval(model, loader, criterion, device, active_modalities):
    """Zero one modality at a time. Returns {modality: f1} for each active modality."""
    original = model.modalities.copy()
    results = {}
    for ablate in sorted(active_modalities):
        model.modalities = original - {ablate}
        model.eval()
        *_, f1, _, _, _ = evaluate(model, loader, criterion, device)
        results[ablate] = f1
    model.modalities = original
    return results


def _parse_wb_list(values):
    """W&B passes nargs='*'/nargs='+' args as a Python-repr string, e.g.
    \"['resnet_rgb', 'resnet_tactile']\" instead of separate tokens.
    Detect and unwrap that case so --freeze and --modalities work in sweeps."""
    import ast
    if not values:
        return []
    if len(values) == 1 and values[0].startswith('['):
        parsed = ast.literal_eval(values[0])
        return [str(x) for x in parsed]
    return values


def main():
    args   = parse_args()
    # Normalize list args that W&B may pass as a Python-repr string
    args.freeze       = _parse_wb_list(args.freeze or [])
    args.modalities   = _parse_wb_list(args.modalities)
    args.test_objects = _parse_wb_list(args.test_objects or [])
    args.test_poses   = [int(x) for x in _parse_wb_list([str(v) for v in (args.test_poses or [])])]
    # Fraction-based overrides — keep anneal/DRS milestones proportional to n_iters
    if args.anneal_frac is not None:
        args.anneal_iter = int(args.anneal_frac * args.n_iters)
    if args.drs_frac is not None:
        args.drs_iter = int(args.drs_frac * args.n_iters)
    # Auto-cap batch_size when ResNets are unfrozen — effective ResNet batch is batch_size * L * F1.
    # Budget: ~28 GB, ~175 MB/image with gradients, 2 encoders.
    resnet_unfrozen = ('resnet_rgb' not in args.freeze) or ('resnet_tactile' not in args.freeze)
    if resnet_unfrozen:
        max_bs = max(1, 28_000 // (args.L * args.F1 * 2 * 175))
        if args.batch_size > max_bs:
            print(f"[INFO] ResNet unfrozen: auto-capping batch_size {args.batch_size} → {max_bs} "
                  f"(effective ResNet batch = {max_bs * args.L * args.F1})")
            args.batch_size = max_bs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # W&B initialisation
    use_wandb = _WANDB_AVAILABLE and args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            entity=args.wandb_entity,
            config=vars(args),
        )
        # Use "iter" as x-axis for all training/val metrics so plots align properly
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
    FT_DIM = _dl.F2 * 6   # recompute after CLI override — not captured at import time
    GR_DIM = _dl.F2 * 2

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
        args.anneal_iter = args.n_iters + 1   # disable LR anneal
        args.drs_iter = args.n_iters + 1      # disable DRS (S≠ may be empty with 1 sample)
        print("Overfit mode: using 1 sample for train/val/test, DRS disabled")
    else:
        train_set, val_set, test_set = make_split(ds, args)
        print(f"Split ({args.split}): train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
        print_dataset_stats(ds, train_set, val_set, test_set)

    # deferred sampling
    sampler = DRSSampler(
        dataset=ds,
        sigma=args.sigma,
        batch_size=args.batch_size,
        indices=train_set.indices,
    )

    train_loader = make_loader(train_set, sampler=sampler, num_workers=args.num_workers)
    val_loader   = make_loader(val_set,   batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader  = make_loader(test_set,  batch_size=args.batch_size, num_workers=args.num_workers)

    # pos_weight: upweight minority (unstable) to avoid predicting only majority class
    train_labels = [ds.samples[i]['label'].item() for i in train_set.indices]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    print(f"pos_weight={pos_weight.item():.3f} (n_pos={n_pos}, n_neg={n_neg})")

    # Model
    freeze_set = set(args.freeze or [])
    model = GraspStabilityLSTM(
        frames_per_sec=args.F1,
        ft_dim=FT_DIM,
        gripper_dim=GR_DIM,
        hidden_dim=args.hidden_dim,
        lstm_layers=args.lstm_layers,
        bidirectional=not args.unidirectional,
        dropout=args.dropout,
        freeze_resnet_rgb=     ('resnet_rgb'      in freeze_set),
        freeze_resnet_tactile= ('resnet_tactile'  in freeze_set),
        freeze_projection=     ('projection'      in freeze_set),
        freeze_gru=            ('gru'             in freeze_set),
        freeze_classifier=     ('classifier'      in freeze_set),
        n_outputs=args.n_outputs,
        modalities=args.modalities,
    ).to(device)

    # Loss — BCE for single logit, weighted CE for two logits
    if args.n_outputs == 1:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        ce_weight = torch.tensor([1.0, pos_weight.item()], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=ce_weight)

    # Optimizer — ResNet params (if unfrozen) get lr/10 to avoid destroying pretrained features
    resnet_param_ids = {id(p) for p in list(model.rgb_encoder.parameters())
                                      + list(model.tactile_encoder.parameters())}
    head_params    = [p for p in model.parameters() if p.requires_grad and id(p) not in resnet_param_ids]
    resnet_params  = [p for p in model.parameters() if p.requires_grad and id(p) in resnet_param_ids]
    param_groups   = [{'params': head_params, 'lr': args.lr}]
    if resnet_params:
        param_groups.append({'params': resnet_params, 'lr': args.lr / 10})
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:  # adamw
        optimizer = torch.optim.AdamW(
            param_groups, lr=args.lr, weight_decay=args.weight_decay)

    # LR scheduler
    if args.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    elif args.lr_scheduler == 'cosine_warm':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.cosine_t0, T_mult=args.cosine_t_mult)
    else:
        scheduler = None

    # W&B: log derived model + data stats not captured in args
    if use_wandb:
        n_params_total     = sum(p.numel() for p in model.parameters())
        n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update({
            'n_params_total':       n_params_total,
            'n_params_trainable':   n_params_trainable,
            'loss_fn':              'BCE' if args.n_outputs == 1 else ('CrossEntropy+tau' if args.tau > 0 else 'CrossEntropy'),
            'tau':                  args.tau,
            'optimizer_type':       args.optimizer,
            'lr_resnet':            args.lr / 10 if resnet_params else 0.0,
            'scheduler_type':       args.lr_scheduler,
            'modalities_str':       '+'.join(sorted(args.modalities)),
            'n_active_modalities':  len(args.modalities),
            'freeze_components':        sorted(freeze_set),
            'freeze_resnet_rgb':        'resnet_rgb'     in freeze_set,
            'freeze_resnet_tactile':    'resnet_tactile' in freeze_set,
            'freeze_projection':        'projection'     in freeze_set,
            'freeze_gru':               'gru'            in freeze_set,
            'freeze_classifier':        'classifier'     in freeze_set,
            'n_train':              len(train_set),
            'n_val':                len(val_set),
            'n_test':               len(test_set),
            'n_pos_train':          n_pos,
            'n_neg_train':          n_neg,
            'pos_weight_value':     pos_weight.item(),
            'class_balance_train':  n_pos / max(len(train_labels), 1),
        }, allow_val_change=True)

    # checkpoint paths — use W&B run ID to avoid collisions between parallel agents
    if use_wandb:
        save_dir = os.path.join('trained_models', wandb.run.id)
        args.model_save_path = os.path.join(save_dir, 'best_model.pt')
    else:
        save_dir = os.path.dirname(args.model_save_path) or '.'
    latest_path = os.path.join(save_dir, 'model_latest.pt')
    os.makedirs(save_dir, exist_ok=True)

    # training loop
    best_val_f1 = 0.0
    iteration   = 0

    while iteration < args.n_iters:
        model.train()

        for batch in train_loader:
            if iteration >= args.n_iters:
                break

            # LR anneal / schedule
            if args.lr_scheduler == 'step' and iteration == args.anneal_iter:
                scheduler.step()  # type: ignore[union-attr]
                print(f"[iter {iteration}] LR annealed to {optimizer.param_groups[0]['lr']:.2e}")
            elif args.lr_scheduler == 'cosine_warm' and scheduler is not None:
                scheduler.step(iteration)
            # DRS activates at drs_iter (decoupled; can be later to avoid overcorrection)
            if iteration == args.drs_iter:
                sampler.activate()
                print(f"[iter {iteration}] DRS activated")

            tac, rgb, ft, grip, gf, label, _, lengths = batch_to_device(batch, device)

            optimizer.zero_grad()
            logits = model(tac, rgb, ft, grip, gf)
            if args.n_outputs == 1:
                loss = criterion(logits.squeeze(1), label.float())
            else:
                loss = criterion(logits, label.long())
                if args.tau > 0.0:
                    # Tau regularization: penalize L2 norm of majority class (S=, class 0)
                    # weight vector in the final classifier layer.
                    # Pulls W_majority toward zero; leaves W_minority untouched.
                    majority_w = model.classifier[-1].weight[0]  # shape: (64,)
                    loss = loss + args.tau * majority_w.norm(2)
            loss.backward()

            # Gradient clipping + norm logging
            if args.clip_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_grad_norm).item()
            else:
                grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in model.parameters() if p.grad is not None
                ) ** 0.5

            optimizer.step()

            if iteration % 10 == 0 or iteration == args.n_iters - 1:
                val_loss, val_acc, val_prec, val_rec, val_f1, val_tpr, val_tnr, val_ppr = evaluate(
                    model, val_loader, criterion, device)
                current_lr = optimizer.param_groups[0]['lr']
                print(f"[iter {iteration:4d}] "
                      f"train_loss={loss.item():.4f}  "
                      f"val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%  "
                      f"prec={val_prec:.3f}  rec={val_rec:.3f}  f1={val_f1:.3f}  "
                      f"tpr={val_tpr:.3f}  tnr={val_tnr:.3f}  ppr={val_ppr:.3f}  "
                      f"DRS={'on' if sampler.is_active else 'off'}")

                if use_wandb:
                    wandb.log({
                        'iter':                   iteration,
                        'train/loss':             loss.item(),
                        'train/grad_norm':        grad_norm,
                        'train/batch_size_actual': len(label),
                        'val/loss':               val_loss,
                        'val/acc':                val_acc,
                        'val/precision':          val_prec,
                        'val/recall':             val_rec,
                        'val/f1':                 val_f1,
                        'val/tpr':                val_tpr,
                        'val/tnr':                val_tnr,
                        'val/pos_pred_rate':      val_ppr,
                        'drs_active':             int(sampler.is_active),
                        'lr':                     current_lr,
                        'lr_resnet':              optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else 0.0,
                    }, step=iteration)

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    torch.save(model.state_dict(), args.model_save_path)

                # Rolling latest checkpoint — delete previous, save current
                if os.path.exists(latest_path):
                    os.remove(latest_path)
                torch.save(model.state_dict(), latest_path)

                # Upload both checkpoints to W&B
                if use_wandb:
                    wandb.save(latest_path, base_path=save_dir)
                    if os.path.exists(args.model_save_path):
                        wandb.save(args.model_save_path, base_path=save_dir)

            model.train()   # restore training mode after evaluate()
            iteration += 1

    # test — evaluate both best_model.pt and model_latest.pt
    print("\n=== Test evaluation ===")
    best_test_f1 = None
    for ckpt_label, ckpt_path in [("best_model", args.model_save_path), ("model_latest", latest_path)]:
        if not os.path.exists(ckpt_path):
            print(f"[WARN] {ckpt_path} not found — skipping")
            continue
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        test_loss, test_acc, test_prec, test_rec, test_f1, test_tpr, test_tnr, test_ppr = evaluate(
            model, test_loader, criterion, device)
        print(f"[{ckpt_label}] loss={test_loss:.4f}  acc={test_acc*100:.2f}%  "
              f"prec={test_prec:.3f}  rec={test_rec:.3f}  f1={test_f1:.3f}  "
              f"tpr={test_tpr:.3f}  tnr={test_tnr:.3f}  ppr={test_ppr:.3f}")
        if use_wandb:
            summary = {
                f'test_{ckpt_label}/loss':         test_loss,
                f'test_{ckpt_label}/acc':          test_acc,
                f'test_{ckpt_label}/precision':    test_prec,
                f'test_{ckpt_label}/recall':       test_rec,
                f'test_{ckpt_label}/f1':           test_f1,
                f'test_{ckpt_label}/tpr':          test_tpr,
                f'test_{ckpt_label}/tnr':          test_tnr,
                f'test_{ckpt_label}/pos_pred_rate': test_ppr,
            }
            wandb.log(summary, step=args.n_iters - 1)
            wandb.run.summary.update(summary)

        # Modality ablation on best_model checkpoint
        if ckpt_label == "best_model" and len(args.modalities) > 1:
            print(f"\n  [ablation on {ckpt_label}]")
            ablation = _ablation_eval(model, test_loader, criterion, device, set(args.modalities))
            for mod, abl_f1 in ablation.items():
                drop = test_f1 - abl_f1
                print(f"    drop_{mod}: f1={abl_f1:.3f}  Δf1={-drop:+.3f}")
            if use_wandb:
                abl_log = {}
                for mod, abl_f1 in ablation.items():
                    abl_log[f'test_best/ablation_no_{mod}'] = abl_f1
                    abl_log[f'test_best/ablation_drop_{mod}'] = test_f1 - abl_f1
                wandb.log(abl_log, step=args.n_iters - 1)
                wandb.run.summary.update(abl_log)
            best_test_f1 = test_f1

    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
