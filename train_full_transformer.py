import argparse
import math
import os
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataloader import split_by_object, split_by_pose, uniform_random_split
from dataloader_full import (
    PoseItDataLoaderFull,
    PoseItModality,
    PoseItPaddingMetadata,
    PoseItTimeLayout,
)
from model_transformer import DEFAULT_MODALITIES, SlipTransformer


def parse_args():
    p = argparse.ArgumentParser(description='Train transformer on full-resolution by-second PoseIt data')
    p.add_argument('--root_dir', default='./data')
    p.add_argument('--split', default='random', choices=['object', 'pose', 'random'])
    p.add_argument('--test_objects', nargs='+', default=['mug', 'bowl'])
    p.add_argument('--test_poses', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    p.add_argument('--modalities', nargs='+', default=list(DEFAULT_MODALITIES))
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--nhead', type=int, default=8)
    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--dim_feedforward', type=int, default=2048)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--max_seconds', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--n_iters', type=int, default=20)
    p.add_argument('--log_interval', type=int, default=1)
    p.add_argument('--subsample', type=float, default=1.0)
    p.add_argument('--overfit', action='store_true')
    p.add_argument('--model_save_path', type=str, default='trained_models/full_transformer_best.pt')
    return p.parse_args()


def normalize_modalities(names: Sequence[str]) -> List[PoseItModality]:
    normalized = []
    for name in names:
        if name == 'gripper_force':
            continue
        normalized.append(PoseItModality(name))
    return normalized


def make_split(dataset, args):
    if args.split == 'object':
        return split_by_object(dataset, test_objects=args.test_objects)
    if args.split == 'pose':
        return split_by_pose(dataset, test_pose_indices=args.test_poses)
    return uniform_random_split(dataset)


def pad_along_time(x: torch.Tensor, target_T: int, pad_value=0):
    T = x.shape[0]
    if T == target_T:
        return x
    pad_shape = (target_T - T, *x.shape[1:])
    pad = torch.full(pad_shape, pad_value, dtype=x.dtype)
    return torch.cat([x, pad], dim=0)


def collate_full_by_second(batch):
    max_T = max(item['seconds'].shape[0] for item in batch)
    out: Dict[str, object] = {}

    def stack_time_key(key: str, pad_value=0):
        out[key] = torch.stack([pad_along_time(item[key], max_T, pad_value=pad_value) for item in batch])

    # metadata
    out['label'] = torch.stack([item['label'] for item in batch])
    out['pose_label'] = torch.stack([item['pose_label'] for item in batch])
    out['grasp_label'] = torch.stack([item['grasp_label'] for item in batch])
    out['gripper_force'] = torch.stack([item['gripper_force'] for item in batch])
    out['seconds'] = torch.stack([pad_along_time(item['seconds'], max_T, pad_value=-1) for item in batch])
    out['sequence_mask'] = out['seconds'] >= 0
    out['object'] = [item['object'] for item in batch]
    out['sample_dir'] = [item['sample_dir'] for item in batch]

    maybe_keys = set().union(*(item.keys() for item in batch))
    for key in sorted(maybe_keys):
        if key in out or key in {'time_layout', 'padding_metadata', 'modalities', 'stage_names', 'raw_label_rows', 'ft_columns', 'gripper_columns', 'robot_columns', 'pose_idx', 'force', 'entry_start_timestamp', 'entry_end_timestamp', 'stage_timestamps'}:
            continue
        sample_value = next(item[key] for item in batch if key in item)
        if not isinstance(sample_value, torch.Tensor):
            continue
        if sample_value.ndim == 0:
            out[key] = torch.stack([item[key] for item in batch])
        elif sample_value.shape[0] == batch[0]['seconds'].shape[0]:
            pad_value = False if sample_value.dtype == torch.bool else (-1 if 'second_timestamps' in key or 'frame_indices' in key else (float('nan') if sample_value.dtype.is_floating_point and 'timestamps' in key else 0))
            out[key] = torch.stack([pad_along_time(item[key], max_T, pad_value=pad_value) for item in batch])
        else:
            try:
                out[key] = torch.stack([item[key] for item in batch])
            except Exception:
                pass
    return out


def batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds_all = []
    labels_all = []
    n = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        logits = model(batch)
        labels = batch['label'].float()
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.shape[0]
        preds_all.append((logits > 0).cpu())
        labels_all.append(labels.bool().cpu())
        n += labels.shape[0]
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    preds = torch.cat(preds_all)
    labels = torch.cat(labels_all)
    tp = (preds & labels).sum().item()
    fp = (preds & ~labels).sum().item()
    fn = (~preds & labels).sum().item()
    tn = (~preds & ~labels).sum().item()
    acc = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return total_loss / n, acc, precision, recall, f1


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    dataset_modalities = normalize_modalities(args.modalities)
    ds = PoseItDataLoaderFull(
        root_dir=args.root_dir,
        modalities=dataset_modalities,
        time_layout=PoseItTimeLayout.BY_SECOND,
        padding_metadata=PoseItPaddingMetadata.BOTH,
    )

    if args.subsample < 1.0:
        import random
        k = max(4, int(len(ds.samples) * args.subsample))
        ds.samples = random.sample(ds.samples, k)
        print(f'Subsampled to {len(ds.samples)} samples')

    if args.overfit:
        overfit_set = Subset(ds, [0])
        train_set = val_set = test_set = overfit_set
        print('Overfit mode: 1 sample for train/val/test')
    else:
        train_set, val_set, test_set = make_split(ds, args)
        print(f'Split ({args.split}): train={len(train_set)}, val={len(val_set)}, test={len(test_set)}')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=not args.overfit, num_workers=args.num_workers, collate_fn=collate_full_by_second)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_full_by_second)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_full_by_second)

    train_labels = [int(ds.samples[i]['labels'].get('stability', 'drop') != 'pass') for i in train_set.indices]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    print(f'pos_weight={pos_weight.item():.3f} (n_pos={n_pos}, n_neg={n_neg})')

    model = SlipTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        modalities=args.modalities,
        max_seconds=args.max_seconds,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    save_dir = os.path.dirname(args.model_save_path) or '.'
    os.makedirs(save_dir, exist_ok=True)
    best_val_f1 = -1.0
    iteration = 0

    while iteration < args.n_iters:
        model.train()
        for batch in train_loader:
            if iteration >= args.n_iters:
                break
            batch = batch_to_device(batch, device)
            optimizer.zero_grad()
            logits = model(batch)
            labels = batch['label'].float()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            if iteration % args.log_interval == 0 or iteration == args.n_iters - 1:
                val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, criterion, device)
                print(f'[iter {iteration:03d}] train_loss={loss.item():.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f} prec={val_prec:.3f} rec={val_rec:.3f} f1={val_f1:.3f}')
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    torch.save(model.state_dict(), args.model_save_path)
            iteration += 1

    if os.path.exists(args.model_save_path):
        model.load_state_dict(torch.load(args.model_save_path, map_location=device))
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, criterion, device)
    print(f'Test loss={test_loss:.4f} acc={test_acc:.3f} prec={test_prec:.3f} rec={test_rec:.3f} f1={test_f1:.3f}')


if __name__ == '__main__':
    main()
