from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from dataloader import split_by_object, split_by_pose, uniform_random_split
from dataloader_full import (
    PoseItDataLoaderFull,
    PoseItItemsPerSecond,
    PoseItModality,
    PoseItPaddingMetadata,
    PoseItTimeLayout,
)
from model_transformer import DEFAULT_MODALITIES, SlipTransformer

IMAGE_ENCODER_DEFAULTS: Dict[str, str] = {
    'vit': 'vit_tiny_patch16_224',
    'resnet': 'resnet18',
    'temporal_cnn': 'temporal_cnn_small',
}

SENSOR_CONFIGS: Dict[str, tuple[str, ...]] = {
    'all': tuple(DEFAULT_MODALITIES),
    'vision_only': ('tactile', 'rgb', 'depth', 'side_cam', 'top_cam'),
    'tactile_only': ('tactile',),
    'rgb_only': ('rgb',),
    'no_visual': ('ft', 'gripper', 'robot', 'gripper_force'),
    'proprio_only': ('ft', 'gripper', 'robot'),
    'rgb_ft_gripper': ('rgb', 'ft', 'gripper', 'gripper_force'),
    'all_no_robot': ('tactile', 'rgb', 'depth', 'side_cam', 'top_cam', 'ft', 'gripper', 'gripper_force'),
    'all_no_depth': ('tactile', 'rgb', 'side_cam', 'top_cam', 'ft', 'gripper', 'robot', 'gripper_force'),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train the hierarchical slip transformer on full-resolution PoseIt data.')
    parser.add_argument('--root_dir', type=str, default='./data')
    parser.add_argument('--split', type=str, default='random', choices=['object', 'pose', 'random'])
    parser.add_argument('--test_objects', nargs='+', default=['mug', 'bowl'])
    parser.add_argument('--val_objects', nargs='+', default=None)
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument('--test_poses', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument('--sensor_config', type=str, default='all', choices=sorted(SENSOR_CONFIGS))
    parser.add_argument('--modalities', nargs='+', default=None)
    parser.add_argument('--exclude_modalities', nargs='*', default=[])
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--overfit_all', action='store_true', help='Use the entire loaded dataset for train/val/test.')
    parser.add_argument('--n_iters', type=int, default=50)
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epochs between validation runs.')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_seconds', type=int, default=64)
    parser.add_argument('--image_encoder_type', type=str, default='vit', choices=sorted(IMAGE_ENCODER_DEFAULTS))
    parser.add_argument('--image_encoder_model_name', type=str, default=None)
    parser.add_argument('--vit_model_name', type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--vit_pretrained', action='store_true')
    parser.add_argument('--image_temporal_stem_channels', type=int, default=16)
    parser.add_argument('--image_temporal_branch_channels', type=int, default=48)
    parser.add_argument('--image_temporal_num_blocks', type=int, default=3)
    parser.add_argument('--image_temporal_spatial_downsample', type=int, default=4)
    parser.add_argument('--timeseries_conv_channels', type=int, default=128)
    parser.add_argument('--timeseries_num_conv_layers', type=int, default=3)
    parser.add_argument('--timeseries_kernel_size', type=int, default=5)
    parser.add_argument('--scalar_hidden_dim', type=int, default=128)
    parser.add_argument('--tactile_items_per_second', type=int, default=12)
    parser.add_argument('--rgb_items_per_second', type=int, default=5)
    parser.add_argument('--depth_items_per_second', type=int, default=5)
    parser.add_argument('--side_cam_items_per_second', type=int, default=8)
    parser.add_argument('--top_cam_items_per_second', type=int, default=8)
    parser.add_argument('--ft_items_per_second', type=int, default=100)
    parser.add_argument('--gripper_items_per_second', type=int, default=10)
    parser.add_argument('--robot_items_per_second', type=int, default=128)
    parser.add_argument('--rgb_feature_cache_dir', type=str, default=None)
    parser.add_argument('--model_save_path', type=str, default='trained_models/full_transformer_best.pt')
    parser.add_argument('--wandb_project', type=str, default='TEMU-fullres-transformer')
    parser.add_argument('--wandb_entity', type=str, default='mrsd-smores')
    parser.add_argument('--wandb_run', type=str, default=None)
    parser.add_argument('--wandb_mode', type=str, default='disabled', choices=['disabled', 'offline', 'online'])
    parser.add_argument('--smoke_test_only', action='store_true', help='Run one forward/backward smoke test and exit.')
    return parser.parse_args()


class _NullProgressBar:
    def __init__(self, total: int | None = None, desc: str = '') -> None:
        self.total = total
        self.desc = desc

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def update(self, _: int = 1) -> None:
        return None

    def set_postfix(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


def _make_progress(total: int | None, desc: str, leave: bool = True):
    if _tqdm is None:
        return _NullProgressBar(total=total, desc=desc)
    return _tqdm(total=total, desc=desc, dynamic_ncols=True, leave=leave)


def resolve_modalities(args: argparse.Namespace) -> List[str]:
    names = list(args.modalities) if args.modalities is not None else list(SENSOR_CONFIGS[args.sensor_config])
    if args.exclude_modalities:
        excluded = set(args.exclude_modalities)
        names = [name for name in names if name not in excluded]
    if not names:
        raise ValueError('Resolved modalities are empty after applying sensor config and exclusions.')
    invalid = sorted(set(names) - set(DEFAULT_MODALITIES))
    if invalid:
        raise ValueError(f'Unsupported modalities: {invalid}')
    return names


def resolve_image_encoder_model_name(args: argparse.Namespace) -> str:
    if args.image_encoder_model_name:
        return args.image_encoder_model_name
    if args.vit_model_name:
        return args.vit_model_name
    return IMAGE_ENCODER_DEFAULTS[args.image_encoder_type]


def normalize_modalities(names: Sequence[str]) -> List[PoseItModality]:
    normalized: List[PoseItModality] = []
    for name in names:
        if name == 'gripper_force':
            continue
        normalized.append(PoseItModality(name))
    return normalized


def build_items_per_second(args: argparse.Namespace) -> PoseItItemsPerSecond:
    return PoseItItemsPerSecond(
        tactile=args.tactile_items_per_second,
        rgb=args.rgb_items_per_second,
        depth=args.depth_items_per_second,
        side_cam=args.side_cam_items_per_second,
        top_cam=args.top_cam_items_per_second,
        ft=args.ft_items_per_second,
        gripper=args.gripper_items_per_second,
        robot=args.robot_items_per_second,
    )


def make_split(dataset: Dataset[Any], args: argparse.Namespace):
    if args.overfit_all:
        indices = list(range(len(dataset)))
        subset = Subset(dataset, indices)
        return subset, subset, subset
    if args.split == 'object':
        return split_by_object(
            dataset,
            test_objects=args.test_objects,
            val_objects=args.val_objects,
            seed=args.split_seed,
        )
    if args.split == 'pose':
        return split_by_pose(dataset, test_pose_indices=args.test_poses)
    return uniform_random_split(dataset)


def pad_along_time(tensor: torch.Tensor, target_seconds: int, pad_value: float | int | bool) -> torch.Tensor:
    current_seconds = tensor.shape[0]
    if current_seconds == target_seconds:
        return tensor
    pad_shape = (target_seconds - current_seconds, *tensor.shape[1:])
    padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype)
    return torch.cat([tensor, padding], dim=0)


def collate_full_by_second(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    max_seconds = max(item['seconds'].shape[0] for item in batch)
    collated: Dict[str, Any] = {
        'label': torch.stack([item['label'] for item in batch]),
        'pose_label': torch.stack([item['pose_label'] for item in batch]),
        'grasp_label': torch.stack([item['grasp_label'] for item in batch]),
        'gripper_force': torch.stack([item['gripper_force'] for item in batch]),
        'seconds': torch.stack([pad_along_time(item['seconds'], max_seconds, pad_value=-1) for item in batch]),
        'object': [item['object'] for item in batch],
        'sample_dir': [item['sample_dir'] for item in batch],
    }
    collated['sequence_mask'] = collated['seconds'] >= 0

    skip_non_tensor_keys = {
        'time_layout', 'padding_metadata', 'modalities', 'stage_names', 'raw_label_rows',
        'ft_columns', 'gripper_columns', 'robot_columns', 'object', 'sample_dir',
        'pose_idx', 'force', 'entry_start_timestamp', 'entry_end_timestamp', 'stage_timestamps',
        'seconds', 'label', 'pose_label', 'grasp_label', 'gripper_force',
    }
    all_keys = set().union(*(item.keys() for item in batch))
    for key in sorted(all_keys):
        if key in skip_non_tensor_keys:
            continue
        sample_value = next(item[key] for item in batch if key in item)
        if not isinstance(sample_value, torch.Tensor):
            continue
        assert sample_value.ndim >= 1
        if sample_value.shape[0] != batch[0]['seconds'].shape[0]:
            collated[key] = torch.stack([item[key] for item in batch])
            continue
        if sample_value.dtype == torch.bool:
            pad_value: float | int | bool = False
        elif 'second_timestamps' in key or 'frame_indices' in key:
            pad_value = -1
        elif 'timestamps' in key and sample_value.dtype.is_floating_point:
            pad_value = float('nan')
        else:
            pad_value = 0
        collated[key] = torch.stack([pad_along_time(item[key], max_seconds, pad_value=pad_value) for item in batch])
    return collated


def batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[Dict[str, Any]],
    criterion: nn.Module,
    device: torch.device,
    desc: str = 'Eval',
    show_progress: bool = True,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    pred_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []
    progress_cm = _make_progress(total=len(loader), desc=desc, leave=False) if show_progress else _NullProgressBar()
    with progress_cm as progress:
        for batch in loader:
            batch = batch_to_device(batch, device)
            logits = model(batch)
            labels = batch['label'].float()
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.shape[0]
            total_items += labels.shape[0]
            pred_chunks.append((logits > 0).cpu())
            label_chunks.append(labels.bool().cpu())
            if show_progress:
                progress.update(1)
    if total_items == 0:
        return {'loss': 0.0, 'acc': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    preds = torch.cat(pred_chunks)
    labels = torch.cat(label_chunks)
    tp = (preds & labels).sum().item()
    fp = (preds & ~labels).sum().item()
    fn = (~preds & labels).sum().item()
    tn = (~preds & ~labels).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = (tp + tn) / total_items
    return {'loss': total_loss / total_items, 'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1}


def maybe_init_wandb(args: argparse.Namespace) -> None:
    if args.wandb_mode == 'disabled':
        return
    if not WANDB_AVAILABLE:
        print('[WARN] wandb not installed; disabling wandb logging')
        return
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run,
        mode=args.wandb_mode,
        config=vars(args),
    )
    wandb.define_metric('iter')
    wandb.define_metric('epoch')
    wandb.define_metric('train/*', step_metric='iter')
    wandb.define_metric('val/*', step_metric='iter')
    wandb.define_metric('test/*', step_metric='iter')


def maybe_log(metrics: Mapping[str, float], step: int, epoch: int | None = None) -> None:
    if WANDB_AVAILABLE and wandb.run is not None:
        payload = {'iter': step}
        if epoch is not None:
            payload['epoch'] = epoch
        payload.update(metrics)
        wandb.log(payload, step=step)


def summarize_objects(dataset: Dataset[Any], subset: Subset[Any]) -> List[str]:
    return sorted({str(dataset.samples[i]['object']) for i in subset.indices})


def build_dataset(args: argparse.Namespace) -> PoseItDataLoaderFull:
    image_encoder_model_name = resolve_image_encoder_model_name(args)
    rgb_feature_cache_dir = args.rgb_feature_cache_dir if args.image_encoder_type != 'temporal_cnn' else None
    dataset = PoseItDataLoaderFull(
        root_dir=args.root_dir,
        modalities=normalize_modalities(args.modalities),
        time_layout=PoseItTimeLayout.BY_SECOND,
        padding_metadata=PoseItPaddingMetadata.BOTH,
        items_per_second=build_items_per_second(args),
        rgb_feature_cache_dir=rgb_feature_cache_dir,
        rgb_feature_cache_model_name=image_encoder_model_name,
    )
    if args.subsample < 1.0:
        import random
        k = max(4, int(len(dataset.samples) * args.subsample))
        dataset.samples = random.sample(dataset.samples, k)
        print(f'Subsampled dataset to {len(dataset.samples)} samples')
    return dataset


def build_model(args: argparse.Namespace) -> SlipTransformer:
    image_encoder_model_name = resolve_image_encoder_model_name(args)
    return SlipTransformer(
        d_model=args.d_model,
        image_encoder_type=args.image_encoder_type,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        modalities=args.modalities,
        max_seconds=args.max_seconds,
        image_encoder_model_name=image_encoder_model_name,
        vit_pretrained=args.vit_pretrained,
        image_temporal_stem_channels=args.image_temporal_stem_channels,
        image_temporal_branch_channels=args.image_temporal_branch_channels,
        image_temporal_num_blocks=args.image_temporal_num_blocks,
        image_temporal_spatial_downsample=args.image_temporal_spatial_downsample,
        max_items_per_second={
            'tactile': args.tactile_items_per_second,
            'rgb': args.rgb_items_per_second,
            'depth': args.depth_items_per_second,
            'side_cam': args.side_cam_items_per_second,
            'top_cam': args.top_cam_items_per_second,
            'ft': args.ft_items_per_second,
            'gripper': args.gripper_items_per_second,
            'robot': args.robot_items_per_second,
        },
        timeseries_conv_channels=args.timeseries_conv_channels,
        timeseries_num_conv_layers=args.timeseries_num_conv_layers,
        timeseries_kernel_size=args.timeseries_kernel_size,
        scalar_hidden_dim=args.scalar_hidden_dim,
    )


def run_smoke_test(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = build_dataset(args)
    train_set, _, _ = make_split(dataset, argparse.Namespace(**{**vars(args), 'overfit_all': True}))
    loader = DataLoader(train_set, batch_size=min(args.batch_size, len(train_set)), shuffle=False, num_workers=0, collate_fn=collate_full_by_second)
    batch = next(iter(loader))
    batch = batch_to_device(batch, device)
    model = build_model(args).to(device)
    logits = model(batch)
    loss = nn.BCEWithLogitsLoss()(logits, batch['label'].float())
    loss.backward()
    assert logits.shape == (batch['label'].shape[0],)
    print('Smoke test passed:', tuple(logits.shape), 'loss=', float(loss.item()))


def main() -> None:
    args = parse_args()
    args.modalities = resolve_modalities(args)
    args.image_encoder_model_name = resolve_image_encoder_model_name(args)
    if args.smoke_test_only:
        run_smoke_test(args)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    maybe_init_wandb(args)

    print('Building dataset...')
    dataset = build_dataset(args)
    print('Creating split...')
    train_set, val_set, test_set = make_split(dataset, args)
    print(f'Split ({args.split}): train={len(train_set)}, val={len(val_set)}, test={len(test_set)}')
    print(f'Modalities: {args.modalities}')
    print(f'Image encoder: type={args.image_encoder_type} model={args.image_encoder_model_name}')
    if args.image_encoder_type == 'temporal_cnn' and args.rgb_feature_cache_dir is not None:
        print('RGB feature cache parent: ignored for temporal_cnn')
    elif args.rgb_feature_cache_dir is not None:
        print(f'RGB feature cache parent: {args.rgb_feature_cache_dir}')
    if args.split == 'object':
        print(f'Train objects: {summarize_objects(dataset, train_set)}')
        print(f'Val objects: {summarize_objects(dataset, val_set)}')
        print(f'Test objects: {summarize_objects(dataset, test_set)}')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=not args.overfit_all, num_workers=args.num_workers, collate_fn=collate_full_by_second)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_full_by_second)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_full_by_second)

    train_labels = [int(dataset.samples[i]['labels'].get('stability', 'drop') != 'pass') for i in train_set.indices]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    print(f'pos_weight={pos_weight.item():.3f} (n_pos={n_pos}, n_neg={n_neg})')

    print('Building model...')
    model = build_model(args).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    save_dir = os.path.dirname(args.model_save_path) or '.'
    os.makedirs(save_dir, exist_ok=True)
    best_val_f1 = -1.0
    iteration = 0
    epoch = 0
    last_train_loss = 0.0
    with _make_progress(total=args.n_iters, desc='Training', leave=True) as train_progress:
        while iteration < args.n_iters:
            epoch += 1
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
                last_train_loss = float(loss.item())
                train_progress.update(1)
                if iteration % args.log_interval == 0 or iteration == args.n_iters - 1:
                    train_progress.set_postfix(train_loss=f'{last_train_loss:.4f}', epoch=epoch)
                maybe_log({
                    'train/loss': last_train_loss,
                }, step=iteration, epoch=epoch)
                iteration += 1

            should_validate = epoch % args.val_interval == 0 or iteration >= args.n_iters
            if should_validate and iteration > 0:
                val_metrics = evaluate(model, val_loader, criterion, device, desc=f'Validation (epoch {epoch})')
                print(
                    f"[epoch {epoch:03d} iter {iteration:03d}] train_loss={last_train_loss:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.3f} "
                    f"val_f1={val_metrics['f1']:.3f}"
                )
                maybe_log({
                    'train/loss': last_train_loss,
                    'val/loss': val_metrics['loss'],
                    'val/acc': val_metrics['acc'],
                    'val/precision': val_metrics['precision'],
                    'val/recall': val_metrics['recall'],
                    'val/f1': val_metrics['f1'],
                }, step=max(iteration - 1, 0), epoch=epoch)
                if val_metrics['f1'] > best_val_f1:
                    best_val_f1 = val_metrics['f1']
                    torch.save(model.state_dict(), args.model_save_path)

    if os.path.exists(args.model_save_path):
        model.load_state_dict(torch.load(args.model_save_path, map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device, desc='Test')
    print(
        f"Test loss={test_metrics['loss']:.4f} acc={test_metrics['acc']:.3f} "
        f"prec={test_metrics['precision']:.3f} rec={test_metrics['recall']:.3f} f1={test_metrics['f1']:.3f}"
    )
    maybe_log({
        'test/loss': test_metrics['loss'],
        'test/acc': test_metrics['acc'],
        'test/precision': test_metrics['precision'],
        'test/recall': test_metrics['recall'],
        'test/f1': test_metrics['f1'],
    }, step=args.n_iters, epoch=epoch)
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


if __name__ == '__main__':
    main()
