from __future__ import annotations

import argparse
from pathlib import Path

import torch
import timm
from tqdm.auto import tqdm
from PIL import Image

from dataloader_full import RGB_TRANSFORM, _list_image_files, _rgb_feature_cache_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Precompute frozen frame-encoder features for RGB images.')
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, required=True)
    parser.add_argument('--image_encoder_model_name', type=str, default='resnet18')
    parser.add_argument('--frame_model_name', type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


@torch.no_grad()
def encode_rgb_entries(model: torch.nn.Module, entries, device: torch.device, batch_size: int) -> torch.Tensor:
    chunks = []
    for start in tqdm(range(0, len(entries), batch_size), desc='  Frames', leave=False, dynamic_ncols=True):
        batch_entries = entries[start:start + batch_size]
        images = []
        for _, _, path in batch_entries:
            with Image.open(path) as image:
                images.append(RGB_TRANSFORM(image.convert('RGB')))
        batch = torch.stack(images, dim=0).to(device)
        chunks.append(model(batch).cpu().to(dtype=torch.float16))
    return torch.cat(chunks, dim=0) if chunks else torch.zeros((0, model.num_features), dtype=torch.float16)


def main() -> None:
    args = parse_args()
    if args.frame_model_name is not None and args.image_encoder_model_name == 'resnet18':
        args.image_encoder_model_name = args.frame_model_name
    root_dir = Path(args.root_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = timm.create_model(args.image_encoder_model_name, pretrained=False, num_classes=0, in_chans=3).to(device)
    model.eval()

    sample_dirs = sorted(path for path in root_dir.iterdir() if path.is_dir())
    model_cache_dir = cache_dir / ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '_' for ch in args.image_encoder_model_name)
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    print(f'Caching RGB frame features with model={args.image_encoder_model_name} into {model_cache_dir}')
    for sample_dir in tqdm(sample_dirs, desc='Precomputing RGB feature cache', unit='sample', dynamic_ncols=True):
        cache_path = _rgb_feature_cache_path(cache_dir, args.image_encoder_model_name, sample_dir)
        if cache_path.exists() and not args.force:
            continue

        entries = _list_image_files(sample_dir / 'rgb')
        second_timestamps = torch.tensor([ts for ts, _, _ in entries], dtype=torch.long)
        frame_indices = torch.tensor([frame_idx for _, frame_idx, _ in entries], dtype=torch.long)
        features = encode_rgb_entries(model, entries, device=device, batch_size=args.batch_size)

        torch.save(
            {
                'features': features,
                'second_timestamps': second_timestamps,
                'frame_indices': frame_indices,
                'frame_model_name': args.image_encoder_model_name,
            },
            cache_path,
        )


if __name__ == '__main__':
    main()
