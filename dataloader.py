"""
PoseIt Dataloader

Folder name format: <object>_<timestamp>_F<force>_pose<idx>
  e.g. bangle_1612630172_F80_pose8

stages.csv  : phase_name, unix_timestamp  (integer seconds)
label.csv   : phase_name, result  (pass / slip / drop)
f_t.csv     : time, Fx, Fy, Fz, Tx, Ty, Tz       @ 70 Hz  (integer second timestamps)
gripper.csv : timestamp, left, right             @ 10 Hz  (integer second timestamps)
gelsight/   : .JPG frames                        @ 22-26 Hz
rgb/        : .JPG frames                        @ <10 Hz

Sampling strategy:
  - 20 consecutive integer seconds starting from t_grasp
  - For each second, collect all rows/frames with that timestamp
  - Uniformly sample exactly F items from that bucket
  - Concatenate across 20 seconds

Output shapes:
  ft      : (20, F*6)         — F/T readings per second, flattened
  gripper : (20, F*2)         — gripper readings per second, flattened
  tactile : (20, F, 3, H, W)  — F GelSight frames per second
  rgb     : (20, F, 3, H, W)  — F RGB frames per second

If a bucket has fewer than F items, the last item is forward-filled.
If a bucket is completely empty, zeros / black frames are used.
"""

import re
import csv
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


N_TIMESTEPS = 20   # number of consecutive seconds used as input
LABEL_MAP   = {'pass': 0, 'slip': 1, 'drop': 1}
IMAGE_SIZE  = (224, 224)

F1 = 2
F2 = 2              # items uniformly sampled per second bucket

FT_DIM  = F2 * 6   # F/T  feature size per timestep
GR_DIM  = F2 * 2   # gripper feature size per timestep

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# TODO @bayapilla: If the only consumer of the image is the encoder, then replace with 
'''
self.resnet_transforms = self.resnet_weights.transforms()

'''


def _parse_folder_name(name: str) -> dict:
    m = re.match(r'^(.+)_(\d+)_F(\d+)_pose(\d+)$', name)
    if not m:
        raise ValueError(f"Unexpected folder name: {name}")
    return {
        'object':   m.group(1),
        'start_ts': int(m.group(2)),
        'force':    float(m.group(3)),
        'pose_idx': int(m.group(4)),
    }


def _read_stages(path: Path) -> dict:
    stages = {}
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                try:
                    stages[row[0].strip()] = int(float(row[1].strip()))
                except ValueError:
                    pass
    return stages


def _read_labels(path: Path) -> dict:
    labels = {}
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].strip() in ('grasp', 'pose', 'stability'):
                labels[row[0].strip()] = row[1].strip().lower()
    return labels


def _read_csv_timeseries(path: Path, time_col: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Return (timestamps_int, values), skipping header."""
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            try:
                rows.append([float(v) for v in row])
            except ValueError:
                continue
    if not rows:
        return np.array([], dtype=np.int64), np.array([]).reshape(0, 0)
    arr = np.array(rows)
    ts  = arr[:, time_col].astype(np.int64)
    val = np.delete(arr, time_col, axis=1).astype(np.float32)
    return ts, val


def _sample_bucket(items: np.ndarray, n_cols: int, f: int) -> np.ndarray:
    """
    Uniformly sample exactly f rows from items (shape: k x n_cols).
    Returns flat array (f * n_cols,).
      k >= f : linspace index picks
      0 < k < f : forward-fill last row
      k == 0 : zeros
    """
    k = len(items)
    if k == 0:
        return np.zeros(f * n_cols, dtype=np.float32)
    if k >= f:
        idx    = np.round(np.linspace(0, k - 1, f)).astype(int)
        picked = items[idx]
    else:
        pad    = np.tile(items[-1:], (f - k, 1))
        picked = np.vstack([items, pad])
    return picked.flatten().astype(np.float32)


def _sample_image_bucket(paths: List[Path], f: int) -> List[Path]:
    """
    Uniformly sample exactly f paths from a bucket.
      len >= f : linspace index picks
      0 < len < f : forward-fill last path
      len == 0 : [None] * f
    """
    k = len(paths)
    if k == 0:
        return [None] * f
    if k >= f:
        idx = np.round(np.linspace(0, k - 1, f)).astype(int)
        return [paths[i] for i in idx]
    else:
        return paths + [paths[-1]] * (f - k)


def _list_image_files(folder: Path) -> List[Tuple[int, Path]]:
    """Return sorted list of (timestamp_int, path)."""
    pairs = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'):
            m = re.search(r'(\d+)', p.stem)
            if m:
                pairs.append((int(m.group(1)), p))
    pairs.sort(key=lambda x: x[0])
    return pairs


def _load_image(path: Optional[Path]) -> torch.Tensor:
    if path is None or not path.exists():
        return torch.zeros(3, *IMAGE_SIZE)
    return IMG_TRANSFORM(Image.open(path).convert('RGB'))


def _build_sample(sample_dir: Path) -> Optional[dict]:
    meta   = _parse_folder_name(sample_dir.name)
    stages = _read_stages(sample_dir / 'stages.csv')
    labels = _read_labels(sample_dir / 'label.csv')

    shake_str = labels.get('stability', 'drop')
    pose_str  = labels.get('pose',      'drop')

    if shake_str not in LABEL_MAP or pose_str not in LABEL_MAP:
        return None
    if pose_str == 'drop':
        return None

    t_grasp = stages.get('grasping', stages.get('grasp'))
    if t_grasp is None:
        return None

    # 20 consecutive integer seconds starting from t_grasp
    seconds = [t_grasp + i for i in range(N_TIMESTEPS)]  # [t, t+1, ..., t+19]

    ft_ts, ft_val = _read_csv_timeseries(sample_dir / 'f_t.csv',     time_col=0)
    gr_ts, gr_val = _read_csv_timeseries(sample_dir / 'gripper.csv', time_col=0)

    gel_files = _list_image_files(sample_dir / 'gelsight')
    rgb_files = _list_image_files(sample_dir / 'rgb')

    # Group image paths by integer second
    gel_by_sec = {}
    for ts, p in gel_files:
        gel_by_sec.setdefault(ts, []).append(p)

    rgb_by_sec = {}
    for ts, p in rgb_files:
        rgb_by_sec.setdefault(ts, []).append(p)

    # GelSight baseline: first frame of the first second
    baseline_paths = gel_by_sec.get(t_grasp, [])
    baseline = _load_image(baseline_paths[0] if baseline_paths else None)

    ft_seq      = []
    gr_seq      = []
    tactile_seq = []
    rgb_seq     = []

    for sec in seconds:
        # F/T: all rows with this integer timestamp
        ft_mask  = ft_ts == sec
        ft_bucket = ft_val[ft_mask]                                    # (k, 6)
        ft_seq.append(_sample_bucket(ft_bucket, n_cols=6, f=F2))       # (F2*6,)

        # Gripper
        gr_mask   = gr_ts == sec
        gr_bucket = gr_val[gr_mask]                                    # (k, 2)
        gr_seq.append(_sample_bucket(gr_bucket, n_cols=2, f=F2))       # (F2*2,)

        # GelSight: F frames from this second, subtract baseline
        gel_paths  = _sample_image_bucket(gel_by_sec.get(sec, []), f=F1) # [F paths]
        gel_frames = torch.stack([_load_image(p) - baseline for p in gel_paths])
        tactile_seq.append(gel_frames)                                 # (F, 3, H, W)

        # RGB: F frames from this second
        rgb_paths  = _sample_image_bucket(rgb_by_sec.get(sec, []), f=F1) # [F paths]
        rgb_frames = torch.stack([_load_image(p) for p in rgb_paths])
        rgb_seq.append(rgb_frames)                                     # (F, 3, H, W)

    return {
        'tactile':       torch.stack(tactile_seq),                          # (20, F1, 3, H, W)
        'rgb':           torch.stack(rgb_seq),                              # (20, F1, 3, H, W)
        'ft':            torch.tensor(np.stack(ft_seq), dtype=torch.float32),   # (20, F2*6)
        'gripper':       torch.tensor(np.stack(gr_seq), dtype=torch.float32),   # (20, F2*2)
        'gripper_force': torch.tensor([meta['force']], dtype=torch.float32),    # (1,)
        'label':         torch.tensor(LABEL_MAP[shake_str], dtype=torch.long),
        'pose_label':    torch.tensor(LABEL_MAP[pose_str],  dtype=torch.long),
        'object':        meta['object'],
        'pose_idx':      meta['pose_idx'],
        'force':         meta['force'],
        'sample_dir':    str(sample_dir),
    }


class PoseItDataset(Dataset):
    def __init__(self,
                 root_dir: Optional[str] = None,
                 sample_dirs: Optional[List[str]] = None):
        assert root_dir or sample_dirs, "Provide root_dir or sample_dirs"
        dirs = [Path(d) for d in sample_dirs] if sample_dirs \
               else sorted(Path(root_dir).iterdir())

        self.samples = []
        skipped = 0
        for d in dirs:
            if not d.is_dir():
                continue
            try:
                s = _build_sample(d)
                if s is not None:
                    self.samples.append(s)
                else:
                    skipped += 1
            except Exception as e:
                print(f"[WARN] Skipping {d.name}: {e}")
                skipped += 1

        print(f"Loaded {len(self.samples)} samples ({skipped} skipped)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            s['tactile'],        # (20, F1, 3, H, W)
            s['rgb'],            # (20, F1, 3, H, W)
            s['ft'],             # (20, F2*6)
            s['gripper'],        # (20, F2*2)
            s['gripper_force'],  # (1,)
            s['label'],          # scalar
            s['pose_label'],     # scalar
        )


def split_by_object(dataset, test_objects: List[str], val_ratio: float = 0.1):
    test_idx  = [i for i, s in enumerate(dataset.samples) if s['object'] in test_objects]
    train_val = [i for i in range(len(dataset.samples)) if i not in set(test_idx)]
    np.random.shuffle(train_val)
    n_val = max(1, int(len(train_val) * val_ratio))
    return (
        torch.utils.data.Subset(dataset, train_val[n_val:]),
        torch.utils.data.Subset(dataset, train_val[:n_val]),
        torch.utils.data.Subset(dataset, test_idx),
    )


def split_by_pose(dataset, test_pose_indices: List[int], val_ratio: float = 0.1):
    test_idx  = [i for i, s in enumerate(dataset.samples) if s['pose_idx'] in test_pose_indices]
    train_val = [i for i in range(len(dataset.samples)) if i not in set(test_idx)]
    np.random.shuffle(train_val)
    n_val = max(1, int(len(train_val) * val_ratio))
    return (
        torch.utils.data.Subset(dataset, train_val[n_val:]),
        torch.utils.data.Subset(dataset, train_val[:n_val]),
        torch.utils.data.Subset(dataset, test_idx),
    )


def uniform_random_split(dataset, train_ratio: float = 0.6875, val_ratio: float = 0.15625):
    n   = len(dataset)
    idx = np.random.permutation(n)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return (
        torch.utils.data.Subset(dataset, idx[:n_train].tolist()),
        torch.utils.data.Subset(dataset, idx[n_train:n_train + n_val].tolist()),
        torch.utils.data.Subset(dataset, idx[n_train + n_val:].tolist()),
    )


if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else './data'
    ds   = PoseItDataset(root_dir=root)

    if len(ds) == 0:
        print("No samples found — check your root_dir path.")
    else:
        tac, rgb, ft, grip, gf, label, pose_label = ds[0]
        print(f"tactile      : {tac.shape}")   # (20, F1, 3, 224, 224)
        print(f"rgb          : {rgb.shape}")   # (20, F1, 3, 224, 224)
        print(f"ft           : {ft.shape}")    # (20, F2*6)
        print(f"gripper      : {grip.shape}")  # (20, F2*2)
        print(f"gripper_force: {gf.shape}")    # (1,)
        print(f"label        : {label}")
        print(f"pose_label   : {pose_label}")
        print(f"\nF2={F2}  ->  FT_DIM={FT_DIM}, GR_DIM={GR_DIM}")

        loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)
        tac_b, rgb_b, ft_b, grip_b, gf_b, lbl_b, pl_b = next(iter(loader))
        print(f"\nBatch shapes:")
        print(f"  tactile : {tac_b.shape}")   # (2, 20, F1, 3, 224, 224)
        print(f"  ft      : {ft_b.shape}")    # (2, 20, F2*6)
        print(f"  gripper : {grip_b.shape}")  # (2, 20, F2*2)