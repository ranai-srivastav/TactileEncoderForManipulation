"""
PoseIt Dataloader

Folder name format: <object>_<timestamp>_F<force>_pose<idx>
  e.g. bangle_1612630172_F80_pose8

stages.csv  : phase_name, unix_timestamp
label.csv   : phase_name, result  (pass / slip / drop)
f_t.csv     : time, Fx, Fy, Fz, Tx, Ty, Tz   @ 70 Hz
gripper.csv : not used in sequence (scalar force from folder name only)
gelsight/   : .JPG frames @ 22-26 Hz
rgb/        : .JPG frames @ <10 Hz

F/T uses mean of F2_FT readings falling within each timestep interval → (T, 6).
F2_FT is a configurable constant (default 2).
If an interval is empty, the nearest reading outside the interval is used.
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

N_TIMESTEPS = 20
LABEL_MAP   = {'pass': 0, 'slip': 1, 'drop': 1}
IMAGE_SIZE  = (224, 224)

F2_FT = 2   # min readings per interval to average for F/T; configurable

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _parse_folder_name(name: str) -> dict:
    """Parse object, start_ts, force, pose_idx from folder name."""
    m = re.match(r'^(.+)_(\d+)_F(\d+)_pose(\d+)$', name)
    if not m:
        raise ValueError(f"Unexpected folder name format: {name}")
    return {
        'object':    m.group(1),
        'start_ts':  int(m.group(2)),
        'force':     float(m.group(3)),
        'pose_idx':  int(m.group(4)),
    }


def _read_stages(path: Path) -> dict:
    """Return {phase_name: unix_timestamp}."""
    stages = {}
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                stages[row[0].strip()] = float(row[1].strip())
    return stages


def _read_labels(path: Path) -> dict:
    """Return {phase_name: label_string}."""
    labels = {}
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].strip() in ('grasp', 'pose', 'stability'):
                labels[row[0].strip()] = row[1].strip().lower()
    return labels


def _read_csv_timeseries(path: Path, time_col: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a CSV with a timestamp column.
    Returns (timestamps, values) where values shape is (N, num_other_cols).
    """
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader, None)          # skip header
        for row in reader:
            try:
                rows.append([float(v) for v in row])
            except ValueError:
                continue
    if not rows:
        return np.array([]), np.array([])
    arr = np.array(rows)
    ts  = arr[:, time_col]
    val = np.delete(arr, time_col, axis=1)
    return ts, val


def _mean_readings(ts_array: np.ndarray, val_array: np.ndarray,
                   t_lo: float, t_hi: float) -> np.ndarray:
    """
    Return the mean of all rows whose timestamp falls in [t_lo, t_hi).
    If the interval is empty, fall back to the single nearest reading.
    Returns shape (n_cols,).
    """
    mask = (ts_array >= t_lo) & (ts_array < t_hi)
    rows = val_array[mask]
    if len(rows) == 0:
        # fallback: nearest reading to interval midpoint
        mid = (t_lo + t_hi) / 2.0
        idx = int(np.argmin(np.abs(ts_array - mid)))
        return val_array[idx].astype(np.float32)
    return rows.mean(axis=0).astype(np.float32)


def _list_image_files(folder: Path) -> List[Tuple[float, Path]]:
    """
    Return sorted list of (timestamp_float, path) for all images in folder.
    Timestamp is parsed from filename stem (numeric part).
    """
    pairs = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'):
            m = re.search(r'(\d+(?:\.\d+)?)', p.stem)
            if m:
                pairs.append((float(m.group(1)), p))
    pairs.sort(key=lambda x: x[0])
    return pairs


def _nearest_image(image_files: List[Tuple[float, Path]],
                   query_ts: float) -> Optional[Path]:
    if not image_files:
        return None
    ts_arr = np.array([t for t, _ in image_files])
    idx    = int(np.argmin(np.abs(ts_arr - query_ts)))
    return image_files[idx][1]


def _load_image(path: Optional[Path]) -> torch.Tensor:
    if path is None or not path.exists():
        return torch.zeros(3, *IMAGE_SIZE)
    img = Image.open(path).convert('RGB')
    return IMG_TRANSFORM(img)


# ─── Core Sample Builder ──────────────────────────────────────────────────────

def _build_sample(sample_dir: Path) -> Optional[dict]:
    meta   = _parse_folder_name(sample_dir.name)
    stages = _read_stages(sample_dir / 'stages.csv')
    labels = _read_labels(sample_dir / 'label.csv')

    # ── Labels ────────────────────────────────────────────────────────────────
    shake_str = labels.get('stability', 'drop')
    pose_str  = labels.get('pose',      'drop')

    if shake_str not in LABEL_MAP or pose_str not in LABEL_MAP:
        return None                          # malformed label, skip

    shake_label = LABEL_MAP[shake_str]
    pose_label  = LABEL_MAP[pose_str]

    # Skip samples where object was already dropped before shaking
    if pose_str == 'drop':
        return None

    # ── Phase timestamps ───────────────────────────────────────────────────────
    t_grasp     = stages.get('grasping',  stages.get('grasp'))
    t_pose      = stages.get('pose')
    t_stability = stages.get('stability')

    if t_grasp is None or t_pose is None or t_stability is None:
        return None

    # Input window: grasp_start → stability_start (= end of pose phase)
    t_start = t_grasp
    t_end   = t_stability
    query_times = np.linspace(t_start, t_end, N_TIMESTEPS)

    # ── F/T sensor ────────────────────────────────────────────────────────────
    ft_ts, ft_val = _read_csv_timeseries(sample_dir / 'f_t.csv', time_col=0)

    # ── Image files ───────────────────────────────────────────────────────────
    gel_files = _list_image_files(sample_dir / 'gelsight')
    rgb_files = _list_image_files(sample_dir / 'rgb')

    # Pre-contact GelSight baseline: first frame at/after grasp timestamp
    baseline_path = _nearest_image(gel_files, t_grasp)
    baseline_img  = _load_image(baseline_path)   # (3, H, W)

    # ── Build per-timestep tensors ────────────────────────────────────────────
    # Interval edges: query_times defines N_TIMESTEPS points;
    # interval i spans [query_times[i], query_times[i+1]) with the last
    # interval closed at t_end.
    edges = np.append(query_times, t_end + 1e-6)   # N_TIMESTEPS+1 edges

    tactile_seq = []
    rgb_seq     = []
    ft_seq      = []

    for i, qt in enumerate(query_times):
        t_lo = edges[i]
        t_hi = edges[i + 1]

        # GelSight: nearest frame, subtract baseline
        gel_path = _nearest_image(gel_files, qt)
        gel_img  = _load_image(gel_path) - baseline_img
        tactile_seq.append(gel_img)

        # RGB: nearest frame
        rgb_path = _nearest_image(rgb_files, qt)
        rgb_seq.append(_load_image(rgb_path))

        # F/T: mean of all readings in interval → (6,)
        if ft_ts.size:
            ft_vec = _mean_readings(ft_ts, ft_val, t_lo, t_hi)
        else:
            ft_vec = np.zeros(6, dtype=np.float32)
        ft_seq.append(ft_vec)

    return {
        'tactile':       torch.stack(tactile_seq),                          # (T, 3, H, W)
        'rgb':           torch.stack(rgb_seq),                              # (T, 3, H, W)
        'ft':            torch.tensor(np.stack(ft_seq),                     # (T, 6)
                                      dtype=torch.float32),
        'gripper_force': torch.tensor([meta['force']], dtype=torch.float32),# (1,)
        'label':         torch.tensor(shake_label, dtype=torch.long),
        'pose_label':    torch.tensor(pose_label,  dtype=torch.long),
        'object':        meta['object'],
        'pose_idx':      meta['pose_idx'],
        'force':         meta['force'],
        'sample_dir':    str(sample_dir),
    }


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PoseItDataset(Dataset):
    """
    Args:
        root_dir   : path containing one sub-folder per experiment
        sample_dirs: optionally pass an explicit list of sub-folder paths
                     (used for train/val/test splits)
    """
    def __init__(self,
                 root_dir: Optional[str] = None,
                 sample_dirs: Optional[List[str]] = None):
        assert root_dir or sample_dirs, "Provide root_dir or sample_dirs"

        if sample_dirs:
            dirs = [Path(d) for d in sample_dirs]
        else:
            dirs = sorted(Path(root_dir).iterdir())

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
            s['tactile'],        # (T, 3, H, W)
            s['rgb'],            # (T, 3, H, W)
            s['ft'],             # (T, 6)  — mean of readings per interval
            s['gripper_force'],  # (1,)    — scalar from folder name
            s['label'],          # scalar
            s['pose_label'],     # scalar  — needed for DRS
        )


# ─── Train / Val / Test split helpers ─────────────────────────────────────────

def split_by_object(dataset: PoseItDataset,
                    test_objects: List[str],
                    val_ratio: float = 0.1):
    """
    Hold out all samples whose object name is in test_objects.
    Split remaining into train / val by val_ratio.
    """
    test_idx  = [i for i, s in enumerate(dataset.samples)
                 if s['object'] in test_objects]
    train_val = [i for i in range(len(dataset.samples))
                 if i not in set(test_idx)]

    np.random.shuffle(train_val)
    n_val  = max(1, int(len(train_val) * val_ratio))
    val_idx   = train_val[:n_val]
    train_idx = train_val[n_val:]

    return (
        torch.utils.data.Subset(dataset, train_idx),
        torch.utils.data.Subset(dataset, val_idx),
        torch.utils.data.Subset(dataset, test_idx),
    )


def split_by_pose(dataset: PoseItDataset,
                  test_pose_indices: List[int],
                  val_ratio: float = 0.1):
    """Hold out all samples whose pose_idx is in test_pose_indices."""
    test_idx  = [i for i, s in enumerate(dataset.samples)
                 if s['pose_idx'] in test_pose_indices]
    train_val = [i for i in range(len(dataset.samples))
                 if i not in set(test_idx)]

    np.random.shuffle(train_val)
    n_val  = max(1, int(len(train_val) * val_ratio))
    val_idx   = train_val[:n_val]
    train_idx = train_val[n_val:]

    return (
        torch.utils.data.Subset(dataset, train_idx),
        torch.utils.data.Subset(dataset, val_idx),
        torch.utils.data.Subset(dataset, test_idx),
    )


def uniform_random_split(dataset: PoseItDataset,
                         train_ratio: float = 0.6875,
                         val_ratio:   float = 0.15625):
    """68.75 / 15.625 / 15.625 split matching the paper."""
    n = len(dataset)
    idx = np.random.permutation(n)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return (
        torch.utils.data.Subset(dataset, idx[:n_train].tolist()),
        torch.utils.data.Subset(dataset, idx[n_train:n_train+n_val].tolist()),
        torch.utils.data.Subset(dataset, idx[n_train+n_val:].tolist()),
    )


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else './data'
    ds   = PoseItDataset(root_dir=root)

    if len(ds) == 0:
        print("No samples found — check your root_dir path.")
    else:
        tac, rgb, ft, gf, label, pose_label = ds[0]
        print(f"tactile      : {tac.shape}")    # (20, 3, 224, 224)
        print(f"rgb          : {rgb.shape}")    # (20, 3, 224, 224)
        print(f"ft           : {ft.shape}")     # (20, 6)
        print(f"gripper_force: {gf.shape}")     # (1,)
        print(f"label        : {label}")
        print(f"pose_label   : {pose_label}")

        loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)
        batch  = next(iter(loader))
        print(f"\nBatch tactile shape: {batch[0].shape}")   # (2, 20, 3, 224, 224)