"""
PoseIt Dataloader

Folder name format: <object>_<timestamp>_F<force>_pose<idx>
  e.g. flashlight_1612558475_F80_pose8

stages.csv  : phase_name, unix_timestamp  (integer seconds)
label.csv   : phase_name, result  (pass / slip / drop)
f_t.csv     : time, Fx, Fy, Fz, Tx, Ty, Tz       @ ~70 Hz  (integer second timestamps)
gripper.csv : timestamp, left, right               @ ~10 Hz  (integer second timestamps)
gelsight/   : gelsight_{frame_idx}_{unix_ts}.jpg   @ ~22-26 Hz
rgb/        : rgb_{frame_idx}_{unix_ts}.jpg        @ ~10 Hz

Sampling strategy:
  - Window: the last L integer seconds of the grasping phase,
    i.e. [t_end - L, t_end) where t_end = stages['pose'].
  - For each second, collect all rows/frames with that timestamp.
  - Uniformly sample exactly F items from that bucket (deterministic: linspace).
  - Concatenate across L seconds.
  - Samples where the grasping phase is shorter than L seconds are skipped.

Output shapes (per sample):
  ft       : (L, F*6)         — F/T readings per second, flattened
  gripper  : (L, F*2)         — gripper readings per second, flattened
  tactile  : (L, F, 3, H, W)  — F GelSight difference frames per second
  rgb      : (L, F, 3, H, W)  — F RGB frames per second

If a bucket has fewer than F items, the last item is forward-filled.
If a bucket is completely empty, zeros / black frames are used.

GelSight frames are baseline-subtracted: each frame minus the first
GelSight frame captured at t_grasp (the start of the grasping phase).
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


LABEL_MAP = {'pass': 0, 'slip': 1, 'drop': 1}
IMAGE_SIZE = (224, 224)

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# CSV readers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Deterministic sampling helpers
# ---------------------------------------------------------------------------

def _sample_bucket(items: np.ndarray, n_cols: int, f: int) -> np.ndarray:
    """
    Uniformly sample exactly f rows from items (shape: k x n_cols).
    Returns flat array (f * n_cols,).
      k >= f : linspace index picks  (deterministic)
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


def _sample_image_bucket(paths: List[Path], f: int) -> List[Optional[Path]]:
    """
    Uniformly sample exactly f paths from a bucket.
      len >= f : linspace index picks  (deterministic)
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


def _list_image_files(folder: Path) -> List[Tuple[int, int, Path]]:
    """
    Return sorted list of (unix_timestamp, frame_idx, path).

    Filename format: {modality}_{frame_idx}_{unix_ts}.ext
    We extract the unix timestamp from the LAST digit group in the stem,
    and the frame index from the second-to-last digit group.
    Sorted by (unix_timestamp, frame_idx) for determinism.
    """
    triples = []
    for p in folder.iterdir():
        if p.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'):
            continue
        numbers = re.findall(r'\d+', p.stem)
        if len(numbers) < 2:
            continue
        unix_ts   = int(numbers[-1])   # last number  = unix timestamp
        frame_idx = int(numbers[-2])   # second-to-last = frame index
        triples.append((unix_ts, frame_idx, p))
    triples.sort(key=lambda x: (x[0], x[1]))
    return triples


def _load_image(path: Optional[Path]) -> torch.Tensor:
    if path is None or not path.exists():
        return torch.zeros(3, *IMAGE_SIZE)
    return IMG_TRANSFORM(Image.open(path).convert('RGB'))


# ---------------------------------------------------------------------------
# Metadata builder (called once per folder in __init__)
# ---------------------------------------------------------------------------

def _build_meta(sample_dir: Path, L: int) -> Optional[dict]:
    """
    Parse everything except pixel data.  Returns None if the sample
    should be skipped (bad labels, missing stages, grasping phase < L).
    """
    try:
        meta = _parse_folder_name(sample_dir.name)
    except ValueError:
        return None

    stages = _read_stages(sample_dir / 'stages.csv')
    labels = _read_labels(sample_dir / 'label.csv')

    shake_str = labels.get('stability', 'drop')
    pose_str  = labels.get('pose',      'drop')
    if shake_str not in LABEL_MAP or pose_str not in LABEL_MAP:
        return None

    t_grasp = stages.get('grasping') or stages.get('grasp')
    t_end   = stages.get('pose')       # end of grasping = start of pose phase
    if t_grasp is None or t_end is None:
        return None
    if (t_end - t_grasp) < L:
        return None                    # grasping phase too short for this L

    # Load CSVs fully (small: ~200 KB each) — safe to keep in memory
    ft_ts,  ft_val  = _read_csv_timeseries(sample_dir / 'f_t.csv',     time_col=0)
    gr_ts,  gr_val  = _read_csv_timeseries(sample_dir / 'gripper.csv', time_col=0)

    # Build image-path lookup: second → [path, ...]
    gel_triples = _list_image_files(sample_dir / 'gelsight')
    rgb_triples = _list_image_files(sample_dir / 'rgb')

    gel_by_sec: dict = {}
    for unix_ts, _, p in gel_triples:
        gel_by_sec.setdefault(unix_ts, []).append(p)

    rgb_by_sec: dict = {}
    for unix_ts, _, p in rgb_triples:
        rgb_by_sec.setdefault(unix_ts, []).append(p)

    # GelSight baseline: first frame at t_grasp
    baseline_candidates = gel_by_sec.get(t_grasp, [])
    baseline_path = baseline_candidates[0] if baseline_candidates else None

    return {
        # temporal anchors
        't_grasp':      t_grasp,
        't_end':        t_end,
        # image lookups (paths only, no pixel data)
        'gel_by_sec':   gel_by_sec,
        'rgb_by_sec':   rgb_by_sec,
        'baseline_path': baseline_path,
        # sensor arrays (small, kept in memory)
        'ft_ts':        ft_ts,
        'ft_val':       ft_val,
        'gr_ts':        gr_ts,
        'gr_val':       gr_val,
        # labels (scalars, needed by DRSSampler)
        'label':        LABEL_MAP[shake_str],
        'pose_label':   LABEL_MAP[pose_str],
        # metadata
        'object':       meta['object'],
        'pose_idx':     meta['pose_idx'],
        'force':        meta['force'],
        'sample_dir':   str(sample_dir),
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PoseItDataset(Dataset):
    """
    Args:
        root_dir:    Path to folder containing experiment subdirectories.
        L:           Number of seconds in the temporal window.
                     Window = [t_end - L, t_end) where t_end = stages['pose'].
        F:           Number of items to sample per second (all modalities).
        image_size:  (H, W) to resize images to before encoding.
    """

    def __init__(
        self,
        root_dir: str,
        L: int = 5,
        F: int = 2,
        image_size: Tuple[int, int] = IMAGE_SIZE,
    ):
        self.L = L
        self.F = F
        self.image_size = image_size

        self.samples: List[dict] = []
        skipped = 0
        for d in sorted(Path(root_dir).iterdir()):
            if not d.is_dir():
                continue
            try:
                m = _build_meta(d, L)
                if m is not None:
                    self.samples.append(m)
                else:
                    skipped += 1
            except Exception as e:
                print(f"[WARN] Skipping {d.name}: {e}")
                skipped += 1

        print(f"Loaded {len(self.samples)} samples ({skipped} skipped)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s  = self.samples[idx]
        L  = self.L
        F  = self.F

        # Window: last L seconds of grasping phase
        t_end   = s['t_end']
        seconds = range(t_end - L, t_end)   # [t_end-L, t_end-L+1, ..., t_end-1]

        # GelSight baseline (change image reference)
        baseline = _load_image(s['baseline_path'])   # (3, H, W)

        ft_seq      = []
        gr_seq      = []
        tactile_seq = []
        rgb_seq     = []

        for sec in seconds:
            # --- F/T ---
            ft_mask   = s['ft_ts'] == sec
            ft_bucket = s['ft_val'][ft_mask]                        # (k, 6)
            ft_seq.append(_sample_bucket(ft_bucket, n_cols=6, f=F)) # (F*6,)

            # --- Gripper ---
            gr_mask   = s['gr_ts'] == sec
            gr_bucket = s['gr_val'][gr_mask]                        # (k, 2)
            gr_seq.append(_sample_bucket(gr_bucket, n_cols=2, f=F)) # (F*2,)

            # --- GelSight (baseline-subtracted) ---
            gel_paths  = _sample_image_bucket(s['gel_by_sec'].get(sec, []), F)
            gel_frames = torch.stack([_load_image(p) - baseline for p in gel_paths])
            tactile_seq.append(gel_frames)                          # (F, 3, H, W)

            # --- RGB ---
            rgb_paths  = _sample_image_bucket(s['rgb_by_sec'].get(sec, []), F)
            rgb_frames = torch.stack([_load_image(p) for p in rgb_paths])
            rgb_seq.append(rgb_frames)                              # (F, 3, H, W)

        return {
            'tactile':       torch.stack(tactile_seq),                              # (L, F, 3, H, W)
            'rgb':           torch.stack(rgb_seq),                                  # (L, F, 3, H, W)
            'ft':            torch.tensor(np.stack(ft_seq),  dtype=torch.float32),  # (L, F*6)
            'gripper':       torch.tensor(np.stack(gr_seq),  dtype=torch.float32),  # (L, F*2)
            'gripper_force': torch.tensor([s['force']],       dtype=torch.float32), # (1,)
            'label':         torch.tensor(s['label'],          dtype=torch.long),
            'pose_label':    torch.tensor(s['pose_label'],     dtype=torch.long),
            'object':        s['object'],
            'pose_idx':      s['pose_idx'],
            'force':         s['force'],
            'sample_dir':    s['sample_dir'],
        }


# ---------------------------------------------------------------------------
# Dataset splits
# ---------------------------------------------------------------------------

def split_by_object(dataset: PoseItDataset, test_objects: List[str], val_ratio: float = 0.1):
    test_idx  = [i for i, s in enumerate(dataset.samples) if s['object'] in test_objects]
    train_val = [i for i in range(len(dataset.samples)) if i not in set(test_idx)]
    np.random.shuffle(train_val)
    n_val = max(1, int(len(train_val) * val_ratio))
    return (
        torch.utils.data.Subset(dataset, train_val[n_val:]),
        torch.utils.data.Subset(dataset, train_val[:n_val]),
        torch.utils.data.Subset(dataset, test_idx),
    )


def split_by_pose(dataset: PoseItDataset, test_pose_indices: List[int], val_ratio: float = 0.1):
    test_idx  = [i for i, s in enumerate(dataset.samples) if s['pose_idx'] in test_pose_indices]
    train_val = [i for i in range(len(dataset.samples)) if i not in set(test_idx)]
    np.random.shuffle(train_val)
    n_val = max(1, int(len(train_val) * val_ratio))
    return (
        torch.utils.data.Subset(dataset, train_val[n_val:]),
        torch.utils.data.Subset(dataset, train_val[:n_val]),
        torch.utils.data.Subset(dataset, test_idx),
    )


def uniform_random_split(dataset: PoseItDataset, train_ratio: float = 0.6875, val_ratio: float = 0.15625):
    n   = len(dataset)
    idx = np.random.permutation(n)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return (
        torch.utils.data.Subset(dataset, idx[:n_train].tolist()),
        torch.utils.data.Subset(dataset, idx[n_train:n_train + n_val].tolist()),
        torch.utils.data.Subset(dataset, idx[n_train + n_val:].tolist()),
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else './data'
    L    = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    F    = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    ds = PoseItDataset(root_dir=root, L=L, F=F)

    if len(ds) == 0:
        print("No samples loaded — check root_dir and L (grasping phase may be shorter than L).")
    else:
        sample = ds[0]
        print(f"\nL={L}, F={F}")
        print(f"tactile      : {sample['tactile'].shape}")        # (L, F, 3, H, W)
        print(f"rgb          : {sample['rgb'].shape}")            # (L, F, 3, H, W)
        print(f"ft           : {sample['ft'].shape}")             # (L, F*6)
        print(f"gripper      : {sample['gripper'].shape}")        # (L, F*2)
        print(f"gripper_force: {sample['gripper_force'].shape}")  # (1,)
        print(f"label        : {sample['label']}")
        print(f"pose_label   : {sample['pose_label']}")
        print(f"object       : {sample['object']}")

        loader = DataLoader(ds, batch_size=min(2, len(ds)), shuffle=False, num_workers=0)
        batch  = next(iter(loader))
        print(f"\nBatch shapes:")
        print(f"  tactile : {batch['tactile'].shape}")   # (B, L, F, 3, H, W)
        print(f"  rgb     : {batch['rgb'].shape}")
        print(f"  ft      : {batch['ft'].shape}")        # (B, L, F*6)
        print(f"  gripper : {batch['gripper'].shape}")
