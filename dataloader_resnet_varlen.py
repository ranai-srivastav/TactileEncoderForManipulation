"""
Lazy variable-length PoseIt dataloader for the ResNet50 baseline models.

This mirrors the original PoseIt sampling logic used in training, but keeps the
indexed representation lightweight so full-dataset evaluation does not preload
all image tensors into memory.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from torchvision import transforms

LABEL_MAP = {"pass": 0, "slip": 1, "drop": 1}
IMAGE_SIZE = (224, 224)

F1 = 1
F2 = 1
FT_DIM = F2 * 6
GR_DIM = F2 * 2
L: int | None = None
phase = "grasp+pose"

IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def _parse_folder_name(name: str) -> dict:
    match = re.match(r"^(.+)_(\d+)_F(\d+)_pose(\d+)$", name)
    if not match:
        raise ValueError(f"Unexpected folder name: {name}")
    return {
        "object": match.group(1),
        "start_ts": int(match.group(2)),
        "force": float(match.group(3)),
        "pose_idx": int(match.group(4)),
    }


def _read_stages(path: Path) -> dict:
    stages = {}
    with open(path) as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            try:
                stages[row[0].strip()] = int(float(row[1].strip()))
            except ValueError:
                continue
    return stages


def _read_labels(path: Path) -> dict:
    labels = {}
    with open(path) as handle:
        for row in csv.reader(handle):
            if len(row) >= 2 and row[0].strip() in ("grasp", "pose", "stability"):
                labels[row[0].strip()] = row[1].strip().lower()
    return labels


def _read_csv_timeseries(path: Path, time_col: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rows = []
    with open(path) as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            try:
                rows.append([float(value) for value in row])
            except ValueError:
                continue
    if not rows:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32).reshape(0, 0)
    array = np.array(rows)
    timestamps = array[:, time_col].astype(np.int64)
    values = np.delete(array, time_col, axis=1).astype(np.float32)
    return timestamps, values


def _sample_bucket(items: np.ndarray, n_cols: int, count: int) -> Optional[np.ndarray]:
    n_items = len(items)
    if n_items == 0:
        return np.zeros(count * n_cols, dtype=np.float32)
    if n_items >= count:
        indices = np.round(np.linspace(0, n_items - 1, count)).astype(int)
        return items[indices].reshape(-1).astype(np.float32)
    return None


def _sample_image_bucket(paths: List[Path], count: int) -> Optional[List[Optional[Path]]]:
    n_items = len(paths)
    if n_items == 0:
        return [None] * count
    if n_items >= count:
        indices = np.round(np.linspace(0, n_items - 1, count)).astype(int)
        return [paths[index] for index in indices]
    return None


def _list_image_files(folder: Path) -> List[Tuple[int, int, Path]]:
    triples = []
    for path in folder.iterdir():
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            continue
        numbers = re.findall(r"\d+", path.stem)
        if len(numbers) < 2:
            continue
        unix_ts = int(numbers[-1])
        frame_idx = int(numbers[-2])
        triples.append((unix_ts, frame_idx, path))
    triples.sort(key=lambda item: (item[0], item[1]))
    return triples


def _load_image(path: Optional[Path]) -> torch.Tensor:
    if path is None or not path.exists():
        return torch.zeros(3, *IMAGE_SIZE)
    return IMAGENET_TRANSFORM(Image.open(path).convert("RGB"))


def _index_sample(sample_dir: Path) -> Optional[dict]:
    meta = _parse_folder_name(sample_dir.name)
    stages = _read_stages(sample_dir / "stages.csv")
    labels = _read_labels(sample_dir / "label.csv")

    shake_label = labels.get("stability", "drop")
    pose_label = labels.get("pose", "drop")
    grasp_label = labels.get("grasp", "")
    if shake_label not in LABEL_MAP or pose_label not in LABEL_MAP:
        return None
    if pose_label == "drop":
        return None

    t_grasp = stages.get("grasping", stages.get("grasp"))
    t_stability = stages.get("stability")
    if t_grasp is None or t_stability is None:
        return None

    seconds = list(range(t_grasp, t_stability))
    if L is not None:
        if len(seconds) < L:
            return None
        seconds = seconds[-L:]
    else:
        if len(seconds) < 3:
            return None
    if not seconds:
        return None

    ft_ts, ft_values = _read_csv_timeseries(sample_dir / "f_t.csv", time_col=0)
    gr_ts, gr_values = _read_csv_timeseries(sample_dir / "gripper.csv", time_col=0)

    tactile_files = _list_image_files(sample_dir / "gelsight")
    rgb_files = _list_image_files(sample_dir / "rgb")

    tactile_by_second = {}
    for ts, _, path in tactile_files:
        tactile_by_second.setdefault(ts, []).append(path)

    rgb_by_second = {}
    for ts, _, path in rgb_files:
        rgb_by_second.setdefault(ts, []).append(path)

    baseline_candidates = tactile_by_second.get(t_grasp, [])
    baseline_path = baseline_candidates[0] if baseline_candidates else None

    for second in seconds:
        if _sample_bucket(ft_values[ft_ts == second], n_cols=6, count=F2) is None:
            return None
        if _sample_bucket(gr_values[gr_ts == second], n_cols=2, count=F2) is None:
            return None
        if _sample_image_bucket(tactile_by_second.get(second, []), count=F1) is None:
            return None
        if _sample_image_bucket(rgb_by_second.get(second, []), count=F1) is None:
            return None

    return {
        "seconds": seconds,
        "ft_ts": ft_ts,
        "ft_values": ft_values,
        "gr_ts": gr_ts,
        "gr_values": gr_values,
        "tactile_by_second": tactile_by_second,
        "rgb_by_second": rgb_by_second,
        "baseline_path": baseline_path,
        "gripper_force": torch.tensor([meta["force"]], dtype=torch.float32),
        "label": torch.tensor(LABEL_MAP[shake_label], dtype=torch.long),
        "pose_label": torch.tensor(LABEL_MAP[pose_label], dtype=torch.long),
        "grasp_label": LABEL_MAP.get(grasp_label, -1),
        "object": meta["object"],
        "pose_idx": meta["pose_idx"],
        "force": meta["force"],
        "sample_dir": str(sample_dir),
    }


def _materialize_sample(indexed_sample: dict, *, load_images: bool) -> dict:
    baseline = _load_image(indexed_sample["baseline_path"])
    ft_seq = []
    gr_seq = []
    tactile_seq = []
    rgb_seq = []

    for second in indexed_sample["seconds"]:
        ft_bucket = indexed_sample["ft_values"][indexed_sample["ft_ts"] == second]
        gr_bucket = indexed_sample["gr_values"][indexed_sample["gr_ts"] == second]
        ft_seq.append(_sample_bucket(ft_bucket, n_cols=6, count=F2))
        gr_seq.append(_sample_bucket(gr_bucket, n_cols=2, count=F2))

        if load_images:
            tactile_paths = _sample_image_bucket(
                indexed_sample["tactile_by_second"].get(second, []),
                count=F1,
            )
            tactile_frames = torch.stack([_load_image(path) - baseline for path in tactile_paths])
            tactile_seq.append(tactile_frames)

            rgb_paths = _sample_image_bucket(
                indexed_sample["rgb_by_second"].get(second, []),
                count=F1,
            )
            rgb_frames = torch.stack([_load_image(path) for path in rgb_paths])
            rgb_seq.append(rgb_frames)

    sample = {
        "ft": torch.tensor(np.stack(ft_seq), dtype=torch.float32),
        "gripper": torch.tensor(np.stack(gr_seq), dtype=torch.float32),
        "gripper_force": indexed_sample["gripper_force"],
        "label": indexed_sample["label"],
        "pose_label": indexed_sample["pose_label"],
    }
    if load_images:
        sample["tactile"] = torch.stack(tactile_seq)
        sample["rgb"] = torch.stack(rgb_seq)
    return sample


class PoseItDatasetResNetVarLen(Dataset):
    def __init__(
        self,
        root_dir: Optional[str] = None,
        sample_dirs: Optional[List[str]] = None,
    ) -> None:
        assert root_dir or sample_dirs, "Provide root_dir or sample_dirs"
        directories = [Path(path) for path in sample_dirs] if sample_dirs else sorted(Path(root_dir).iterdir())

        self.samples = []
        skipped = 0
        for directory in tqdm(directories, desc="Indexing ResNet varlen dataset", unit="entry"):
            if not directory.is_dir():
                continue
            try:
                sample = _index_sample(directory)
            except Exception as exc:
                print(f"[WARN] Skipping {directory.name}: {exc}")
                skipped += 1
                continue
            if sample is None:
                skipped += 1
                continue
            self.samples.append(sample)

        print(
            f"Loaded {len(self.samples)} samples ({skipped} skipped)  "
            f"[L={L}, F1={F1}, F2={F2}, phase='{phase}', rgb_preprocess='imagenet']"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = _materialize_sample(self.samples[index], load_images=True)
        return (
            sample["tactile"],
            sample["rgb"],
            sample["ft"],
            sample["gripper"],
            sample["gripper_force"],
            sample["label"],
            sample["pose_label"],
        )

    def sensor_sample(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = _materialize_sample(self.samples[index], load_images=False)
        return sample["ft"], sample["gripper"], sample["gripper_force"]


def compute_sensor_stats(
    dataset: PoseItDatasetResNetVarLen,
    indices: Iterable[int],
    desc: str = "Sensor stats",
) -> dict[str, torch.Tensor]:
    index_list = list(indices)
    ft_rows = []
    gr_rows = []
    gf_rows = []
    for index in tqdm(index_list, desc=desc, unit="sample"):
        ft, gripper, gripper_force = dataset.sensor_sample(index)
        ft_rows.append(ft)
        gr_rows.append(gripper)
        gf_rows.append(gripper_force)

    ft = torch.cat(ft_rows, dim=0)
    gr = torch.cat(gr_rows, dim=0)
    gf = torch.cat(gf_rows, dim=0)

    def _stats(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = values.mean(dim=0)
        std = values.std(dim=0, unbiased=False).clamp_min(1e-6)
        return mean, std

    ft_mean, ft_std = _stats(ft)
    gr_mean, gr_std = _stats(gr)
    gf_mean, gf_std = _stats(gf)
    return {
        "ft_mean": ft_mean,
        "ft_std": ft_std,
        "gr_mean": gr_mean,
        "gr_std": gr_std,
        "gf_mean": gf_mean,
        "gf_std": gf_std,
    }


def collate_variable_length(batch):
    tactile_list, rgb_list, ft_list, gripper_list = [], [], [], []
    gf_list, label_list, pose_label_list, lengths = [], [], [], []

    for tactile, rgb, ft, gripper, gripper_force, label, pose_label in batch:
        lengths.append(tactile.shape[0])
        tactile_list.append(tactile)
        rgb_list.append(rgb)
        ft_list.append(ft)
        gripper_list.append(gripper)
        gf_list.append(gripper_force)
        label_list.append(label)
        pose_label_list.append(pose_label)

    max_steps = max(lengths)

    def pad_to_max(tensors):
        padded = []
        for tensor in tensors:
            if tensor.shape[0] < max_steps:
                pad_shape = (max_steps - tensor.shape[0], *tensor.shape[1:])
                pad = torch.zeros(pad_shape, dtype=tensor.dtype)
                tensor = torch.cat([tensor, pad], dim=0)
            padded.append(tensor)
        return torch.stack(padded)

    return (
        pad_to_max(tactile_list),
        pad_to_max(rgb_list),
        pad_to_max(ft_list),
        pad_to_max(gripper_list),
        torch.stack(gf_list),
        torch.stack(label_list),
        torch.stack(pose_label_list),
        torch.tensor(lengths, dtype=torch.long),
    )


if __name__ == "__main__":
    import sys
    from torch.utils.data import DataLoader

    root = sys.argv[1] if len(sys.argv) > 1 else "./data"
    dataset = PoseItDatasetResNetVarLen(root_dir=root)
    if len(dataset) == 0:
        print("No samples found.")
    else:
        tactile, rgb, ft, gripper, gripper_force, label, pose_label = dataset[0]
        print(f"tactile      : {tactile.shape}")
        print(f"rgb          : {rgb.shape}")
        print(f"ft           : {ft.shape}")
        print(f"gripper      : {gripper.shape}")
        print(f"gripper_force: {gripper_force.shape}")
        print(f"label        : {label}")
        print(f"pose_label   : {pose_label}")

        loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_variable_length)
        batch = next(iter(loader))
        print(f"batched tactile shape: {batch[0].shape}")
        print(f"lengths: {batch[-1].tolist()}")
