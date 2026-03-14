"""
Full-resolution PoseIt dataset loader.

Unlike dataloader.py, this loader does not downsample or clip to a phase
window. It returns the full duration of each dataset entry, preserving all
available observations and their timestamps.
"""

import argparse
import csv
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TypedDict, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


LABEL_MAP = {'pass': 0, 'slip': 1, 'drop': 1}
IMAGE_SIZE = (224, 224)

RGB_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
RAW_IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])


class PoseItModality(StrEnum):
    TACTILE = 'tactile'
    RGB = 'rgb'
    DEPTH = 'depth'
    SIDE_CAM = 'side_cam'
    TOP_CAM = 'top_cam'
    FT = 'ft'
    GRIPPER = 'gripper'
    ROBOT = 'robot'


class PoseItTimeLayout(StrEnum):
    FLAT = 'flat'
    BY_SECOND = 'by_second'


class PoseItPaddingMetadata(StrEnum):
    COUNTS = 'counts'
    MASK = 'mask'
    BOTH = 'both'


ALL_MODALITIES: Tuple[PoseItModality, ...] = tuple(PoseItModality)


class PoseItFullSample(TypedDict, total=False):
    time_layout: PoseItTimeLayout
    padding_metadata: PoseItPaddingMetadata
    seconds: torch.Tensor
    tactile: torch.Tensor
    tactile_timestamps: torch.Tensor
    tactile_second_timestamps: torch.Tensor
    tactile_frame_indices: torch.Tensor
    tactile_counts_per_second: torch.Tensor
    tactile_valid_mask: torch.Tensor
    rgb: torch.Tensor
    rgb_timestamps: torch.Tensor
    rgb_second_timestamps: torch.Tensor
    rgb_frame_indices: torch.Tensor
    rgb_counts_per_second: torch.Tensor
    rgb_valid_mask: torch.Tensor
    depth: torch.Tensor
    depth_timestamps: torch.Tensor
    depth_second_timestamps: torch.Tensor
    depth_frame_indices: torch.Tensor
    depth_counts_per_second: torch.Tensor
    depth_valid_mask: torch.Tensor
    side_cam: torch.Tensor
    side_cam_timestamps: torch.Tensor
    side_cam_second_timestamps: torch.Tensor
    side_cam_frame_indices: torch.Tensor
    side_cam_counts_per_second: torch.Tensor
    side_cam_valid_mask: torch.Tensor
    top_cam: torch.Tensor
    top_cam_timestamps: torch.Tensor
    top_cam_second_timestamps: torch.Tensor
    top_cam_frame_indices: torch.Tensor
    top_cam_counts_per_second: torch.Tensor
    top_cam_valid_mask: torch.Tensor
    ft: torch.Tensor
    ft_timestamps: torch.Tensor
    ft_second_timestamps: torch.Tensor
    ft_columns: List[str]
    ft_counts_per_second: torch.Tensor
    ft_valid_mask: torch.Tensor
    gripper: torch.Tensor
    gripper_timestamps: torch.Tensor
    gripper_second_timestamps: torch.Tensor
    gripper_columns: List[str]
    gripper_counts_per_second: torch.Tensor
    gripper_valid_mask: torch.Tensor
    robot: torch.Tensor
    robot_timestamps: torch.Tensor
    robot_second_timestamps: torch.Tensor
    robot_columns: List[str]
    robot_counts_per_second: torch.Tensor
    robot_valid_mask: torch.Tensor
    stage_names: List[str]
    stage_timestamps: torch.Tensor
    raw_label_rows: List[List[str]]
    label: torch.Tensor
    pose_label: torch.Tensor
    grasp_label: torch.Tensor
    gripper_force: torch.Tensor
    entry_start_timestamp: torch.Tensor
    entry_end_timestamp: torch.Tensor
    object: str
    pose_idx: int
    force: float
    sample_dir: str
    modalities: Tuple[PoseItModality, ...]


@dataclass(frozen=True)
class ImageModalitySpec:
    modality: PoseItModality
    sample_key: str
    entries_key: str
    folder_name: str
    transform: transforms.Compose
    mode: Optional[str]
    zero_channels: int


@dataclass(frozen=True)
class TimeseriesModalitySpec:
    modality: PoseItModality
    sample_key: str
    timestamps_key: str
    values_key: str
    columns_key: str
    relative_path: str


IMAGE_MODALITY_SPECS: Tuple[ImageModalitySpec, ...] = (
    ImageModalitySpec(PoseItModality.TACTILE, 'tactile', 'tactile_entries', 'gelsight', RGB_TRANSFORM, 'RGB', 3),
    ImageModalitySpec(PoseItModality.RGB, 'rgb', 'rgb_entries', 'rgb', RGB_TRANSFORM, 'RGB', 3),
    ImageModalitySpec(PoseItModality.DEPTH, 'depth', 'depth_entries', 'depth', RAW_IMAGE_TRANSFORM, None, 1),
    ImageModalitySpec(PoseItModality.SIDE_CAM, 'side_cam', 'side_cam_entries', 'side_cam', RGB_TRANSFORM, 'RGB', 3),
    ImageModalitySpec(PoseItModality.TOP_CAM, 'top_cam', 'top_cam_entries', 'top_cam', RGB_TRANSFORM, 'RGB', 3),
)

TIMESERIES_MODALITY_SPECS: Tuple[TimeseriesModalitySpec, ...] = (
    TimeseriesModalitySpec(PoseItModality.FT, 'ft', 'ft_ts', 'ft_val', 'ft_columns', 'f_t.csv'),
    TimeseriesModalitySpec(PoseItModality.GRIPPER, 'gripper', 'gripper_ts', 'gripper_val', 'gripper_columns', 'gripper.csv'),
    TimeseriesModalitySpec(PoseItModality.ROBOT, 'robot', 'robot_ts', 'robot_val', 'robot_columns', 'robot.csv'),
)


def _parse_folder_name(name: str) -> dict:
    m = re.match(r'^(.+)_(\d+)_F(\d+)_pose(\d+)$', name)
    if not m:
        raise ValueError(f"Unexpected folder name: {name}")
    return {
        'object': m.group(1),
        'start_ts': int(m.group(2)),
        'force': float(m.group(3)),
        'pose_idx': int(m.group(4)),
    }


def _read_stages(path: Path) -> Dict[str, int]:
    stages: Dict[str, int] = {}
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                try:
                    stages[row[0].strip()] = int(float(row[1].strip()))
                except ValueError:
                    pass
    return stages


def _read_label_rows(path: Path) -> List[List[str]]:
    rows = []
    with open(path) as f:
        for row in csv.reader(f):
            rows.append([cell.strip() for cell in row])
    return rows


def _extract_phase_labels(rows: List[List[str]]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for row in rows:
        if len(row) >= 2 and row[0] in ('grasp', 'pose', 'stability'):
            labels[row[0]] = row[1].lower()
    return labels


def _read_csv_timeseries(path: Path, time_col: int = 0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader, None) or []
        for row in reader:
            try:
                rows.append([float(v) for v in row])
            except ValueError:
                continue

    columns = [header[i].strip() for i in range(len(header)) if i != time_col]
    if not rows:
        return (
            np.array([], dtype=np.int64),
            np.zeros((0, len(columns)), dtype=np.float32),
            columns,
        )

    # Keep timestamps at float64 here so large Unix seconds survive exactly
    # before we cast the non-time columns down to float32.
    arr = np.array(rows, dtype=np.float64)
    ts = arr[:, time_col].astype(np.int64)
    values = np.delete(arr, time_col, axis=1).astype(np.float32)
    return ts, values, columns


def _list_image_files(folder: Path) -> List[Tuple[int, int, Path]]:
    triples = []
    if not folder.exists():
        return triples

    for path in folder.iterdir():
        if path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'):
            continue
        numbers = re.findall(r'\d+', path.stem)
        if len(numbers) < 2:
            continue
        frame_idx = int(numbers[-2])
        unix_ts = int(numbers[-1])
        triples.append((unix_ts, frame_idx, path))

    triples.sort(key=lambda item: (item[0], item[1]))
    return triples


def _load_image(path: Path,
                transform,
                mode: Optional[str],
                zero_channels: int) -> torch.Tensor:
    if path is None or not path.exists():
        return torch.zeros(zero_channels, *IMAGE_SIZE, dtype=torch.float32)

    with Image.open(path) as image:
        if mode is not None:
            image = image.convert(mode)
        return transform(image)


def _stack_image_entries(entries: List[Tuple[int, int, Path]],
                         transform,
                         mode: Optional[str],
                         zero_channels: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not entries:
        return (
            torch.zeros((0, zero_channels, *IMAGE_SIZE), dtype=torch.float32),
            torch.zeros(0, dtype=torch.float64),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
        )

    frames = torch.stack([
        _load_image(path, transform=transform, mode=mode, zero_channels=zero_channels)
        for _, _, path in entries
    ])
    second_timestamps = torch.tensor([ts for ts, _, _ in entries], dtype=torch.long)
    timestamps = _interpolate_second_timestamps(second_timestamps.numpy())
    frame_indices = torch.tensor([frame_idx for _, frame_idx, _ in entries], dtype=torch.long)
    return frames, timestamps, second_timestamps, frame_indices


def _interpolate_second_timestamps(second_timestamps: np.ndarray) -> torch.Tensor:
    """Expand integer-second timestamps to evenly spaced sub-second values.

    For n samples within the same second s, this emits:
      s + [0/n, 1/n, ..., (n-1)/n]
    Example: five samples in second 10 -> [10.0, 10.2, 10.4, 10.6, 10.8]
    """
    if len(second_timestamps) == 0:
        return torch.zeros(0, dtype=torch.float64)

    seconds = np.asarray(second_timestamps, dtype=np.int64)
    interpolated = np.empty(len(seconds), dtype=np.float64)

    start = 0
    while start < len(seconds):
        second = seconds[start]
        end = start + 1
        while end < len(seconds) and seconds[end] == second:
            end += 1

        count = end - start
        interpolated[start:end] = second + (np.arange(count, dtype=np.float64) / count)
        start = end

    return torch.from_numpy(interpolated)


def _timeseries_to_tensor(values: np.ndarray) -> torch.Tensor:
    if values.size == 0:
        width = values.shape[1] if values.ndim == 2 else 0
        return torch.zeros((0, width), dtype=torch.float32)
    return torch.tensor(values, dtype=torch.float32)


def _label_tensor(labels: Dict[str, str], key: str) -> torch.Tensor:
    return torch.tensor(LABEL_MAP.get(labels.get(key, ''), -1), dtype=torch.long)


def _timestamp_scalar(value: Optional[int]) -> torch.Tensor:
    if value is None:
        return torch.tensor(float('nan'), dtype=torch.float64)
    return torch.tensor(float(value), dtype=torch.float64)


def _normalize_modalities(
    modalities: Optional[Iterable[PoseItModality]],
) -> Tuple[PoseItModality, ...]:
    if modalities is None:
        return ALL_MODALITIES

    normalized = []
    seen = set()
    for modality in modalities:
        if modality not in seen:
            normalized.append(modality)
            seen.add(modality)
    return tuple(normalized)


def _normalize_time_layout(time_layout: PoseItTimeLayout) -> PoseItTimeLayout:
    return PoseItTimeLayout(time_layout)


def _normalize_padding_metadata(padding_metadata: PoseItPaddingMetadata) -> PoseItPaddingMetadata:
    return PoseItPaddingMetadata(padding_metadata)


def _include_counts(padding_metadata: PoseItPaddingMetadata) -> bool:
    return padding_metadata in (PoseItPaddingMetadata.COUNTS, PoseItPaddingMetadata.BOTH)


def _include_mask(padding_metadata: PoseItPaddingMetadata) -> bool:
    return padding_metadata in (PoseItPaddingMetadata.MASK, PoseItPaddingMetadata.BOTH)


def _empty_timeseries() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    return np.array([], dtype=np.int64), np.zeros((0, 0), dtype=np.float32), []


def _second_grid(start_timestamp: Optional[int], end_timestamp: Optional[int]) -> torch.Tensor:
    if start_timestamp is None or end_timestamp is None or end_timestamp < start_timestamp:
        return torch.zeros(0, dtype=torch.long)
    return torch.arange(start_timestamp, end_timestamp + 1, dtype=torch.long)


def _max_count_from_seconds(second_timestamps: np.ndarray) -> int:
    if len(second_timestamps) == 0:
        return 0
    _, counts = np.unique(second_timestamps, return_counts=True)
    return int(counts.max())


def _max_count_from_entries(entries: List[Tuple[int, int, Path]]) -> int:
    if not entries:
        return 0
    seconds = np.array([ts for ts, _, _ in entries], dtype=np.int64)
    return _max_count_from_seconds(seconds)


def _second_bucket_layout(second_timestamps: torch.Tensor,
                          seconds: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Return per-second slice bounds for sorted second timestamps.

    `second_timestamps` and `seconds` are assumed to be sorted in ascending
    order. The returned `starts` / `ends` arrays index into the original item
    tensors so all related fields can reuse the same layout work.
    """
    if len(seconds) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, torch.zeros(0, dtype=torch.long)

    ts_np = second_timestamps.detach().cpu().numpy()
    seconds_np = seconds.detach().cpu().numpy()
    starts = np.searchsorted(ts_np, seconds_np, side='left')
    ends = np.searchsorted(ts_np, seconds_np, side='right')
    counts = torch.from_numpy((ends - starts).astype(np.int64))
    return starts, ends, counts


def _pad_tensor_by_layout(values: torch.Tensor,
                          starts: np.ndarray,
                          ends: np.ndarray,
                          pad_value: float = 0.0,
                          max_count: Optional[int] = None) -> torch.Tensor:
    target_max_count = int((ends - starts).max()) if len(starts) > 0 else 0
    if max_count is not None:
        target_max_count = max(target_max_count, max_count)

    shape = (len(starts), target_max_count, *values.shape[1:])
    padded = torch.full(shape, pad_value, dtype=values.dtype)

    for i, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
        if start == end:
            continue
        padded[i, :end - start] = values[start:end]

    return padded


def _pad_scalar_by_layout(values: torch.Tensor,
                          starts: np.ndarray,
                          ends: np.ndarray,
                          pad_value,
                          max_count: Optional[int] = None) -> torch.Tensor:
    target_max_count = int((ends - starts).max()) if len(starts) > 0 else 0
    if max_count is not None:
        target_max_count = max(target_max_count, max_count)

    padded = torch.full((len(starts), target_max_count), pad_value, dtype=values.dtype)

    for i, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
        if start == end:
            continue
        padded[i, :end - start] = values[start:end]

    return padded


def _bucket_image_entries(entries: List[Tuple[int, int, Path]],
                          seconds: torch.Tensor,
                          transform,
                          mode: Optional[str],
                          zero_channels: int,
                          max_count: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    frames, timestamps, second_timestamps, frame_indices = _stack_image_entries(
        entries,
        transform=transform,
        mode=mode,
        zero_channels=zero_channels,
    )
    starts, ends, counts = _second_bucket_layout(second_timestamps, seconds)
    bucketed_frames = _pad_tensor_by_layout(frames, starts, ends, pad_value=0.0, max_count=max_count)
    bucketed_timestamps = _pad_scalar_by_layout(
        timestamps, starts, ends, pad_value=float('nan'), max_count=max_count
    )
    bucketed_second_timestamps = _pad_scalar_by_layout(
        second_timestamps, starts, ends, pad_value=-1, max_count=max_count
    )
    bucketed_frame_indices = _pad_scalar_by_layout(
        frame_indices, starts, ends, pad_value=-1, max_count=max_count
    )
    valid_mask = bucketed_second_timestamps >= 0
    return (
        bucketed_frames,
        bucketed_timestamps,
        bucketed_second_timestamps,
        bucketed_frame_indices,
        counts,
        valid_mask,
    )


def _bucket_timeseries(values: np.ndarray,
                       timestamps: np.ndarray,
                       seconds: torch.Tensor,
                       max_count: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    values_tensor = _timeseries_to_tensor(values)
    second_timestamps = torch.tensor(timestamps, dtype=torch.long)
    interp_timestamps = _interpolate_second_timestamps(timestamps)
    starts, ends, counts = _second_bucket_layout(second_timestamps, seconds)
    bucketed_values = _pad_tensor_by_layout(
        values_tensor, starts, ends, pad_value=0.0, max_count=max_count
    )
    bucketed_timestamps = _pad_scalar_by_layout(
        interp_timestamps, starts, ends, pad_value=float('nan'), max_count=max_count
    )
    bucketed_second_timestamps = _pad_scalar_by_layout(
        second_timestamps, starts, ends, pad_value=-1, max_count=max_count
    )
    valid_mask = bucketed_second_timestamps >= 0
    return bucketed_values, bucketed_timestamps, bucketed_second_timestamps, counts, valid_mask


def _maybe_add_padding_metadata(result: PoseItFullSample,
                                sample_key: str,
                                counts: torch.Tensor,
                                valid_mask: torch.Tensor,
                                padding_metadata: PoseItPaddingMetadata) -> None:
    if _include_counts(padding_metadata):
        result[f'{sample_key}_counts_per_second'] = counts
    if _include_mask(padding_metadata):
        result[f'{sample_key}_valid_mask'] = valid_mask


def _entry_time_bounds(stages: Dict[str, int],
                       sensor_timestamps: List[np.ndarray],
                       image_entries: List[List[Tuple[int, int, Path]]]) -> Tuple[Optional[int], Optional[int]]:
    mins: List[int] = []
    maxs: List[int] = []

    if stages:
        stage_values = list(stages.values())
        mins.append(min(stage_values))
        maxs.append(max(stage_values))

    for ts in sensor_timestamps:
        if len(ts) > 0:
            mins.append(int(ts.min()))
            maxs.append(int(ts.max()))

    for entries in image_entries:
        if entries:
            mins.append(entries[0][0])
            maxs.append(entries[-1][0])

    if not mins:
        return None, None
    return min(mins), max(maxs)


def _build_full_entry_index(sample_dir: Path,
                            modalities: Tuple[PoseItModality, ...]) -> Dict[str, object]:
    meta = _parse_folder_name(sample_dir.name)
    stages = _read_stages(sample_dir / 'stages.csv')
    label_rows = _read_label_rows(sample_dir / 'label.csv')
    labels = _extract_phase_labels(label_rows)

    selected = set(modalities)

    timeseries_data = {}
    for spec in TIMESERIES_MODALITY_SPECS:
        if spec.modality in selected:
            path = sample_dir / spec.relative_path
            if path.exists():
                ts, values, columns = _read_csv_timeseries(path, time_col=0)
            else:
                ts, values, columns = _empty_timeseries()
        else:
            ts, values, columns = _empty_timeseries()
        timeseries_data[spec.timestamps_key] = ts
        timeseries_data[spec.values_key] = values
        timeseries_data[spec.columns_key] = columns

    image_data = {}
    for spec in IMAGE_MODALITY_SPECS:
        image_data[spec.entries_key] = (
            _list_image_files(sample_dir / spec.folder_name)
            if spec.modality in selected else []
        )

    entry_start, entry_end = _entry_time_bounds(
        stages=stages,
        sensor_timestamps=[timeseries_data[spec.timestamps_key] for spec in TIMESERIES_MODALITY_SPECS],
        image_entries=[image_data[spec.entries_key] for spec in IMAGE_MODALITY_SPECS],
    )

    sample_index = {
        'sample_dir': sample_dir,
        'object': meta['object'],
        'pose_idx': meta['pose_idx'],
        'force': meta['force'],
        'stages': stages,
        'label_rows': label_rows,
        'labels': labels,
        'entry_start_timestamp': entry_start,
        'entry_end_timestamp': entry_end,
        'modalities': modalities,
    }
    sample_index.update(timeseries_data)
    sample_index.update(image_data)
    return sample_index


def _dataset_shape_stats(samples: List[Dict[str, object]],
                         modalities: Tuple[PoseItModality, ...]) -> Dict[PoseItModality, int]:
    max_items_per_second = {modality: 0 for modality in modalities}
    selected = set(modalities)

    for sample in samples:
        for spec in IMAGE_MODALITY_SPECS:
            if spec.modality in selected:
                max_items_per_second[spec.modality] = max(
                    max_items_per_second[spec.modality],
                    _max_count_from_entries(sample[spec.entries_key]),
                )
        for spec in TIMESERIES_MODALITY_SPECS:
            if spec.modality in selected:
                max_items_per_second[spec.modality] = max(
                    max_items_per_second[spec.modality],
                    _max_count_from_seconds(sample[spec.timestamps_key]),
                )

    return max_items_per_second


class PoseItDataLoaderFull(Dataset):
    """Lazy full-duration PoseIt dataset loader.

    Parameters:
        root_dir:
            Dataset root containing episode folders such as
            ``rubikCube_1612464613_F80_pose1``.
        sample_dirs:
            Explicit list of episode directories to index instead of scanning
            ``root_dir``.
        modalities:
            Typed subset of modalities to load. Use ``PoseItModality`` values so
            IDE autocomplete and type checking can catch mistakes.
        time_layout:
            ``PoseItTimeLayout.FLAT`` returns one tensor per modality with shape
            ``(N_items, ...)``.
            ``PoseItTimeLayout.BY_SECOND`` buckets each modality by integer
            second and returns shape ``(T, M_modality, ...)``, where ``T`` is the
            number of seconds in that sample and ``M_modality`` is the
            dataset-wide max number of items seen in any one second for that
            modality.
        padding_metadata:
            Controls whether ``BY_SECOND`` mode returns
            ``*_counts_per_second``, ``*_valid_mask``, or both.

    Notes:
        In ``BY_SECOND`` mode, only axes 1+ are dataset-consistent. The leading
        time axis ``T`` is allowed to vary across samples.
        Data tensors are padded with zeros, interpolated timestamps with
        ``nan``, and coarse integer timestamps / frame indices with ``-1``.
    """

    def __init__(self,
                 root_dir: Optional[str] = None,
                 sample_dirs: Optional[List[str]] = None,
                 modalities: Optional[Iterable[PoseItModality]] = None,
                 time_layout: PoseItTimeLayout = PoseItTimeLayout.FLAT,
                 padding_metadata: PoseItPaddingMetadata = PoseItPaddingMetadata.COUNTS):
        assert root_dir or sample_dirs, "Provide root_dir or sample_dirs"
        if root_dir is not None and not Path(root_dir).is_dir():
            raise FileNotFoundError(f"Dataset root does not exist or is not a directory: {root_dir}")
        dirs = [Path(d) for d in sample_dirs] if sample_dirs \
               else sorted(Path(root_dir).iterdir())

        self.modalities = _normalize_modalities(modalities)
        self._selected_modalities = set(self.modalities)
        self.time_layout = _normalize_time_layout(time_layout)
        self.padding_metadata = _normalize_padding_metadata(padding_metadata)
        self.samples = []
        skipped = 0
        for directory in dirs:
            if not directory.is_dir():
                continue
            try:
                self.samples.append(_build_full_entry_index(directory, modalities=self.modalities))
            except Exception as exc:
                print(f"[WARN] Skipping {directory.name}: {exc}")
                skipped += 1

        self.max_items_per_second = _dataset_shape_stats(
            self.samples,
            self.modalities,
        )
        print(f"Loaded {len(self.samples)} full-duration samples ({skipped} skipped)  "
              f"[modalities={[m.value for m in self.modalities]}, layout={self.time_layout.value}, "
              f"padding={self.padding_metadata.value}]")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> PoseItFullSample:
        sample = self.samples[idx]
        stage_items = sorted(sample['stages'].items(), key=lambda item: item[1])
        result: PoseItFullSample = {
            'time_layout': self.time_layout,
            'padding_metadata': self.padding_metadata,
            'stage_names': [name for name, _ in stage_items],
            'stage_timestamps': torch.tensor([float(ts) for _, ts in stage_items], dtype=torch.float64),
            'raw_label_rows': sample['label_rows'],
            'label': _label_tensor(sample['labels'], 'stability'),
            'pose_label': _label_tensor(sample['labels'], 'pose'),
            'grasp_label': _label_tensor(sample['labels'], 'grasp'),
            'gripper_force': torch.tensor([sample['force']], dtype=torch.float32),
            'entry_start_timestamp': _timestamp_scalar(sample['entry_start_timestamp']),
            'entry_end_timestamp': _timestamp_scalar(sample['entry_end_timestamp']),
            'object': sample['object'],
            'pose_idx': sample['pose_idx'],
            'force': sample['force'],
            'sample_dir': str(sample['sample_dir']),
            'modalities': self.modalities,
        }
        actual_seconds = _second_grid(sample['entry_start_timestamp'], sample['entry_end_timestamp'])
        if self.time_layout == PoseItTimeLayout.BY_SECOND:
            result['seconds'] = actual_seconds

        for spec in IMAGE_MODALITY_SPECS:
            if spec.modality not in self._selected_modalities:
                continue
            if self.time_layout == PoseItTimeLayout.BY_SECOND:
                values, timestamps, second_timestamps, frame_indices, counts, valid_mask = _bucket_image_entries(
                    sample[spec.entries_key],
                    seconds=actual_seconds,
                    transform=spec.transform,
                    mode=spec.mode,
                    zero_channels=spec.zero_channels,
                    max_count=self.max_items_per_second[spec.modality],
                )
                _maybe_add_padding_metadata(
                    result,
                    spec.sample_key,
                    counts,
                    valid_mask,
                    self.padding_metadata,
                )
            else:
                values, timestamps, second_timestamps, frame_indices = _stack_image_entries(
                    sample[spec.entries_key],
                    transform=spec.transform,
                    mode=spec.mode,
                    zero_channels=spec.zero_channels,
                )
            result[spec.sample_key] = values
            result[f'{spec.sample_key}_timestamps'] = timestamps
            result[f'{spec.sample_key}_second_timestamps'] = second_timestamps
            result[f'{spec.sample_key}_frame_indices'] = frame_indices

        for spec in TIMESERIES_MODALITY_SPECS:
            if spec.modality not in self._selected_modalities:
                continue
            if self.time_layout == PoseItTimeLayout.BY_SECOND:
                values, timestamps, second_timestamps, counts, valid_mask = _bucket_timeseries(
                    sample[spec.values_key],
                    sample[spec.timestamps_key],
                    seconds=actual_seconds,
                    max_count=self.max_items_per_second[spec.modality],
                )
                _maybe_add_padding_metadata(
                    result,
                    spec.sample_key,
                    counts,
                    valid_mask,
                    self.padding_metadata,
                )
            else:
                second_timestamps = torch.tensor(sample[spec.timestamps_key], dtype=torch.long)
                values = _timeseries_to_tensor(sample[spec.values_key])
                timestamps = _interpolate_second_timestamps(sample[spec.timestamps_key])
            result[spec.sample_key] = values
            result[f'{spec.sample_key}_timestamps'] = timestamps
            result[f'{spec.sample_key}_second_timestamps'] = second_timestamps
            result[spec.columns_key] = sample[spec.columns_key]

        return result


def collate_poseit_full(batch):
    """Keep variable-length full-duration samples intact."""
    return batch


def _shape_str(value) -> str:
    if hasattr(value, 'shape'):
        return str(tuple(value.shape))
    if isinstance(value, list):
        return f'list[{len(value)}]'
    return str(value)


def _print_sample_summary(sample_index: int,
                          sample: PoseItFullSample,
                          modalities: Tuple[PoseItModality, ...]) -> None:
    print(f"  Sample {sample_index}: {sample['object']} pose={sample['pose_idx']} force={sample['force']}")
    if 'seconds' in sample:
        print(f"    seconds: {_shape_str(sample['seconds'])}")
    for spec in IMAGE_MODALITY_SPECS:
        if spec.modality not in modalities or spec.sample_key not in sample:
            continue
        print(
            f"    {spec.sample_key:<8} data={_shape_str(sample[spec.sample_key])} "
            f"ts={_shape_str(sample[f'{spec.sample_key}_timestamps'])}"
        )
    for spec in TIMESERIES_MODALITY_SPECS:
        if spec.modality not in modalities or spec.sample_key not in sample:
            continue
        print(
            f"    {spec.sample_key:<8} data={_shape_str(sample[spec.sample_key])} "
            f"ts={_shape_str(sample[f'{spec.sample_key}_timestamps'])}"
        )


def _check_flat_sample(sample: PoseItFullSample,
                       modalities: Tuple[PoseItModality, ...]) -> None:
    for spec in IMAGE_MODALITY_SPECS:
        if spec.modality not in modalities or spec.sample_key not in sample:
            continue
        data = sample[spec.sample_key]
        timestamps = sample[f'{spec.sample_key}_timestamps']
        coarse = sample[f'{spec.sample_key}_second_timestamps']
        frame_indices = sample[f'{spec.sample_key}_frame_indices']
        assert data.shape[0] == timestamps.shape[0] == coarse.shape[0] == frame_indices.shape[0]

    for spec in TIMESERIES_MODALITY_SPECS:
        if spec.modality not in modalities or spec.sample_key not in sample:
            continue
        data = sample[spec.sample_key]
        timestamps = sample[f'{spec.sample_key}_timestamps']
        coarse = sample[f'{spec.sample_key}_second_timestamps']
        assert data.shape[0] == timestamps.shape[0] == coarse.shape[0]


def _check_by_second_sample(dataset: PoseItDataLoaderFull,
                            sample: PoseItFullSample,
                            modalities: Tuple[PoseItModality, ...]) -> None:
    assert 'seconds' in sample
    seconds = sample['seconds']
    assert seconds.ndim == 1
    t = seconds.shape[0]

    for spec in IMAGE_MODALITY_SPECS:
        if spec.modality not in modalities or spec.sample_key not in sample:
            continue
        data = sample[spec.sample_key]
        timestamps = sample[f'{spec.sample_key}_timestamps']
        coarse = sample[f'{spec.sample_key}_second_timestamps']
        frame_indices = sample[f'{spec.sample_key}_frame_indices']
        assert data.shape[0] == timestamps.shape[0] == coarse.shape[0] == frame_indices.shape[0] == t
        assert data.shape[1] == dataset.max_items_per_second[spec.modality]
        assert timestamps.shape == coarse.shape == frame_indices.shape == data.shape[:2]

        if _include_counts(dataset.padding_metadata):
            counts = sample[f'{spec.sample_key}_counts_per_second']
            assert counts.shape == (t,)
            assert int(counts.max().item()) <= data.shape[1]
        if _include_mask(dataset.padding_metadata):
            valid_mask = sample[f'{spec.sample_key}_valid_mask']
            assert valid_mask.shape == data.shape[:2]
            assert torch.equal(valid_mask, coarse >= 0)
        if dataset.padding_metadata == PoseItPaddingMetadata.BOTH:
            counts = sample[f'{spec.sample_key}_counts_per_second']
            valid_mask = sample[f'{spec.sample_key}_valid_mask']
            assert torch.equal(counts, valid_mask.sum(dim=1))

    for spec in TIMESERIES_MODALITY_SPECS:
        if spec.modality not in modalities or spec.sample_key not in sample:
            continue
        data = sample[spec.sample_key]
        timestamps = sample[f'{spec.sample_key}_timestamps']
        coarse = sample[f'{spec.sample_key}_second_timestamps']
        assert data.shape[0] == timestamps.shape[0] == coarse.shape[0] == t
        assert data.shape[1] == dataset.max_items_per_second[spec.modality]
        assert timestamps.shape == coarse.shape == data.shape[:2]

        if _include_counts(dataset.padding_metadata):
            counts = sample[f'{spec.sample_key}_counts_per_second']
            assert counts.shape == (t,)
            assert int(counts.max().item()) <= data.shape[1]
        if _include_mask(dataset.padding_metadata):
            valid_mask = sample[f'{spec.sample_key}_valid_mask']
            assert valid_mask.shape == data.shape[:2]
            assert torch.equal(valid_mask, coarse >= 0)
        if dataset.padding_metadata == PoseItPaddingMetadata.BOTH:
            counts = sample[f'{spec.sample_key}_counts_per_second']
            valid_mask = sample[f'{spec.sample_key}_valid_mask']
            assert torch.equal(counts, valid_mask.sum(dim=1))


def _run_case(name: str, dataset: PoseItDataLoaderFull) -> None:
    print()
    print(f"=== {name} ===")
    print(f"Shape conventions:")
    if dataset.time_layout == PoseItTimeLayout.FLAT:
        print("  images      : (N_items, C, H, W)")
        print("  time-series : (N_items, D)")
        print("  timestamps  : (N_items,)")
    else:
        print("  images      : (T, M_modality, C, H, W)")
        print("  time-series : (T, M_modality, D)")
        print("  timestamps  : (T, M_modality)")
        print("  counts      : (T,) when requested")
        print("  valid_mask  : (T, M_modality) when requested")
    print(f"Dataset-wide max items/sec: "
          f"{ {mod.value: dataset.max_items_per_second[mod] for mod in dataset.modalities} }")

    for i in range(len(dataset)):
        sample = dataset[i]
        if dataset.time_layout == PoseItTimeLayout.FLAT:
            _check_flat_sample(sample, dataset.modalities)
        else:
            _check_by_second_sample(dataset, sample, dataset.modalities)
        _print_sample_summary(i, sample, dataset.modalities)


def _parse_main_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run illustrative checks for PoseItDataLoaderFull.")
    parser.add_argument(
        'dataset_root',
        nargs='?',
        default='./data',
        help='Path to the dataset root directory.',
    )
    return parser.parse_args()


def _run_illustrative_checks(dataset_root: str) -> None:
    root = Path(dataset_root)
    if not root.is_dir():
        raise SystemExit(f"Dataset root does not exist or is not a directory: {root}")
    print(f"Dataset root: {root}")

    flat_all = PoseItDataLoaderFull(root_dir=str(root))
    if len(flat_all) == 0:
        print("No samples found — check your dataset_root path.")
        return

    by_second_modalities = (
        PoseItModality.RGB,
        PoseItModality.FT,
        PoseItModality.GRIPPER,
    )
    by_second_counts = PoseItDataLoaderFull(
        root_dir=str(root),
        modalities=by_second_modalities,
        time_layout=PoseItTimeLayout.BY_SECOND,
        padding_metadata=PoseItPaddingMetadata.COUNTS,
    )
    by_second_mask = PoseItDataLoaderFull(
        root_dir=str(root),
        modalities=by_second_modalities,
        time_layout=PoseItTimeLayout.BY_SECOND,
        padding_metadata=PoseItPaddingMetadata.MASK,
    )
    by_second_both = PoseItDataLoaderFull(
        root_dir=str(root),
        modalities=by_second_modalities,
        time_layout=PoseItTimeLayout.BY_SECOND,
        padding_metadata=PoseItPaddingMetadata.BOTH,
    )

    _run_case("FLAT / ALL_MODALITIES", flat_all)
    _run_case("BY_SECOND / COUNTS / RGB+FT+GRIPPER", by_second_counts)
    _run_case("BY_SECOND / MASK / RGB+FT+GRIPPER", by_second_mask)
    _run_case("BY_SECOND / BOTH / RGB+FT+GRIPPER", by_second_both)
    print("\nAll illustrative checks passed.")


if __name__ == '__main__':
    args = _parse_main_args()
    _run_illustrative_checks(args.dataset_root)
