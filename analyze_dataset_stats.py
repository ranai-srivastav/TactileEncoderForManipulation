"""Analyze PoseIt dataset durations, labels, and per-second modality counts.

This script scans a PoseIt dataset root and reports:
- grasp+pose duration ``T`` for every sample, where ``T = stability - grasping``
- phase label distributions for grasp, pose, stability, and retract
- joint label distributions for grasp+pose and shake+retract
- the subset where grasp and pose both pass, and whether stability/retract
  later slip or drop
- per-modality counts-per-second statistics over each sample's full entry
  duration ``[entry_start_timestamp, entry_end_timestamp]``

This version is optimized for large datasets:
- no image pixels are loaded
- CSV time-series files only parse the timestamp column
- directory scans use ``os.scandir``
- samples are processed with a thread pool
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None


PHASE_LABEL_KEYS = ('grasp', 'pose', 'stability', 'retract')
FAIL_LABELS = {'slip', 'drop'}
FOLDER_RE = re.compile(r'^(.+)_(\d+)_F(\d+)_pose(\d+)$')
DIGIT_RE = re.compile(r'\d+')
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}

IMAGE_MODALITY_FOLDERS: Tuple[Tuple[str, str], ...] = (
    ('tactile', 'gelsight'),
    ('rgb', 'rgb'),
    ('depth', 'depth'),
    ('side_cam', 'side_cam'),
    ('top_cam', 'top_cam'),
)
TIMESERIES_MODALITY_FILES: Tuple[Tuple[str, str], ...] = (
    ('ft', 'f_t.csv'),
    ('gripper', 'gripper.csv'),
    ('robot', 'robot.csv'),
)
ALL_MODALITY_NAMES: Tuple[str, ...] = tuple(
    [name for name, _ in IMAGE_MODALITY_FOLDERS]
    + [name for name, _ in TIMESERIES_MODALITY_FILES]
)


class _NullProgressBar:
    def __init__(self, total: int, desc: str):
        self.total = total
        self.desc = desc

    def update(self, _: int = 1) -> None:
        return None

    def close(self) -> None:
        return None

    def write(self, message: str) -> None:
        print(message)

    def __enter__(self) -> '_NullProgressBar':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _make_progress(total: int, desc: str):
    if _tqdm is None:
        return _NullProgressBar(total=total, desc=desc)
    return _tqdm(total=total, desc=desc, unit='sample', dynamic_ncols=True)


def _parse_folder_name(name: str) -> Dict[str, Any]:
    match = FOLDER_RE.match(name)
    if not match:
        raise ValueError(f"Unexpected folder name: {name}")
    return {
        'object': match.group(1),
        'start_ts': int(match.group(2)),
        'force': float(match.group(3)),
        'pose_idx': int(match.group(4)),
    }


def _read_stages(path: Path) -> Dict[str, int]:
    stages: Dict[str, int] = {}
    with path.open(encoding='utf-8', errors='ignore') as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            try:
                stages[row[0].strip()] = int(float(row[1].strip()))
            except ValueError:
                continue
    return stages


def _read_label_rows(path: Path) -> List[List[str]]:
    rows: List[List[str]] = []
    with path.open(encoding='utf-8', errors='ignore') as handle:
        for row in csv.reader(handle):
            rows.append([cell.strip() for cell in row])
    return rows


def _phase_label_map(label_rows: Sequence[Sequence[str]]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for row in label_rows:
        if len(row) >= 2 and row[0] in PHASE_LABEL_KEYS:
            labels[row[0]] = row[1].lower()
    return labels


def _first_present(stages: Mapping[str, int], keys: Iterable[str]) -> Optional[int]:
    for key in keys:
        if key in stages:
            return stages[key]
    return None


def _duration_seconds(stages: Mapping[str, int],
                      start_keys: Sequence[str],
                      end_keys: Sequence[str]) -> Optional[int]:
    start = _first_present(stages, start_keys)
    end = _first_present(stages, end_keys)
    if start is None or end is None or end < start:
        return None
    return end - start


def _full_entry_duration_seconds(start_timestamp: Optional[int],
                                 end_timestamp: Optional[int]) -> Optional[int]:
    if start_timestamp is None or end_timestamp is None or end_timestamp < start_timestamp:
        return None
    return (end_timestamp - start_timestamp) + 1


def _scan_image_timestamp_counts(folder: Path) -> Tuple[Dict[int, int], Optional[int], Optional[int]]:
    counts: Dict[int, int] = {}
    min_ts: Optional[int] = None
    max_ts: Optional[int] = None

    if not folder.exists():
        return counts, None, None

    with os.scandir(folder) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            suffix = Path(entry.name).suffix.lower()
            if suffix not in IMAGE_EXTENSIONS:
                continue
            numbers = DIGIT_RE.findall(Path(entry.name).stem)
            if len(numbers) < 2:
                continue
            second = int(numbers[-1])
            counts[second] = counts.get(second, 0) + 1
            min_ts = second if min_ts is None else min(min_ts, second)
            max_ts = second if max_ts is None else max(max_ts, second)

    return counts, min_ts, max_ts


def _scan_csv_timestamp_counts(path: Path) -> Tuple[Dict[int, int], Optional[int], Optional[int]]:
    counts: Dict[int, int] = {}
    min_ts: Optional[int] = None
    max_ts: Optional[int] = None

    if not path.exists():
        return counts, None, None

    with path.open(encoding='utf-8', errors='ignore') as handle:
        next(handle, None)
        for line in handle:
            timestamp_text = line.partition(',')[0].strip()
            if not timestamp_text:
                continue
            try:
                second = int(float(timestamp_text))
            except ValueError:
                continue
            counts[second] = counts.get(second, 0) + 1
            min_ts = second if min_ts is None else min(min_ts, second)
            max_ts = second if max_ts is None else max(max_ts, second)

    return counts, min_ts, max_ts


def _counts_array(counts_by_second: Mapping[int, int],
                  start_timestamp: Optional[int],
                  end_timestamp: Optional[int]) -> np.ndarray:
    if start_timestamp is None or end_timestamp is None or end_timestamp < start_timestamp:
        return np.zeros(0, dtype=np.int64)

    counts = np.zeros(end_timestamp - start_timestamp + 1, dtype=np.int64)
    for second, value in counts_by_second.items():
        if start_timestamp <= second <= end_timestamp:
            counts[second - start_timestamp] = value
    return counts


def _distribution_stats(values: np.ndarray) -> Dict[str, Any]:
    if len(values) == 0:
        return {
            'count': 0,
            'min': None,
            'max': None,
            'mean': None,
            'std': None,
            'median': None,
            'p05': None,
            'p25': None,
            'p75': None,
            'p95': None,
            'sum': None,
            'zeros': 0,
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        'count': int(arr.size),
        'min': int(arr.min()),
        'max': int(arr.max()),
        'mean': float(arr.mean()),
        'std': float(arr.std(ddof=0)),
        'median': float(np.median(arr)),
        'p05': float(np.percentile(arr, 5)),
        'p25': float(np.percentile(arr, 25)),
        'p75': float(np.percentile(arr, 75)),
        'p95': float(np.percentile(arr, 95)),
        'sum': int(arr.sum()),
        'zeros': int((arr == 0).sum()),
    }


def _counter_dict(counter: Counter[str]) -> Dict[str, int]:
    return {key: int(value) for key, value in sorted(counter.items())}


def _fmt_stat(value: Any) -> str:
    if value is None:
        return 'None'
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _sample_table_row(sample: Mapping[str, Any]) -> str:
    labels = sample['labels']
    return (
        f"{sample['sample_name']:<45} "
        f"T={sample['grasp_pose_seconds']!s:<4} "
        f"entry={sample['entry_duration_seconds']!s:<4} "
        f"grasp={labels.get('grasp', 'missing'):<7} "
        f"pose={labels.get('pose', 'missing'):<7} "
        f"stability={labels.get('stability', 'missing'):<7} "
        f"retract={labels.get('retract', 'missing'):<7}"
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _iter_sample_dirs(dataset_root: Path) -> List[Path]:
    sample_dirs: List[Path] = []
    with os.scandir(dataset_root) as entries:
        for entry in entries:
            if entry.is_dir():
                sample_dirs.append(Path(entry.path))
    sample_dirs.sort(key=lambda path: path.name)
    return sample_dirs


def _analyze_sample_dir(sample_dir: Path) -> Dict[str, Any]:
    meta = _parse_folder_name(sample_dir.name)
    stages = _read_stages(sample_dir / 'stages.csv')
    label_rows = _read_label_rows(sample_dir / 'label.csv')
    labels = _phase_label_map(label_rows)

    min_candidates: List[int] = []
    max_candidates: List[int] = []
    if stages:
        stage_values = list(stages.values())
        min_candidates.append(min(stage_values))
        max_candidates.append(max(stage_values))

    per_modality_counts: Dict[str, Dict[int, int]] = {}

    for modality, folder_name in IMAGE_MODALITY_FOLDERS:
        counts, min_ts, max_ts = _scan_image_timestamp_counts(sample_dir / folder_name)
        per_modality_counts[modality] = counts
        if min_ts is not None:
            min_candidates.append(min_ts)
        if max_ts is not None:
            max_candidates.append(max_ts)

    for modality, relative_path in TIMESERIES_MODALITY_FILES:
        counts, min_ts, max_ts = _scan_csv_timestamp_counts(sample_dir / relative_path)
        per_modality_counts[modality] = counts
        if min_ts is not None:
            min_candidates.append(min_ts)
        if max_ts is not None:
            max_candidates.append(max_ts)

    entry_start_timestamp = min(min_candidates) if min_candidates else None
    entry_end_timestamp = max(max_candidates) if max_candidates else None

    grasp_pose_seconds = _duration_seconds(stages, ('grasping', 'grasp'), ('stability',))
    stability_to_retract_seconds = _duration_seconds(stages, ('stability',), ('retract',))
    retract_to_release_seconds = _duration_seconds(stages, ('retract',), ('release', 'fin'))
    entry_duration_seconds = _full_entry_duration_seconds(entry_start_timestamp, entry_end_timestamp)

    modality_counts_arrays: Dict[str, np.ndarray] = {}
    modality_count_stats: Dict[str, Dict[str, Any]] = {}
    for modality in ALL_MODALITY_NAMES:
        counts_array = _counts_array(
            per_modality_counts[modality],
            entry_start_timestamp,
            entry_end_timestamp,
        )
        modality_counts_arrays[modality] = counts_array
        modality_count_stats[modality] = _distribution_stats(counts_array)

    return {
        'sample_name': sample_dir.name,
        'sample_dir': str(sample_dir),
        'object': meta['object'],
        'pose_idx': int(meta['pose_idx']),
        'force': float(meta['force']),
        'labels': labels,
        'stage_timestamps': {key: int(value) for key, value in sorted(stages.items())},
        'grasp_pose_seconds': grasp_pose_seconds,
        'stability_to_retract_seconds': stability_to_retract_seconds,
        'retract_to_release_seconds': retract_to_release_seconds,
        'entry_duration_seconds': entry_duration_seconds,
        'modality_counts_per_second': modality_count_stats,
        'modality_counts_arrays': modality_counts_arrays,
    }


def analyze_dataset(dataset_root: Path, workers: int) -> Dict[str, Any]:
    sample_dirs = _iter_sample_dirs(dataset_root)

    grasp_pose_durations: List[int] = []
    full_entry_durations: List[int] = []
    stability_to_retract_durations: List[int] = []
    retract_to_release_durations: List[int] = []

    label_counters = {key: Counter() for key in PHASE_LABEL_KEYS}
    grasp_pose_joint = Counter()
    shake_retract_joint = Counter()
    clean_grasp_pose_subset = Counter()

    per_modality_counts: Dict[str, List[np.ndarray]] = defaultdict(list)
    per_sample: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []

    def consume(sample: Dict[str, Any]) -> None:
        labels = sample['labels']

        if sample['grasp_pose_seconds'] is not None:
            grasp_pose_durations.append(sample['grasp_pose_seconds'])
        if sample['stability_to_retract_seconds'] is not None:
            stability_to_retract_durations.append(sample['stability_to_retract_seconds'])
        if sample['retract_to_release_seconds'] is not None:
            retract_to_release_durations.append(sample['retract_to_release_seconds'])
        if sample['entry_duration_seconds'] is not None:
            full_entry_durations.append(sample['entry_duration_seconds'])

        for key in PHASE_LABEL_KEYS:
            label_counters[key][labels.get(key, 'missing')] += 1

        grasp_pose_joint[f"{labels.get('grasp', 'missing')}|{labels.get('pose', 'missing')}"] += 1
        shake_retract_joint[f"{labels.get('stability', 'missing')}|{labels.get('retract', 'missing')}"] += 1

        if labels.get('grasp') == 'pass' and labels.get('pose') == 'pass':
            stability_label = labels.get('stability')
            retract_label = labels.get('retract')
            if stability_label == 'pass' and retract_label == 'pass':
                clean_grasp_pose_subset['no_later_failure'] += 1
            elif stability_label in FAIL_LABELS or retract_label in FAIL_LABELS:
                clean_grasp_pose_subset['later_slip_or_drop'] += 1
            else:
                clean_grasp_pose_subset['later_missing_or_unknown'] += 1

        for modality, counts in sample.pop('modality_counts_arrays').items():
            per_modality_counts[modality].append(counts)

        per_sample.append(sample)

    total_samples = len(sample_dirs)
    with _make_progress(total=total_samples, desc='Analyzing samples') as progress:
        if workers <= 1:
            for sample_dir in sample_dirs:
                try:
                    sample = _analyze_sample_dir(sample_dir)
                except Exception as exc:
                    skipped.append({'sample_dir': str(sample_dir), 'error': str(exc)})
                else:
                    consume(sample)
                progress.update(1)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_dir = {
                    executor.submit(_analyze_sample_dir, sample_dir): sample_dir
                    for sample_dir in sample_dirs
                }
                for future in as_completed(future_to_dir):
                    sample_dir = future_to_dir[future]
                    try:
                        sample = future.result()
                    except Exception as exc:
                        skipped.append({'sample_dir': str(sample_dir), 'error': str(exc)})
                    else:
                        consume(sample)
                    progress.update(1)

    per_sample.sort(key=lambda sample: sample['sample_name'])
    skipped.sort(key=lambda item: item['sample_dir'])

    modality_stats: Dict[str, Dict[str, Any]] = {}
    for modality in ALL_MODALITY_NAMES:
        stacked = (
            np.concatenate(per_modality_counts[modality])
            if per_modality_counts[modality]
            else np.zeros(0, dtype=np.int64)
        )
        modality_stats[modality] = _distribution_stats(stacked)

    return {
        'dataset_root': str(dataset_root),
        'num_samples': len(per_sample),
        'num_skipped': len(skipped),
        'workers': workers,
        'skipped': skipped,
        'duration_stats': {
            'grasp_pose_seconds': _distribution_stats(np.asarray(grasp_pose_durations, dtype=np.int64)),
            'stability_to_retract_seconds': _distribution_stats(
                np.asarray(stability_to_retract_durations, dtype=np.int64)
            ),
            'retract_to_release_seconds': _distribution_stats(
                np.asarray(retract_to_release_durations, dtype=np.int64)
            ),
            'entry_duration_seconds': _distribution_stats(np.asarray(full_entry_durations, dtype=np.int64)),
        },
        'label_distributions': {
            key: _counter_dict(counter)
            for key, counter in label_counters.items()
        },
        'joint_label_distributions': {
            'grasp_pose': _counter_dict(grasp_pose_joint),
            'stability_retract': _counter_dict(shake_retract_joint),
        },
        'clean_grasp_pose_subset': {
            'count': int(sum(clean_grasp_pose_subset.values())),
            'later_slip_or_drop': int(clean_grasp_pose_subset['later_slip_or_drop']),
            'no_later_failure': int(clean_grasp_pose_subset['no_later_failure']),
            'later_missing_or_unknown': int(clean_grasp_pose_subset['later_missing_or_unknown']),
        },
        'modality_counts_per_second': modality_stats,
        'samples': per_sample,
    }


def _print_summary(report: Mapping[str, Any]) -> None:
    print(f"Dataset root: {report['dataset_root']}")
    print(f"Samples: {report['num_samples']}  Skipped: {report['num_skipped']}  Workers: {report['workers']}")

    print("\nGrasp+pose T statistics (T = stability - grasping):")
    t_stats = report['duration_stats']['grasp_pose_seconds']
    print(
        "  count={count} min={min} max={max} mean={mean} median={median} "
        "std={std} p05={p05} p95={p95}".format(
            count=t_stats['count'],
            min=_fmt_stat(t_stats['min']),
            max=_fmt_stat(t_stats['max']),
            mean=_fmt_stat(t_stats['mean']),
            median=_fmt_stat(t_stats['median']),
            std=_fmt_stat(t_stats['std']),
            p05=_fmt_stat(t_stats['p05']),
            p95=_fmt_stat(t_stats['p95']),
        )
    )

    subset = report['clean_grasp_pose_subset']
    print("\nSubset: grasp=pass and pose=pass")
    print(
        "  total={count} later_slip_or_drop={later_slip_or_drop} "
        "no_later_failure={no_later_failure} later_missing_or_unknown={later_missing_or_unknown}".format(
            **subset
        )
    )

    print("\nPer-sample T and phase labels:")
    for sample in report['samples']:
        print(f"  {_sample_table_row(sample)}")

    print("\nPhase label distributions:")
    for phase in PHASE_LABEL_KEYS:
        print(f"  {phase:<9} {report['label_distributions'][phase]}")

    print("\nJoint label distributions:")
    print(f"  grasp+pose        {report['joint_label_distributions']['grasp_pose']}")
    print(f"  stability+retract {report['joint_label_distributions']['stability_retract']}")

    print("\nCounts-per-second statistics by modality:")
    for modality, stats in report['modality_counts_per_second'].items():
        print(
            f"  {modality:<8} count={stats['count']} zeros={stats['zeros']} "
            f"min={_fmt_stat(stats['min'])} max={_fmt_stat(stats['max'])} "
            f"mean={_fmt_stat(stats['mean'])} median={_fmt_stat(stats['median'])} "
            f"std={_fmt_stat(stats['std'])} p05={_fmt_stat(stats['p05'])} "
            f"p95={_fmt_stat(stats['p95'])}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze PoseIt dataset durations, labels, and counts-per-second.",
    )
    parser.add_argument(
        'dataset_root',
        type=Path,
        help="Path to the PoseIt dataset root.",
    )
    parser.add_argument(
        '--output-json',
        type=Path,
        default=None,
        help="Optional path to write the full report as JSON.",
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of worker threads used to scan episode directories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_dataset(args.dataset_root, workers=args.workers)
    _print_summary(report)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=_json_default) + '\n',
            encoding='utf-8',
        )
        print(f"\nWrote JSON report to {args.output_json}")


if __name__ == '__main__':
    main()
