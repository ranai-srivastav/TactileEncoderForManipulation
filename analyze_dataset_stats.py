"""Analyze PoseIt dataset durations, labels, and per-second modality counts.

This script scans a PoseIt dataset root and reports:
- grasp+pose duration ``T`` for every sample, where ``T = stability - grasping``
- phase label distributions for grasp, pose, stability, and retract
- joint label distributions for grasp+pose and shake+retract
- per-modality counts-per-second statistics over each sample's full entry
  duration ``[entry_start_timestamp, entry_end_timestamp]``

Counts-per-second include zero-count seconds when a modality has no data during
part of a sample's full entry timeline. This matches the second grid used by
``PoseItDataLoaderFull`` in ``BY_SECOND`` mode.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from dataloader_full import (
    ALL_MODALITIES,
    IMAGE_MODALITY_SPECS,
    TIMESERIES_MODALITY_SPECS,
    PoseItModality,
    _build_full_entry_index,
    _second_grid,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **_: Any):  # type: ignore[no-redef]
        return iterable


PHASE_LABEL_KEYS = ('grasp', 'pose', 'stability', 'retract')
FAIL_LABELS = {'slip', 'drop'}


def _phase_label_map(label_rows: Sequence[Sequence[str]]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for row in label_rows:
        if len(row) >= 2:
            key = row[0].strip()
            if key in PHASE_LABEL_KEYS:
                labels[key] = row[1].strip().lower()
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


def _counts_per_second_from_timestamps(second_timestamps: np.ndarray,
                                       seconds: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(seconds), dtype=np.int64)
    if len(seconds) == 0 or len(second_timestamps) == 0:
        return counts

    unique_seconds, unique_counts = np.unique(second_timestamps, return_counts=True)
    positions = np.searchsorted(seconds, unique_seconds)
    valid = (
        (positions >= 0)
        & (positions < len(seconds))
        & (seconds[positions] == unique_seconds)
    )
    counts[positions[valid]] = unique_counts[valid]
    return counts


def _counts_per_second_from_entries(entries: Sequence[tuple[int, int, Path]],
                                    seconds: np.ndarray) -> np.ndarray:
    if not entries:
        return np.zeros(len(seconds), dtype=np.int64)
    entry_seconds = np.fromiter((ts for ts, _, _ in entries), dtype=np.int64, count=len(entries))
    return _counts_per_second_from_timestamps(entry_seconds, seconds)


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


def analyze_dataset(dataset_root: Path) -> Dict[str, Any]:
    sample_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())

    grasp_pose_durations: List[int] = []
    full_entry_durations: List[int] = []
    stability_to_retract_durations: List[int] = []
    retract_to_release_durations: List[int] = []

    label_counters = {key: Counter() for key in PHASE_LABEL_KEYS}
    grasp_pose_joint = Counter()
    shake_retract_joint = Counter()
    clean_grasp_pose_subset = Counter()

    per_modality_counts: Dict[PoseItModality, List[np.ndarray]] = defaultdict(list)
    per_sample: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []

    for sample_dir in tqdm(
        sample_dirs,
        desc='Analyzing samples',
        unit='sample',
        dynamic_ncols=True,
    ):
        try:
            sample = _build_full_entry_index(sample_dir, modalities=ALL_MODALITIES)
        except Exception as exc:
            skipped.append({'sample_dir': str(sample_dir), 'error': str(exc)})
            continue

        labels = _phase_label_map(sample['label_rows'])
        stages = sample['stages']
        seconds = _second_grid(
            sample['entry_start_timestamp'],
            sample['entry_end_timestamp'],
        ).numpy()

        grasp_pose_seconds = _duration_seconds(stages, ('grasping', 'grasp'), ('stability',))
        stability_to_retract_seconds = _duration_seconds(stages, ('stability',), ('retract',))
        retract_to_release_seconds = _duration_seconds(stages, ('retract',), ('release', 'fin'))
        entry_duration_seconds = _full_entry_duration_seconds(
            sample['entry_start_timestamp'],
            sample['entry_end_timestamp'],
        )

        if grasp_pose_seconds is not None:
            grasp_pose_durations.append(grasp_pose_seconds)
        if stability_to_retract_seconds is not None:
            stability_to_retract_durations.append(stability_to_retract_seconds)
        if retract_to_release_seconds is not None:
            retract_to_release_durations.append(retract_to_release_seconds)
        if entry_duration_seconds is not None:
            full_entry_durations.append(entry_duration_seconds)

        for key in PHASE_LABEL_KEYS:
            label_counters[key][labels.get(key, 'missing')] += 1

        grasp_pose_joint[f"{labels.get('grasp', 'missing')}|{labels.get('pose', 'missing')}"] += 1
        shake_retract_joint[f"{labels.get('stability', 'missing')}|{labels.get('retract', 'missing')}"] += 1

        grasp_clean = labels.get('grasp') == 'pass'
        pose_clean = labels.get('pose') == 'pass'
        if grasp_clean and pose_clean:
            stability_label = labels.get('stability')
            retract_label = labels.get('retract')
            if stability_label == 'pass' and retract_label == 'pass':
                clean_grasp_pose_subset['no_later_failure'] += 1
            elif stability_label in FAIL_LABELS or retract_label in FAIL_LABELS:
                clean_grasp_pose_subset['later_slip_or_drop'] += 1
            else:
                clean_grasp_pose_subset['later_missing_or_unknown'] += 1

        sample_modality_stats: Dict[str, Dict[str, Any]] = {}
        for spec in IMAGE_MODALITY_SPECS:
            counts = _counts_per_second_from_entries(sample[spec.entries_key], seconds)
            per_modality_counts[spec.modality].append(counts)
            sample_modality_stats[spec.sample_key] = _distribution_stats(counts)
        for spec in TIMESERIES_MODALITY_SPECS:
            counts = _counts_per_second_from_timestamps(sample[spec.timestamps_key], seconds)
            per_modality_counts[spec.modality].append(counts)
            sample_modality_stats[spec.sample_key] = _distribution_stats(counts)

        per_sample.append({
            'sample_name': sample_dir.name,
            'sample_dir': str(sample_dir),
            'object': sample['object'],
            'pose_idx': int(sample['pose_idx']),
            'force': float(sample['force']),
            'labels': labels,
            'stage_timestamps': {key: int(value) for key, value in sorted(stages.items())},
            'grasp_pose_seconds': grasp_pose_seconds,
            'stability_to_retract_seconds': stability_to_retract_seconds,
            'retract_to_release_seconds': retract_to_release_seconds,
            'entry_duration_seconds': entry_duration_seconds,
            'modality_counts_per_second': sample_modality_stats,
        })

    modality_stats: Dict[str, Dict[str, Any]] = {}
    for modality in ALL_MODALITIES:
        stacked = (
            np.concatenate(per_modality_counts[modality])
            if per_modality_counts[modality]
            else np.zeros(0, dtype=np.int64)
        )
        modality_stats[modality.value] = _distribution_stats(stacked)

    return {
        'dataset_root': str(dataset_root),
        'num_samples': len(per_sample),
        'num_skipped': len(skipped),
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
    print(f"Samples: {report['num_samples']}  Skipped: {report['num_skipped']}")

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
    print(f"  grasp+pose       {report['joint_label_distributions']['grasp_pose']}")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_dataset(args.dataset_root)
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
