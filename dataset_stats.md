# Dataset Stats

Latest full-dataset analysis run:
- Date: March 14, 2026
- Machine: Bridges2
- Dataset root: `/ocean/projects/cis260031p/shared/dataset/Gelsight`
- Analysis script: `analyze_dataset_stats.py`
- Optimized analysis commit: `a00f24b` (`Optimize dataset analysis scanning`)

Saved remote artifacts:
- Full JSON: `/ocean/projects/cis260031p/afadia/afadia-full-res-dataset/analysis_outputs/poseit_dataset_stats_full.json`
- Full text summary: `/ocean/projects/cis260031p/afadia/afadia-full-res-dataset/analysis_outputs/poseit_dataset_stats_summary.txt`
- Stable-subset JSON: `/ocean/projects/cis260031p/afadia/afadia-full-res-dataset/analysis_outputs/poseit_dataset_stable_subset_stats.json`
- Stable-subset text summary: `/ocean/projects/cis260031p/afadia/afadia-full-res-dataset/analysis_outputs/poseit_dataset_stable_subset_stats.txt`

Note:
- The older `poseit_dataset_stats_full.txt` on Bridges2 is from an aborted slow run and should not be treated as the final report.
- For binary stable/fail splits below, `stable = pass` and `fail = any non-pass label`. This folds the one anomalous `stability=0.5` label into the fail bucket.

## Overall Dataset

- Samples: `1888`
- Skipped: `0`
- Analysis workers: `8`
- Total full-entry one-second intervals: `60213`

### Grasp+Pose Duration

`T = stability - grasping`

| Stat | Value |
| --- | ---: |
| count | 1888 |
| min | 5 |
| p05 | 7 |
| p25 | 8 |
| median | 9 |
| mean | 9.317 |
| p75 | 10 |
| p95 | 12 |
| max | 16 |
| std | 1.608 |
| sum | 17591 |

### Full Entry Duration

| Stat | Value |
| --- | ---: |
| count | 1888 |
| min | 22 |
| p05 | 27 |
| p25 | 30 |
| median | 32 |
| mean | 31.892 |
| p75 | 34 |
| p95 | 38 |
| max | 45 |
| std | 3.336 |
| sum | 60213 |

## Phase Label Distributions

- `grasp`: `pass 1086`, `slip 777`, `drop 25`
- `pose`: `pass 1326`, `slip 346`, `drop 191`, `notpresent 25`
- `stability`: `pass 1044`, `slip 430`, `drop 197`, `notpresent 216`, `0.5 1`
- `retract`: `pass 979`, `slip 436`, `drop 61`, `notpresent 412`

### Joint Label Distributions

- `grasp+pose`: `pass|pass 991`, `pass|slip 70`, `pass|drop 25`, `slip|pass 335`, `slip|slip 276`, `slip|drop 166`, `drop|notpresent 25`
- `stability+retract`: `pass|pass 945`, `pass|slip 92`, `pass|drop 7`, `slip|pass 34`, `slip|slip 343`, `slip|drop 53`, `drop|drop 1`, `drop|notpresent 196`, `notpresent|notpresent 216`, `0.5|slip 1`

## Counts Per Second

These statistics are computed over the full entry timeline for all samples, with zero-count seconds included.

| Modality | min | median | max | mean | std | zeros |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tactile | 0 | 30 | 42 | 28.737 | 4.344 | 176 |
| rgb | 0 | 5 | 13 | 4.836 | 0.796 | 542 |
| depth | 0 | 5 | 7 | 4.836 | 0.810 | 676 |
| side_cam | 0 | 30 | 36 | 29.014 | 4.535 | 262 |
| top_cam | 0 | 30 | 36 | 29.004 | 4.538 | 250 |
| ft | 0 | 100 | 289 | 96.732 | 14.533 | 250 |
| gripper | 0 | 10 | 17 | 8.821 | 2.688 | 1840 |
| robot | 0 | 2970 | 9429 | 2874.814 | 445.954 | 31 |

## Initially Stable Subset

Definition:
- Initially stable means `grasp=pass` and `pose=pass`
- Subset size: `991`

### Split By `stability` Label

Since `pose` is already fixed to `pass` inside this subset, this is the literal binary interpretation of "stable vs fail in pose+stability".

- Stayed stable at `stability`: `874`
- Became fail at `stability`: `117`

| Group | count | min | median | mean | max | std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stability stable | 874 | 5 | 9 | 9.156 | 15 | 1.646 |
| stability fail | 117 | 6 | 9 | 9.504 | 13 | 1.406 |

### Split By `stability + retract`

- Stayed stable through both later phases: `866`
- Failed in either later phase: `125`

| Group | count | min | median | mean | max | std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stability+retract stable | 866 | 5 | 9 | 9.158 | 15 | 1.644 |
| stability+retract fail | 125 | 5 | 9 | 9.464 | 13 | 1.440 |

## Initially Stable Interval Counts vs Global Median

This section counts one-second intervals across all `991` initially stable trajectories, using the global dataset median count-per-second for each modality as the threshold.

- Total one-second intervals in this subset: `31485`

| Modality | Global median | `< median` | `= median` | `> median` | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| depth | 5 | 2975 | 27237 | 1273 | 31485 |
| ft | 100 | 2429 | 28828 | 228 | 31485 |
| gripper | 10 | 6451 | 24949 | 85 | 31485 |
| rgb | 5 | 2952 | 27346 | 1187 | 31485 |
| robot | 2970 | 14490 | 2372 | 14623 | 31485 |
| side_cam | 30 | 3260 | 26664 | 1561 | 31485 |
| tactile | 30 | 9739 | 21507 | 239 | 31485 |
| top_cam | 30 | 3422 | 26645 | 1418 | 31485 |
