# OGM V/T Full-Dataset Ablation

Run:
- W&B run: `mrsd-smores/TEMU/2kf4dxtk`
- Run name: `ogm-active-tact-rgb-run1`
- Training commit: `1be0755be0ca670fd7ba4af4a34083f1702d65d3`

Evaluation setup:
- Evaluated on the full loaded dataset, not the original random train/val/test split
- Dataset root: `/ocean/projects/cis260031p/shared/dataset/Gelsight`
- Loader config: `L=5`, `F1=1`, `F2=1`, `phase='grasp+pose'`
- Loaded samples: `1671`
- Skipped samples: `217`

Metrics:

| Setting | Active Modalities | Accuracy | Precision | Recall |
| --- | --- | ---: | ---: | ---: |
| all-modalities | V T | 0.8115 | 0.7102 | 0.8405 |
| V-only | V | 0.6595 | 0.5317 | 0.7767 |
| T-only | T | 0.7343 | 0.6352 | 0.6858 |

Remote artifacts:
- JSON: `/ocean/projects/cis260031p/afadia/ogm-vt-full-dataset-ablation/TactileEncoderForManipulation/analysis_outputs/ogm_vt_full_dataset_ablation.json`
- Markdown: `/ocean/projects/cis260031p/afadia/ogm-vt-full-dataset-ablation/TactileEncoderForManipulation/analysis_outputs/ogm_vt_full_dataset_ablation.md`

Notes:
- Positive class is failure (`slip` or `drop`), negative class is `pass`
- Sequence input is from `grasp+pose`
- Target label is the later `stability` outcome
