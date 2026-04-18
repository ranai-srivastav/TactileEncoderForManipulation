# TactileEncoderForManipulation

Group project for CMU 11-777 Multimodal Machine Learning.

Predicts **grasp stability** (slip / drop) during robotic manipulation from five multimodal sensors: RGB camera, GelSight tactile sensor, force-torque, gripper state, and gripper force level.

## Team
- Aayush Fadia
- Bhaswanth Ayapilla
- Megan Lee
- Parth Singh
- Ranai Srivastav

---

## Setup

```bash
module load anaconda3
conda activate /ocean/projects/cis260031p/shared/temu_conda
```

**Dataset location:** `/ocean/projects/cis260031p/shared/dataset/Gelsight/`
493 episodes, 26 object types, force levels F5 / F40 / F80.
Folder format: `<object>_<timestamp>_F<force>_pose<idx>`

### CLIP + T3 encoders (`--model clipt3`)

Uses CLIP ViT-L/14 (LAION-2B, 768-d) for RGB and T3 large (304M, 1024-d) for tactile:

```bash
pip install -r requirements_clipt3.txt
python scripts/download_pretrained.py --shared_dir /ocean/projects/cis260031p/shared
```

Then train with `--model clipt3`. Pretrained weights go to `shared/pretrained/` and `shared/.cache/huggingface/`.

**Evaluate a saved checkpoint** (same flags as training for data/model):

```bash
python scripts/eval_test.py -c trained_models/clipt3_5k_best.pt --model clipt3 \
  --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
  --modalities V T FT G GF --split random
```

Use `--which val` or `--which test` for a single split. For `--split random`, add `--seed N` only if training used the same `numpy` seed.

---

## File Overview

### `dataloader.py`

**Purpose:** Loads the GelSight dataset, parses episode folders, builds per-sample tensors, and provides train/val/test split utilities.

Each episode is sampled at a fixed rate. By default, `train.py` uses **variable-length** sequences (no `L` clipping) and `collate_variable_length` for padding. With `--fixed_length`, episodes are clipped to `L` seconds and the default collate is used. GelSight frames are baseline-subtracted (each frame minus the first frame at grasp time). Episodes with partially-filled temporal buckets are skipped with a `[WARN]` print.

**What you can change here:**
| Constant | Default | Effect |
|----------|---------|--------|
| `F1` | `1` | Image frames sampled per second — higher = richer visual signal, much more GPU memory |
| `F2` | `1` | Sensor readings sampled per second — affects `FT_DIM` and `GR_DIM` |
| `L` | `20` | Max seconds per episode |
| `phase` | `'grasp+pose'` | Which episode phases to include in a sample |

> These module-level defaults match the `train.py` CLI defaults. `train.py` sets `_dl.L`, `_dl.F1`, and `_dl.F2` before constructing the dataset, but since the defaults are identical, no override is needed unless you're changing them.

---

### `model.py` — `GraspStabilityLSTM`

**Purpose:** Defines the multimodal fusion architecture. Two frozen ResNet50 backbones encode tactile and RGB frames independently per second. Their embeddings are concatenated with force-torque, gripper state, and gripper force, then passed through a two-stage projection (with LayerNorm) to `hidden_dim`, processed by a 2-layer bidirectional LSTM, and classified by a small MLP.

```
tactile (B,T,F1,3,H,W) ──► ResNet50 ──► (B,T,F1×2048) ─┐
rgb     (B,T,F1,3,H,W) ──► ResNet50 ──► (B,T,F1×2048) ─┤
ft      (B,T,6)         ──────────────────────────────── ┤ concat → project → BiLSTM → FC → (B,1)
gripper (B,T,2)         ──────────────────────────────── ┤
gf      (B,1)           ──────────────────────────────── ┘
```

Projection is two-stage: `Linear(pre_lstm_dim → hidden_dim×2) → ReLU → LayerNorm → Dropout → Linear(hidden_dim×2 → hidden_dim) → ReLU → Dropout`. LayerNorm (not BatchNorm) is used so that normalization operates over the feature dimension for each `(batch, timestep)` position independently.

**What you can change here:**
| Parameter | Default | Effect |
|-----------|---------|--------|
| `hidden_dim` | `256` | Width of projection layer and LSTM hidden state |
| `lstm_layers` | `2` | Number of LSTM layers — deeper = more temporal capacity |
| `dropout` | `0.1` | Applied after projection and in classifier |
| `freeze_resnet` | `True` | Set to `False` to fine-tune ResNet50 (expensive, much more memory) |
| `modalities` | all 5 | Subset to ablate — disabled modalities are zeroed, shape is unchanged |

---

### `train.py`

**Purpose:** Main training script. Handles dataset loading, splitting, DRS sampler setup, the training loop (SGD + StepLR), validation every 10 iterations, checkpoint saving, and W&B logging.

See the [Training Reference](#training-reference) section below for all CLI arguments.

**What you can change here:**
- Optimizer: currently SGD with momentum=0.9. To use Adam, edit lines ~239–244.
- Scheduler: currently `StepLR(step_size=1, gamma=0.1)` — a single ×0.1 LR drop at `anneal_iter`. Edit line 245 to use cosine annealing, etc.
- Logging frequency: `iteration % 10` — change `10` to log more or less often.
- Checkpoints: every 10 iterations, `model_latest.pt` is saved (rolling — previous deleted) and both it and `best_model.pt` are uploaded to W&B. `best_model.pt` is only written when DRS is active and `val_f1` improves, so it is never a trivial all-negative solution.

---

### `sampler.py` — `DRSSampler`

**Purpose:** Implements Deferred Resampling (DRS) to counter class imbalance. Training samples are partitioned into S= (examples where `pose_label == label`, majority ~80%) and S≠ (minority ~20%). Each batch is constructed by thinning S= examples with probability `r/σ`, so the effective S≠/S= ratio in each batch approaches `σ`.

DRS is active from the first iteration. Pass `--sigma` to control the resampling ratio.

**What you can change here:**
| Parameter | Notes |
|-----------|-------|
| `sigma` | Target S≠/S= ratio per batch. `0.5` = moderate rebalancing, `1.0` = full balance. Must be `≥ r` (natural ratio). |
| `batch_size` | Pre-thinning batch size. Effective batch size after DRS will be smaller (~`sigma/(sigma+1)` × `batch_size`). |
| `seed` | Set for reproducible sampling. |

---

## Training Reference

### All `train.py` Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--root_dir` | `./data` | Path to the dataset root (folder containing episode subdirectories) |
| `--split` | `object` | How to split data: `object` (held-out objects), `pose` (held-out pose indices), `random` (random 70/15/15) |
| `--test_object_ids` | — | Zero-based indices into the sorted alphabetical object list (printed at startup). Use with `--split object`. |
| `--n_test_objects` | — | Randomly pick N objects for test. Use with `--split object`. Ignored if `--test_object_ids` is given. |
| `--test_pose_ids` | — | `pose_idx` integers from folder names to hold out for test. Use with `--split pose`. |
| `--n_test_poses` | — | Randomly pick N pose IDs for test. Use with `--split pose`. Ignored if `--test_pose_ids` is given. |
| `--sigma` | `1.0` | DRS target S≠/S= ratio. Use `0.5` for tactile-only runs, `1.0` for vision or full fusion. Resampling is active from the first iteration. |
| `--batch_size` | `200` | Pre-DRS batch size. Effective batch is smaller after DRS thinning |
| `--lr` | `0.01` | Initial learning rate for SGD |
| `--weight_decay` | `0.01` | L2 regularization strength. Set to `0.0` to remove regularization |
| `--dropout` | `0.1` | Dropout applied in projection layer and classifier head |
| `--hidden_dim` | `256` | LSTM hidden size and projection width |
| `--n_iters` | `600` | Total training iterations (not epochs) |
| `--anneal_iter` | `300` | Iteration at which LR is multiplied by 0.1 |
| `--num_workers` | `4` | DataLoader worker processes. Use `0` for debugging |
| `--modalities` | `V T FT G GF` | Active input modalities. Any subset of: `V` (RGB), `T` (tactile), `FT` (force-torque), `G` (gripper), `GF` (gripper force) |
| `--fixed_length` | off | If set, clip episodes to `L` seconds and use fixed batch shapes. Default: variable-length sequences |
| `--no_standardize_sensors` | off | If set, use raw FT/gripper values. Default: standardise sensors using train-split statistics |
| `--L` | `20` | Max seconds per episode when `--fixed_length` is set; ignored for the default variable-length mode |
| `--F1` | `1` | Image frames sampled per second. Overrides `dataloader.F1` before dataset construction |
| `--F2` | `1` | Sensor readings sampled per second. Overrides `dataloader.F2` before dataset construction |
| `--subsample` | `1.0` | Fraction of dataset to load (e.g. `0.01` = 1%). Useful for quick tests |
| `--wandb_project` | `TEMU` | W&B project name. Set to `None` to disable W&B logging |
| `--wandb_run` | `None` | W&B run name (auto-generated if omitted) |
| `--wandb_entity` | `mrsd-smores` | W&B team/entity |
| `--overfit` | off | Flag: use a single sample for train/val/test to sanity-check the model |
| `--model_save_path` | `trained_models/best_model.pt` | Path for the best-val-F1 checkpoint. A rolling `model_latest.pt` is also saved in the same directory every 10 iterations |

---

## Test Set Selection

When using `--split object` or `--split pose`, the script always prints the full set of available objects or pose IDs at startup so you can see what is in the loaded dataset before deciding what to hold out.

### Object split (`--split object`)

**By index** — the startup printout lists objects sorted alphabetically with a zero-based index. Pass those indices to hold out specific objects:
```bash
python MBT/mbt_train.py --split object --test_object_ids 0 5 ...
```

**By random count** — let the script randomly select N objects for you:
```bash
python MBT/mbt_train.py --split object --n_test_objects 3 ...
```

> Note: when `--subsample` is used, the index is computed from the subsampled dataset, so IDs may not match a full-dataset run. For production runs (no subsample) the index is stable.

### Pose split (`--split pose`)

**By explicit pose IDs** — pass the integer `pose_idx` values from the folder names:
```bash
python MBT/mbt_train.py --split pose --test_pose_ids 1 2 3 4 5 ...
```

**By random count** — let the script randomly select N pose IDs:
```bash
python MBT/mbt_train.py --split pose --n_test_poses 3 ...
```

Precedence when multiple args are given:
- Object split: `--test_object_ids` takes precedence over `--n_test_objects`.
- Pose split: `--test_pose_ids` takes precedence over `--n_test_poses`.
- `--split object` with neither arg → error; `--split pose` with neither arg → error.

---

## Example Commands

### 1. Smoke test (fast, 1% data, no W&B)
Verifies the pipeline runs end-to-end in under a minute. Uses random split, tiny subsample, DRS on from iteration 0.
```bash
python train.py \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
    --split random --subsample 0.01 \
    --anneal_iter 0 --n_iters 20 \
    --batch_size 2 --num_workers 0 \
    --wandb_project None
```

### 2. Sanity check — single-sample overfit
Trains on one sample for 500 iterations. Loss should drop steadily toward 0, confirming the model can memorize data. Use this when debugging architecture changes.
```bash
python train.py \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
    --overfit --n_iters 500 --batch_size 1 \
    --lr 0.001 --weight_decay 0.0 \
    --num_workers 0 --wandb_project None
```

### 3. Full training run — all modalities, object split
The standard experiment. Object indices are printed at startup; pass two of them with `--test_object_ids`. DRS activates at iteration 300.
```bash
python train.py \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
    --split object --test_object_ids 0 5 \
    --modalities V T FT G GF \
    --anneal_iter 300 --n_iters 600 \
    --sigma 1.0 --lr 0.01 --L 20 \
    --wandb_run full-all-modalities-obj-split
```

### 4. Ablation — tactile only
Disables all modalities except GelSight tactile. Use `sigma 0.5` as recommended for tactile-only runs.
```bash
python train.py \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
    --split random \
    --modalities T \
    --sigma 0.5 \
    --anneal_iter 300 --n_iters 600 \
    --wandb_run ablation-tactile-only
```

### 5. Ablation — vision only (RGB)
Disables all non-visual modalities to measure the contribution of the RGB camera alone.
```bash
python train.py \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
    --split random \
    --modalities V \
    --sigma 1.0 \
    --anneal_iter 300 --n_iters 600 \
    --wandb_run ablation-rgb-only
```

### 6. Ablation — no visual modalities (FT + gripper only)
Tests whether the model can predict stability from force-torque and gripper state alone, without any camera input.
```bash
python train.py \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
    --split random \
    --modalities FT G GF \
    --sigma 1.0 \
    --anneal_iter 300 --n_iters 600 \
    --wandb_run ablation-ft-gripper-only
```

### 7. Pose-split generalization test
Holds out specific pose indices for testing. Evaluates whether the model generalizes to unseen object orientations. Available pose IDs are printed at startup.
```bash
python train.py \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
    --split pose --test_pose_ids 1 2 3 4 5 \
    --modalities V T FT G GF \
    --anneal_iter 300 --n_iters 600 \
    --wandb_run pose-split-full-fusion
```

---

## Modality Ablations

The `--modalities` flag accepts any subset of `V T FT G GF`. Disabled modalities are zeroed out in the model forward pass — the architecture shape is unchanged, so you can mix and match freely without modifying code.

| Key | Sensor | Dimension |
|-----|--------|-----------|
| `V` | RGB camera | `F1 × 2048` per second |
| `T` | GelSight tactile | `F1 × 2048` per second |
| `FT` | Force-torque | `F2 × 6 = 12` per second |
| `G` | Gripper state | `F2 × 2 = 4` per second |
| `GF` | Gripper force command | `1` (static scalar) |

**Recommended sigma values:**
- Tactile-only (`T`): `--sigma 0.5`
- All other combinations: `--sigma 1.0`

**Suggested ablation matrix** (log each as a separate W&B run):

| Run name | `--modalities` |
|----------|---------------|
| `all` | `V T FT G GF` |
| `no-tactile` | `V FT G GF` |
| `no-rgb` | `T FT G GF` |
| `tactile-only` | `T` |
| `rgb-only` | `V` |
| `sensors-only` | `FT G GF` |
| `vision-only` | `V T` |
| `vision+ft` | `V T FT` |

---

## W&B Hyperparameter Sweep

Save the following as `sweep.yaml` and launch with `wandb sweep sweep.yaml`, then run agents with `wandb agent <sweep-id>`.

```yaml
program: train.py
method: bayes
metric:
  name: val/f1
  goal: maximize

parameters:
  root_dir:
    value: /ocean/projects/cis260031p/shared/dataset/Gelsight
  split:
    value: random
  n_iters:
    value: 600
  anneal_iter:
    value: 300
  num_workers:
    value: 4
  L:
    value: 20

  # Hyperparameters to sweep
  lr:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.1
  weight_decay:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.1
  dropout:
    values: [0.0, 0.1, 0.3, 0.5]
  hidden_dim:
    values: [128, 256, 512]
  sigma:
    values: [0.5, 1.0]
  batch_size:
    values: [100, 200]
```

Launch a sweep agent on Bridges-2:
```bash
wandb sweep sweep.yaml          # prints <sweep-id>
wandb agent mrsd-smores/TEMU/<sweep-id>
```

To run multiple agents in parallel across SLURM jobs, put `wandb agent ...` in your job script and submit multiple copies.

---

## Tips

- **Slow convergence?** Try increasing `--lr` to `0.05` or switching `--anneal_iter` to `200` for an earlier LR drop.
- **Overfitting on train?** Increase `--weight_decay` (try `0.05`) or `--dropout` (try `0.3`).
- **Class imbalance?** Decrease `--anneal_iter` to activate DRS earlier, or increase `--sigma` toward `1.0`.
- **OOM on GPU?** Reduce `--batch_size`, `--L`, or `--hidden_dim`. `F1` is already `1` by default; increasing it multiplies image memory by `F1`.
- **Debugging pipeline?** Use `--subsample 0.01 --num_workers 0 --n_iters 20` for fast iteration.
