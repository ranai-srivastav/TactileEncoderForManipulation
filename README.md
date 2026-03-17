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

---

## File Overview

### `dataloader.py`

Loads the GelSight dataset, parses episode folders, builds per-sample tensors, and provides train/val/test split utilities.

Each episode is sampled at a fixed rate and clipped to `L` seconds. GelSight frames are baseline-subtracted (each frame minus the first frame at grasp time). Episodes with partially-filled temporal buckets are skipped with a `[WARN]` print.

**What you can change here:**
| Constant | Default | Effect |
|----------|---------|--------|
| `F1` | `1` | Image frames sampled per second — higher = richer visual signal, much more GPU memory |
| `F2` | `1` | Sensor readings sampled per second — affects `FT_DIM` and `GR_DIM` |
| `L` | `20` | Max seconds per episode |
| `phase` | `'grasp+pose'` | Which episode phases to include in a sample |

> These module-level defaults match the `train.py` CLI defaults. `train.py` always sets `_dl.L`, `_dl.F1`, and `_dl.F2` before constructing the dataset.

---

### `model.py` — `GraspStabilityLSTM`

Multimodal fusion architecture. Two ResNet50 backbones (frozen by default) encode tactile and RGB frames independently per second. Their embeddings are concatenated with force-torque, gripper state, and gripper force, then passed through a two-stage projection MLP (with LayerNorm), processed by a bidirectional GRU, and classified by a small MLP head.

```
tactile (B,T,F1,3,H,W) ──► ResNet50 ──► (B,T,F1×2048) ─┐
rgb     (B,T,F1,3,H,W) ──► ResNet50 ──► (B,T,F1×2048) ─┤
ft      (B,T,FT_DIM)    ──────────────────────────────── ┤ concat → projection → GRU → classifier → (B, n_outputs)
gripper (B,T,GR_DIM)    ──────────────────────────────── ┤
gf      (B,1)           ──────────────────────────────── ┘
```

**Architecture stages (each independently freezable via `--freeze`):**

| Stage | `--freeze` key | Default |
|-------|---------------|---------|
| RGB ResNet50 | `resnet_rgb` | **frozen** |
| Tactile ResNet50 | `resnet_tactile` | **frozen** |
| Projection MLP (`pre_lstm_dim → hidden_dim×2 → hidden_dim`) | `projection` | trainable |
| Bidirectional GRU (N layers) | `gru` | trainable |
| Classifier MLP (`hidden_dim×2 → 64 → n_outputs`) | `classifier` | trainable |

---

### `sampler.py` — `DRSSampler`

Implements Deferred Resampling (DRS) to counter class imbalance. Training samples are partitioned into S= (examples where `pose_label == label`, majority) and S≠ (minority). Each batch thins S= examples so the effective S≠/S= ratio approaches `σ`.

DRS is **deferred**: it behaves as a standard random sampler until `activate()` is called at `--drs_iter`.

---

## Training Reference

### All `train.py` Arguments

#### Data & splits

| Argument | Default | Description |
|----------|---------|-------------|
| `--root_dir` | `./data` | Path to dataset root |
| `--split` | `object` | Split strategy: `object` (held-out objects), `pose` (held-out poses), `random` (70/15/15) |
| `--test_objects` | `mug bowl` | Objects held out for test when `--split object` |
| `--test_poses` | `1 2 3 4 5` | Pose indices held out when `--split pose` |
| `--L` | `20` | Max seconds per episode (clips longer; drops shorter) |
| `--F1` | `1` | Image frames per second (overrides `dataloader.F1`) |
| `--F2` | `1` | Sensor readings per second (overrides `dataloader.F2`) |
| `--subsample` | `1.0` | Fraction of dataset to load. `0.01` = 1% for quick tests |
| `--overfit` | off | Flag: use 1 sample for train/val/test to sanity-check the model |

#### Training loop

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_iters` | `600` | Total training iterations (not epochs) |
| `--batch_size` | `32` | Pre-DRS batch size. Auto-capped when ResNets unfrozen (see Freeze controls) |
| `--num_workers` | `4` | DataLoader workers. Use `0` for debugging |
| `--lr` | `0.01` | Initial learning rate |
| `--weight_decay` | `0.01` | L2 regularization |
| `--clip_grad_norm` | `1.0` | Max gradient norm for clipping. `0` = disabled |

#### Optimizer

| Argument | Default | Description |
|----------|---------|-------------|
| `--optimizer` | `sgd` | `sgd` (momentum=0.9) or `adamw` |

#### LR scheduler

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr_scheduler` | `step` | `step` — single ×0.1 drop at `anneal_iter`<br>`cosine_warm` — CosineAnnealingWarmRestarts, stepped every iter<br>`none` — constant LR |
| `--anneal_iter` | `300` | Iteration at which `step` scheduler steps. ⚠️ Must be < `--n_iters` or the drop never fires |
| `--anneal_frac` | `None` | If set, overrides `--anneal_iter` with `int(anneal_frac × n_iters)`. Use in sweeps to keep the drop proportional to training length |
| `--cosine_t0` | `100` | T_0 (iters per first restart cycle) for `cosine_warm` |
| `--cosine_t_mult` | `2` | T_mult (cycle length multiplier) for `cosine_warm` |

> `--anneal_iter` is ignored when `--lr_scheduler cosine_warm` or `none`. `--cosine_t0`/`--cosine_t_mult` are ignored when `--lr_scheduler step` or `none`.

#### Class imbalance (DRS)

| Argument | Default | Description |
|----------|---------|-------------|
| `--sigma` | `0.5` | DRS target S≠/S= ratio per batch. `0.5` = gentle, `1.0` = full balance. Must be ≥ natural ratio |
| `--drs_iter` | `400` | Iteration at which DRS activates. ⚠️ Must be < `--n_iters` or DRS never fires |
| `--drs_frac` | `None` | If set, overrides `--drs_iter` with `int(drs_frac × n_iters)`. Use in sweeps to keep DRS activation proportional to training length |

#### Model architecture

| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden_dim` | `256` | Projection and GRU hidden width |
| `--lstm_layers` | `2` | Number of GRU layers |
| `--unidirectional` | off | Flag: use unidirectional GRU (default: bidirectional) |
| `--dropout` | `0.1` | Dropout in projection and classifier |
| `--n_outputs` | `1` | `1` = single logit + BCEWithLogitsLoss; `2` = two logits + CrossEntropyLoss |
| `--tau` | `0.0` | Adds `tau × ‖W_majority‖₂` to CE loss (n_outputs=2 only). Penalizes majority class (S=) weight norm. Start small (e.g. 0.01); tau=0 disables |
| `--modalities` | `V T FT G GF` | Active input modalities. Disabled modalities are zeroed, shape unchanged |

#### Freeze controls

`--freeze` accepts a list of component names. Defaults to `resnet_rgb resnet_tactile` (both ResNets frozen).

```bash
# Default — both ResNets frozen, everything else trainable
python train.py --freeze resnet_rgb resnet_tactile

# Train everything end-to-end (expensive — ResNet fine-tuning)
python train.py --freeze

# Freeze ResNets + GRU, train only projection and classifier
python train.py --freeze resnet_rgb resnet_tactile gru

# Freeze ResNets + projection, train only GRU and classifier
python train.py --freeze resnet_rgb resnet_tactile projection
```

| `--freeze` value | Component frozen |
|-----------------|-----------------|
| `resnet_rgb` | RGB ResNet50 (default frozen) |
| `resnet_tactile` | Tactile ResNet50 (default frozen) |
| `projection` | Per-second projection MLP |
| `gru` | All GRU layers |
| `classifier` | Final classifier MLP |

> **Differential LR when ResNets unfrozen:** unfrozen ResNet params are placed in a separate optimizer param group with `lr = args.lr / 10`. This prevents the large LR destroying pretrained features. The scheduler scales both groups proportionally.

> **Auto batch-size cap:** when any ResNet is unfrozen, `batch_size` is automatically capped to `28_000 // (L × F1 × 2 × 175)` (≈ 28 GB budget at ~175 MB/image with gradients, 2 encoders). Example caps: L=5 → 16, L=9 → 9, L=13 → 6. A warning is printed if the cap fires.

#### Checkpointing

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_save_path` | `trained_models/best_model.pt` | Path for best-val-F1 checkpoint. When W&B is active, overridden to `trained_models/<wandb_run_id>/best_model.pt` — prevents collisions between parallel sweep agents |

#### W&B logging

| Argument | Default | Description |
|----------|---------|-------------|
| `--wandb_project` | `TEMU` | W&B project name. Set to `None` to disable |
| `--wandb_run` | `None` | W&B run name (auto-generated if omitted) |
| `--wandb_entity` | `mrsd-smores` | W&B team/entity |

**W&B logs per run (config):** all CLI args + `n_params_total`, `n_params_trainable`, `loss_fn`, `tau`, `lr_resnet`, `optimizer_type`, `scheduler_type`, `modalities_str`, `n_active_modalities`, per-component freeze flags, `n_train/val/test`, `n_pos/neg_train`, `pos_weight_value`, `class_balance_train`

**W&B logs per iteration:** `train/loss`, `train/grad_norm`, `train/batch_size_actual`, `val/loss`, `val/acc`, `val/precision`, `val/recall`, `val/f1`, `val/tpr`, `val/tnr`, `val/pos_pred_rate`, `lr`, `lr_resnet`, `drs_active`

**W&B logs at test:** per-checkpoint loss/acc/prec/rec/f1/tpr/tnr/ppr + per-modality ablation (`test_best/ablation_no_X`, `test_best/ablation_drop_X`)

---

## Example Commands

### Smoke test (fast, 1% data, no W&B)
```bash
python train.py \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
    --split random --subsample 0.01 \
    --n_iters 20 --batch_size 2 --num_workers 0 \
    --wandb_project None
```

### Sanity check — single-sample overfit
Loss should drop steadily toward 0, confirming the model can memorize data.
```bash
python train.py \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
    --overfit --n_iters 500 --batch_size 1 \
    --lr 0.001 --weight_decay 0.0 \
    --num_workers 0 --wandb_project None
```

### Full training run — all modalities, object split
```bash
python train.py \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
    --split object --test_objects mug bowl flashlight \
    --modalities V T FT G GF \
    --anneal_frac 0.5 --drs_frac 0.67 --n_iters 600 \
    --sigma 1.0 --lr 0.01 --L 20 \
    --wandb_run full-all-modalities-obj-split
```

### AdamW + cosine warm restarts
```bash
python train.py \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
    --split random --optimizer adamw --lr 3e-4 \
    --lr_scheduler cosine_warm --cosine_t0 100 --cosine_t_mult 2 \
    --n_iters 600 --modalities V T FT G GF \
    --wandb_run adamw-cosine-full
```

### Tactile-only ablation
```bash
python train.py \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
    --split random --modalities T --sigma 0.5 \
    --anneal_iter 300 --n_iters 600 \
    --wandb_run ablation-tactile-only
```

### Fine-tune only GRU + classifier (keep ResNets + projection frozen)
```bash
python train.py \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
    --split random --freeze resnet_rgb resnet_tactile projection \
    --n_iters 600 --wandb_run finetune-gru-only
```

---

## Modality Ablations

The `--modalities` flag accepts any subset of `V T FT G GF`. Disabled modalities are zeroed in the forward pass — shape is unchanged.

| Key | Sensor | Dimension (F1=F2=1) |
|-----|--------|---------------------|
| `V` | RGB camera | 2048 per second |
| `T` | GelSight tactile | 2048 per second |
| `FT` | Force-torque | 6 per second |
| `G` | Gripper state | 2 per second |
| `GF` | Gripper force command | 1 (static scalar) |

**Suggested ablation matrix:**

| Run name | `--modalities` |
|----------|---------------|
| `all` | `V T FT G GF` |
| `no-tactile` | `V FT G GF` |
| `no-rgb` | `T FT G GF` |
| `vision-only` | `V T` |
| `rgb-only` | `V` |
| `tactile-only` | `T` |
| `sensors-only` | `FT G GF` |
| `vision+ft` | `V T FT` |

> `--modalities` controls which inputs are active. At test time, `train.py` also runs an automatic per-modality ablation on `best_model.pt` and logs the F1 drop to W&B.

---

## W&B Hyperparameter Sweeps

Two focused sweep files are provided, each targeting a different research question. Both use all modalities and `split=object`.

```bash
# Initialize a sweep (run once)
wandb sweep sweep_data.yaml           # prints <sweep-id>
wandb agent mrsd-smores/TEMU/<sweep-id>

# Parallelism on Bridges-2: put wandb agent in a SLURM job script and sbatch multiple copies
```

> **⚠️ Sweep method choice:** `random` is preferred over `bayes` for mixed discrete+continuous params (optimizer, lr_scheduler, n_outputs). Bayesian optimization assumes continuous inputs, doesn't parallelize well across multiple SLURM agents, and underperforms random search on noisy objectives.

| File | Method | Runs | Question |
|------|--------|------|----------|
| [`sweep_data.yaml`](sweep_data.yaml) | grid | 15 | How does episode length and held-out object group affect generalization? |
| [`sweep_optimizer.yaml`](sweep_optimizer.yaml) | random | unlimited | What optimizer, LR schedule, and architecture config maximize F1? |

### sweep_data.yaml — data generalization (15 runs, grid)

Sweeps `L` (5–30 s) × `test_objects` (3 object groups). Everything else fixed. Grid is appropriate here because the space is small and you want exhaustive coverage.

### sweep_optimizer.yaml — training config (random, ~50–100 runs)

Sweeps optimizer, LR, scheduler, DRS timing, hidden_dim, dropout, lstm_layers, n_outputs, freeze strategy.

**Uses `--anneal_frac` and `--drs_frac`** (fractions of `n_iters`) to ensure LR drop and DRS activation always fire — no invalid combinations regardless of `n_iters`.

> `--anneal_iter` / `--drs_iter` set absolute iteration numbers — they silently do nothing if > `n_iters`.
> `--anneal_frac` / `--drs_frac` set them as a fraction of `n_iters` and are preferred in sweeps.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `val/pos_pred_rate ≈ 0` | Model collapsed to all-negative | Lower `--sigma` or increase `--drs_iter` activation; check `pos_weight` |
| `val/tpr ≈ 0, val/tnr ≈ 1` | Same collapse — caught early by TPR/TNR split | Same as above |
| `val/f1` plateaued early | LR too high or annealed too late | Try `--anneal_iter 150` or `--optimizer adamw --lr 3e-4` |
| OOM on GPU (frozen ResNets) | Batch size or hidden_dim too large | Reduce `--batch_size`, `--hidden_dim`, or `--L` |
| OOM on GPU (unfrozen ResNets) | B×T×F1 images processed in one ResNet call; all intermediates stored for backprop | Auto-cap fires automatically; manually use smaller `--batch_size` or `--L` |
| Slow convergence | LR too low | Increase `--lr` to `0.05` or switch to `cosine_warm` |
| Overfitting (train↑ val↓) | Too little regularization | Increase `--weight_decay` (try `0.05`) or `--dropout` (try `0.3`) |
