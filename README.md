# TactileEncoderForManipulation

CMU 11-777 Multimodal Machine Learning project. Predicts grasp stability (slip/drop) from five sensors: RGB, GelSight tactile, force-torque, gripper state, gripper force.

**Team:** Aayush Fadia, Bhaswanth Ayapilla, Megan Lee, Parth Singh, Ranai Srivastav

---

## Setup

```bash
module load anaconda3
conda activate /ocean/projects/cis260031p/shared/temu_conda
```

Dataset: `/ocean/projects/cis260031p/shared/dataset/Gelsight/` — 493 episodes, 26 objects, force levels F5/F40/F80.

---

## Training (MBT)

The active model is a Multimodal Bottleneck Transformer (`MBT/mbt_model.py`). Submit to Bridges-2 via `MBT/train_job.sh`, or run directly:

```bash
# Sanity check — single sample, should overfit to ~0 loss
python MBT/mbt_train.py --overfit --n_iters 100 --lr 0.001 --num_workers 0 \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight --wandb_project None

# Smoke test — 1% data, no W&B
python MBT/mbt_train.py --split random --subsample 0.01 \
    --n_iters 50 --batch_size 2 --grad_accum 2 --num_workers 0 \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight --wandb_project None

# Full run (matches train_job.sh)
python MBT/mbt_train.py \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
    --split object --n_test_objects 5 \
    --adapter_dim 128 --n_iters 1500 --anneal_iter 1000 \
    --batch_size 4 --grad_accum 8 --lr 1e-4 --L 9

# Encoder finetuning (unfreeze both backbones, lower effective LR)
python MBT/mbt_train.py \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
    --split object --n_test_objects 5 \
    --no_freeze_vit --no_freeze_t3 \
    --lr 1e-4 --n_iters 1500 --anneal_iter 1000 --L 9
```

## Fine-tuning from a checkpoint

```bash
python MBT/mbt_finetune.py \
    --checkpoint trained_models/best_mbt_model.pt \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
    --split random --n_iters 600 --anneal_iter 400
```

Architecture config (modalities, L, F1/F2, etc.) is read from the checkpoint — only dataset and training hyperparameters are needed.

---

## CLI Reference — `mbt_train.py`

### Data & split
| Flag | Default | Notes |
|------|---------|-------|
| `--root_dir` | `./data` | Dataset root |
| `--split` | `object` | `object`, `pose`, or `random` |
| `--test_object_ids` | — | Object indices (printed at startup); use with `--split object` |
| `--n_test_objects` | — | Randomly pick N objects; use with `--split object` |
| `--test_pose_ids` | — | pose_idx integers from folder names; use with `--split pose` |
| `--n_test_poses` | — | Randomly pick N pose IDs; use with `--split pose` |
| `--L` | `20` | Max seconds per episode (job.sh uses 9) |
| `--F1` | `1` | Image frames per second |
| `--F2` | `1` | Sensor readings per second |
| `--subsample` | `1.0` | Fraction of dataset to load |

### Model
| Flag | Default | Notes |
|------|---------|-------|
| `--modalities` | `V T FT G GF` | Active modalities; any subset |
| `--num_bottlenecks` | `4` | MBT bottleneck tokens (paper default) |
| `--fusion_layer` | `8` | Layer where bottleneck fusion starts (paper default) |
| `--adapter_dim` | `64` | AdaptFormer hidden dim; 0 = no adapter |
| `--dropout` | `0.1` | |
| `--freeze_vit` | `True` | `--no_freeze_vit` to unfreeze ViT-Base RGB backbone |
| `--freeze_t3` | `True` | `--no_freeze_t3` to unfreeze T3 tactile backbone |
| `--pretrained_dir` | shared/pretrained | T3 weights location |

When encoders are unfrozen, their params train at `lr × 0.01` (vs `lr × 0.1` for adapters, `lr × 1` for everything else).

### Training
| Flag | Default | Notes |
|------|---------|-------|
| `--batch_size` | `4` | Micro-batch size |
| `--grad_accum` | `8` | Effective batch = batch_size × grad_accum |
| `--lr` | `1e-4` | Peak learning rate |
| `--weight_decay` | `0.01` | |
| `--n_iters` | `600` | Total iterations |
| `--anneal_iter` | `300` | Cosine decay starts here; set > n_iters to disable |
| `--sigma` | `0.5` | DRS S≠/S= ratio (0.5 = mild rebalancing, 1.0 = 50/50) |
| `--seed` | — | |

### Logging / checkpoints
| Flag | Default | Notes |
|------|---------|-------|
| `--wandb_project` | `TEMU` | Set `None` to disable |
| `--wandb_entity` | `mrsd-smores` | |
| `--wandb_run` | — | Auto-generated if omitted |
| `--model_save_path` | `trained_models/best_mbt_model.pt` | Best val-F1 checkpoint |
| `--overfit` | off | Single-sample sanity check |
| `--num_workers` | `4` | Use 0 for debugging |

---

## Modalities

| Key | Sensor | Shape |
|-----|--------|-------|
| `V` | RGB camera | `(B, T, F1, 3, 224, 224)` |
| `T` | GelSight tactile | `(B, T, F1, 3, 224, 224)` |
| `FT` | Force-torque | `(B, T, F2×6)` |
| `G` | Gripper state | `(B, T, F2×2)` |
| `GF` | Gripper force scalar | `(B, 1)` |

Disabled modalities are zeroed in the forward pass — architecture shapes are unchanged.
