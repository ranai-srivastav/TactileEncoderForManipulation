# CLAUDE.md

Persistent context for Claude Code. **Update this at the end of each session.**

---

## Project

**TactileEncoderForManipulation** — CMU 11-777 course project.
Predicts slip/drop during robotic grasping from multimodal sensors (RGB, GelSight tactile, force-torque, gripper state).

**Active branch:** `main`
**HPC system:** Bridges-2 (PSC), allocation `cis260031p`

---

## Environment

```bash
module load anaconda3
conda activate /ocean/projects/cis260031p/shared/temu_conda
bash conda_jupyter.bash        # launch Jupyter
```

**Dataset:** `/ocean/projects/cis260031p/shared/dataset/Gelsight/`
493 episodes, 26 object types, force levels F5/F40/F80.
Folder format: `<object>_<timestamp>_F<force>_pose<idx>`

---

## Current Architecture (as of this session)

### dataloader.py

Module-level constants (single source of truth — train.py CLI defaults match these):
```python
F1     = 1        # image frames sampled per second
F2     = 1        # sensor readings sampled per second
FT_DIM = F2 * 6  # flattened F/T dim per timestep  (= 6 when F2=1)
GR_DIM = F2 * 2  # flattened gripper dim per timestep  (= 2 when F2=1)
L      = 20       # max seconds per episode
phase  = 'grasp+pose'
```
`train.py` sets `_dl.L`, `_dl.F1`, `_dl.F2` **before** constructing `PoseItDataset`, then
recomputes local `FT_DIM = _dl.F2 * 6` and `GR_DIM = _dl.F2 * 2` — avoids stale import values.

`_build_sample` output per sample (stored in `ds.samples`, NOT all returned by `__getitem__`):
- `tactile`: `(T, F1, 3, 224, 224)` — GelSight, baseline-subtracted
- `rgb`: `(T, F1, 3, 224, 224)` — RGB camera
- `ft`: `(T, FT_DIM)` — force-torque, flat per second
- `gripper`: `(T, GR_DIM)` — gripper state, flat per second
- `gripper_force`: `(1,)` — static scalar from folder name
- `label`: scalar long — 0=pass, 1=slip/drop (stability phase, **training target**)
- `pose_label`: scalar long — 0=pass, 1=slip/drop (pose phase)
- `grasp_label`: int — 0=pass, 1=fail, -1=unknown (grasp phase, **not in `__getitem__`**)
- `object`, `pose_idx`, `force`, `sample_dir` — metadata

`__getitem__` returns 7-tuple: `(tactile, rgb, ft, gripper, gripper_force, label, pose_label)`

`collate_variable_length` exists in the module but is **not used** in any DataLoader.
Since `L` is always set, all T are equal → default PyTorch collate works fine.
`lengths` is instead computed in `batch_to_device` from `tac.shape[1]`.

Key behaviors:
- Buckets with `0 < k < F` frames/readings: print `[WARN]` and skip the sample (`return None`)
- Buckets with `k == 0`: return zeros / black frames (silent)
- Episodes clipped to `L` seconds if `L is not None`
- GelSight frames are baseline-subtracted (frame − first frame at `t_grasp`)

### model.py — `GraspStabilityLSTM`

```
ResNet50 (frozen by default, 2048-d) × 2   ← tactile_encoder + rgb_encoder
Per second t:
  tac_emb = tactile_encoder(tac[:,t])  → (B, F1*2048)
  rgb_emb = rgb_encoder(rgb[:,t])      → (B, F1*2048)
  concat [tac_emb, rgb_emb, ft[:,t], gripper[:,t], gf] → (B, pre_lstm_dim=4105)  # F1=1: 1*2048*2+6+2+1
  Linear(4105→hidden_dim*2) → ReLU → LayerNorm(hidden_dim*2) → Dropout
    → Linear(hidden_dim*2→hidden_dim) → ReLU → Dropout → (B, hidden_dim)

sequence of L steps → N-layer GRU (bidirectional or unidirectional — controlled by args)
  → cat[forward at t=T-1, backward at t=0] → (B, hidden_dim*2)   [if bidirectional]
  → FC(hidden*2→64) → ReLU → Dropout → FC(64→n_outputs)
    n_outputs=1: (B,1) logit → BCEWithLogitsLoss; predict: logit > 0
    n_outputs=2: (B,2) logits → CrossEntropyLoss; predict: argmax
```

Constructor args:
```python
GraspStabilityLSTM(
    frames_per_sec=1,             # F1 — image frames per second
    ft_dim=6,                     # FT_DIM = _dl.F2 * 6  (recomputed after CLI override)
    gripper_dim=2,                # GR_DIM = _dl.F2 * 2  (recomputed after CLI override)
    hidden_dim=256,               # --hidden_dim
    lstm_layers=2,                # --lstm_layers
    bidirectional=True,           # True unless --unidirectional
    dropout=0.1,                  # --dropout
    freeze_resnet_rgb=True,       # 'resnet_rgb'     in --freeze
    freeze_resnet_tactile=True,   # 'resnet_tactile' in --freeze
    freeze_projection=False,      # 'projection'     in --freeze
    freeze_gru=False,             # 'gru'            in --freeze
    freeze_classifier=False,      # 'classifier'     in --freeze
    n_outputs=1,                  # --n_outputs (1=BCE, 2=CE)
    modalities=None,              # set of {'V','T','FT','G','GF'}; None = all active
)
```
Note: uses `nn.GRU` internally despite the class name `GraspStabilityLSTM`.
`model.n_outputs` stored for use in `evaluate()` to select prediction path.
`train()` override keeps frozen encoders in `eval()` mode so their BN stats do not drift.

Modality masking: disabled modalities zeroed before any encoder (in `forward`).
Keys: `V`=RGB, `T`=tactile, `FT`=force-torque, `G`=gripper, `GF`=gripper_force.

### train.py

Key CLI args:
```
--root_dir          path to dataset (default ./data)
--split             object | pose | random
--test_objects      used with --split object
--test_poses        used with --split pose
--sigma             DRS target S≠/S= ratio (default 0.5)
--drs_iter          iteration at which DRS activates (default 400; decoupled from LR anneal)
--drs_frac          float: if set, drs_iter = int(drs_frac * n_iters). Overrides --drs_iter.
                    Use in sweeps so DRS always fires regardless of n_iters value.
--anneal_iter       iteration at which StepLR steps down (default 300; only for --lr_scheduler step)
--anneal_frac       float: if set, anneal_iter = int(anneal_frac * n_iters). Overrides --anneal_iter.
                    Use in sweeps so LR drop always fires regardless of n_iters value.
--batch_size        default 32; auto-capped when ResNets are unfrozen (see below)
--lr                default 0.01; ResNet params trained at lr/10 (see optimizer param groups)
--weight_decay      default 0.01
--dropout           default 0.1
--hidden_dim        default 256
--lstm_layers       default 2
--unidirectional    flag: use unidirectional GRU (default: bidirectional)
--n_iters           total training iterations
--num_workers       default 4
--modalities        e.g. --modalities V T FT  (subset to activate)
--L                 max seconds per episode (default 20)
--F1                image frames per second (default 1; sets dataloader.F1)
--F2                sensor readings per second (default 1; sets dataloader.F2)
--subsample         fraction of dataset to load (e.g. 0.01 for quick tests)
--wandb_project     W&B project name (default "TEMU"; set to None to disable)
--wandb_run         W&B run name (optional)
--wandb_entity      W&B entity/team (default "mrsd-smores")
--overfit           flag: use 1 sample for train/val/test — sanity-check mode
--model_save_path   path for best checkpoint (default "trained_models/best_model.pt");
                    overridden to "trained_models/<wandb_run_id>/best_model.pt" when W&B active
                    (prevents checkpoint collisions between parallel sweep agents)
# --- sweep / optimizer ---
--optimizer         sgd (momentum=0.9) | adamw  (default: sgd)
--lr_scheduler      step | cosine_warm | none  (default: step)
--cosine_t0         T_0 for CosineAnnealingWarmRestarts (default 100)
--cosine_t_mult     T_mult (default 2)
# --- architecture ---
--n_outputs         1 (BCE) | 2 (CrossEntropy)  (default: 1)
--freeze            list of components to freeze (default: resnet_rgb resnet_tactile)
                    choices: resnet_rgb, resnet_tactile, projection, gru, classifier
                    pass --freeze with no args to train everything end-to-end
--clip_grad_norm    max gradient norm (default 1.0; 0 = disabled)
--tau               float (default 0.0): adds tau * ||W_majority||_2 to CE loss.
                    Only active with --n_outputs 2. Penalizes majority class (S=) weight norm.
```

**Auto batch-size cap (ResNets unfrozen):**
The forward pass packs ALL `B × T × F1` frames into a single ResNet call. When ResNets are
unfrozen, all intermediate activations must be stored for backprop (~175 MB/image with grads).
If any ResNet is unfrozen, `batch_size` is automatically capped:
```
max_bs = 28_000 // (L * F1 * 2 * 175)   # 28 GB budget, 2 encoders, 175 MB/image
```
For L=5: max ~16; L=9: max ~9; L=13: max ~6. A warning is printed if the cap fires.

**Optimizer param groups (differential LR):**
ResNet parameters (when unfrozen) are placed in a separate param group with `lr = args.lr / 10`.
All other trainable parameters (projection, GRU, classifier) use `lr = args.lr`.
This prevents large LR destroying pretrained ResNet features during fine-tuning.
The scheduler scales both groups by the same factor, preserving the 1/10 ratio throughout.

Execution flow:
1. Set `_dl.L`, `_dl.F1`, `_dl.F2`; recompute local `FT_DIM = _dl.F2*6`, `GR_DIM = _dl.F2*2`
   Apply `--anneal_frac`/`--drs_frac` overrides; auto-cap `batch_size` if ResNets unfrozen
2. Load dataset, optionally subsample (`max(4, int(N * subsample))` samples)
3. If `--overfit`: shrink to 1 sample, use it for train/val/test, disable DRS and LR anneal
   Else: split → `print_dataset_stats` (per-phase pass/fail/unknown for all/train/val/test)
4. Compute `pos_weight = n_neg / n_pos` from train split labels
5. Create `DRSSampler` (inactive until `drs_iter`)
6. Build model, criterion (BCE or CE), optimizer with param groups (ResNets lr/10), scheduler
7. W&B `config.update` — adds derived stats: `n_params_total`, `n_params_trainable`, `loss_fn`,
   `tau`, `lr_resnet`, `optimizer_type`, `scheduler_type`, `modalities_str`, `n_active_modalities`,
   freeze flags, `n_train/val/test`, `n_pos_train`, `n_neg_train`, `pos_weight_value`, `class_balance_train`
   Checkpoint path set to `trained_models/<wandb_run_id>/` (collision-safe for parallel agents)
8. Training loop: every 10 iters — evaluate, log metrics to console + W&B (`lr`, `lr_resnet`,
   `train/loss`, `train/grad_norm`, `val/f1`, `val/tpr`, `val/tnr`, `val/pos_pred_rate`, etc.), then:
   - Save `best_model.pt` when `val_f1 > best_val_f1` (no DRS gate)
   - Save rolling `model_latest.pt` in same dir (delete previous before writing)
   - `model.train()` called at top of while loop and after each evaluate block
9. Test evaluation: loads **both** `best_model.pt` and `model_latest.pt`, reports metrics for each;
   missing checkpoints print `[WARN]` and are skipped. After `best_model` eval, runs modality
   ablation (drop one modality at a time) and logs F1 drop per modality.

`evaluate()` returns 8-tuple: `(loss, acc, precision, recall, f1, tpr, tnr, pos_pred_rate)`
- `tpr` = sensitivity (recall on positives)
- `tnr` = specificity (recall on negatives)
- `ppr` = positive prediction rate (near-0 means model collapsed to predicting all-negative)

`batch_to_device(batch, device)`:
- Unpacks 7-tuple from default PyTorch collate
- Computes `lengths = [tac.shape[1]] * tac.shape[0]` (uniform since L is fixed)
- Returns 8-tuple including `lengths` for forward-compatibility with variable-length sequences

### sampler.py — `DRSSampler`

Partitions indices into `S=` (pose_label == label) and `S≠` groups.
Starts inactive (uniform sampling). `sampler.activate()` called at `drs_iter`.
`sigma` = target ratio `|S≠| / |S=|` in each batch. Must be `>= r` (natural dataset ratio).
Yields variable-size batches after activation.

Key constraint: `sigma >= r` (if `sigma == r`, DRS is a no-op; if `sigma < r`, raises `ValueError`).
Sampling uses `replace=True` automatically when `batch_size > len(train_indices)`.

### test.ipynb

Section 0 defines all experiment config:
```python
ROOT_DIR, SUBSAMPLE, BATCH_SIZE, L_MAX,
F1_CFG, F2_CFG,    # image/sensor fps — override _dl.F1/_dl.F2 in Section 1
HIDDEN_DIM,        # must match --hidden_dim in train.py (default 256)
SPLIT, TEST_OBJECTS, TEST_POSES, SIGMA, SAMPLE_IDX
```
Section 1 applies overrides before constructing any dataset (mirrors train.py):
```python
_dl.L = L_MAX; _dl.F1 = F1_CFG; _dl.F2 = F2_CFG
F1 = _dl.F1; F2 = _dl.F2; FT_DIM = _dl.F2 * 6; GR_DIM = _dl.F2 * 2
```

---

## Files

| File | Status | Notes |
|------|--------|-------|
| `dataloader.py` | ✅ Current | All bugs fixed; `uniform_random_split` guards empty splits; progress prints every 50 folders |
| `model.py` | ✅ Current | ResNet50, modality masking, flat concat, GRU; per-component freeze (5 components); n_outputs (1=BCE, 2=CE) |
| `train.py` | ✅ Current | Sweep-ready: optimizer (SGD/AdamW), scheduler (step/cosine_warm/none), n_outputs, `--freeze` list, differential LR (ResNets lr/10), auto batch-size cap, per-run-ID checkpoint paths, rich W&B logging, modality ablation at test |
| `sampler.py` | ✅ Current | DRS fixed: `replace` guard, `sigma < r` check |
| `test.ipynb` | ✅ Current | F1_CFG/F2_CFG/HIDDEN_DIM in Section 0; proper override in Section 1; frames_per_sec=F1 throughout |
| `README.md` | ⚠ Stale | See TODO.md #6 |
| `CLAUDE.md` | ✅ This file | |

---

## Work In Progress / Next Steps

- **Known bugs/improvements:** see `TODO.md`
- **Full training run** not yet validated end-to-end.

---

## Quick Reference Commands

```bash
# Smoke test (1% data, DRS on from iter 0)
python train.py --split random --subsample 0.01 --drs_iter 0 \
    --n_iters 20 --batch_size 2 --num_workers 0 \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
    --wandb_project None

# Single-sample overfit check
python train.py --overfit --n_iters 150 --batch_size 1 \
    --lr 0.001 --weight_decay 0.0 --num_workers 0 \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
    --wandb_project None

# Full training run (all modalities, W&B logging to TEMU project)
python train.py --split random --anneal_iter 300 --drs_iter 400 --n_iters 600 \
    --L 20 --modalities V T FT G GF \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight

# AdamW + cosine warm restart sweep candidate
python train.py --split random --optimizer adamw --lr 3e-4 \
    --lr_scheduler cosine_warm --cosine_t0 100 --cosine_t_mult 2 \
    --n_iters 600 --modalities V T FT G GF \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight

# Vision-only ablation (sweep modality interactions)
python train.py --split random --modalities V --n_iters 600 \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight

# Dataloader smoke test
python dataloader.py /ocean/projects/cis260031p/shared/dataset/Gelsight

# Verify DRS balance
python visualize_sampler.py --root /ocean/projects/cis260031p/shared/dataset/Gelsight \
    --n_samples 20 --sigma 1.0 --batch_size 200 --n_batches 100
```

---

## Key Decisions Made

- **ResNet50** chosen over ResNet18 for vision backbone
- **Flat concat** for FT/gripper (no small MLP encoders) — matches `parth_dev` style
- **Loss function** selectable: BCE (n_outputs=1) or CrossEntropy (n_outputs=2); both use pos_weight = n_neg/n_pos
- **Optimizer** selectable: SGD (momentum=0.9) or AdamW via `--optimizer`
- **Scheduler** selectable: StepLR (step at `anneal_iter`, gamma=0.1), CosineAnnealingWarmRestarts (stepped every iter), or none; **DRS decoupled** — activates at `drs_iter` (default 400)
- **GRU** used internally (class still named `GraspStabilityLSTM`); `lstm_layers` and `bidirectional` are constructor args
- **BiLSTM readout**: `cat[lstm_out[:,-1,:h], lstm_out[:,0,h:]]` — forward at T + backward at 0
- **L enforced in dataloader** (not in model.forward) — `_build_sample` drops sequences shorter than L; clips longer ones to `seconds[-L:]` (last L seconds, closest to stability event)
- **Modality masking** via zero-multiplication in `forward()` — disabled modalities still pass through encoders (zeroed input), shape is preserved
- **GRU operates over T seconds** (one step per second, F1 frames flattened per step) — not over T×F1 individual frames
- **`grasp_label`** stored in `ds.samples` but not returned by `__getitem__` — metadata only, used for dataset stats printing
- **Bucket underfill** (0 < k < F): prints `[WARN]` and skips sample — no forward-fill
- **`collate_variable_length` not used** — all DataLoaders use default PyTorch collate since `L` guarantees uniform T
- **Dual test eval** — both `best_model.pt` (best val_f1) and `model_latest.pt` (final weights) evaluated at end; missing checkpoint prints `[WARN]` and is skipped
- **Differential LR** — ResNet params (if unfrozen) trained at `lr/10` via separate param group; head params (projection, GRU, classifier) use full `lr`; scheduler scales both groups proportionally
- **Auto batch-size cap** — when ResNets unfrozen, `batch_size` capped to `28_000 // (L × F1 × 2 × 175)` to avoid OOM (effective ResNet batch = B×T×F1)
- **Per-run checkpoint paths** — when W&B active, saves to `trained_models/<wandb_run_id>/`; prevents collision between parallel sweep agents sharing the filesystem
- **anneal_frac / drs_frac** — sweep-friendly fraction-based alternatives; override absolute iter values in-place before `wandb.init`, so `vars(args)` captures the computed values

## Bugs Fixed

| Location | Bug | Fix |
|----------|-----|-----|
| `sampler.py:132` | `replace=False` crashes when `batch_size > len(train_indices)` | `replace = batch_size > len(all_indices)` |
| `sampler.py:88` | `sigma <= r` raised `ValueError` even when `sigma == r` | Changed to `sigma < r` |
| `dataloader.py:367` | `uniform_random_split` produced empty val set for small N | `n_val = max(1, ...)`, `n_train = max(1, min(..., n-n_val-1))` |
| `train.py` | `torch.load` missing `map_location` | Added `map_location=device` |
| `train.py:evaluate` | Division by zero when loader has no batches | Added `if n == 0: return 0,0,0,0,0` guard |
| `train.py` | `model.train()` not called after `evaluate()` | Called at top of while loop and after evaluate |
| `model.py` | `lstm_out[:, -1, :]` reads wrong end of backward stream | `cat[lstm_out[:,-1,:h], lstm_out[:,0,h:]]` |
| `model.py` | Constructor defaults didn't match dataloader constants | Fixed to `frames_per_sec=1, ft_dim=6, gripper_dim=2` |
| `model.py:projection` | `nn.BatchNorm1d` on `(B, T, F)` sequence | Replaced with `nn.LayerNorm` |
| `train.py` | `frames_per_sec=F2` should be `F1` | Fixed to `frames_per_sec=args.F1` |
| `train.py` | `FT_DIM`/`GR_DIM` stale if `--F2` non-default | Recomputed after `_dl.F2 = args.F2` |
| `train.py` | `best_model.pt` gated on `sampler.is_active` → never saved in overfit mode | Removed gate; save on any `val_f1` improvement |
| `train.py` | Test eval crashes if `best_model.pt` was never written | Dual test eval with `os.path.exists` guard per checkpoint |
| `test.ipynb` | `frames_per_sec=F2` in Sections 12 & 13 | Fixed to `frames_per_sec=F1` |
| `test.ipynb` | No F1/F2 config in Section 0; `_dl.F1`/`_dl.F2` never set | Added `F1_CFG`/`F2_CFG`/`HIDDEN_DIM` to Section 0; override + recompute in Section 1 |
| `test.ipynb` | `hidden_dim=512` hardcoded in Section 12 (train.py default is 256) | Replaced with `HIDDEN_DIM` config var |
| `train.py` | Parallel sweep agents all write to same `trained_models/best_model.pt` → size mismatch at test eval | Checkpoint dir now `trained_models/<wandb_run_id>/` when W&B active |
| `train.py` | CUDA OOM with large batch + unfrozen ResNet — B×T×F1 images processed in single ResNet call | Auto-cap: `max_bs = 28_000 // (L × F1 × 2 × 175)` when ResNets unfrozen |
