# CLAUDE.md

Persistent context for Claude Code. **Update this at the end of each session.**

---

## Project

**TactileEncoderForManipulation** — CMU 11-777 course project.
Predicts slip/drop during robotic grasping from multimodal sensors (RGB, GelSight tactile, force-torque, gripper state).

**Active branch:** `ranais/new_dataloader`
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
FT_DIM = 6        # F2 * 6
GR_DIM = 2        # F2 * 2
L      = 20       # max seconds per episode
phase  = 'grasp+pose'
```
`train.py` sets `_dl.L`, `_dl.F1`, `_dl.F2` before constructing the dataset; defaults are identical so no override is needed unless explicitly changing them.

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
ResNet50 (frozen, 2048-d) × 2   ← tactile_encoder + rgb_encoder
Per second t:
  tac_emb = tactile_encoder(tac[:,t])  → (B, F1*2048)
  rgb_emb = rgb_encoder(rgb[:,t])      → (B, F1*2048)
  concat [tac_emb, rgb_emb, ft[:,t], gripper[:,t], gf] → (B, pre_lstm_dim=4105)  # F1=1: 1*2048*2+6+2+1
  Linear(4105→hidden_dim*2) → ReLU → LayerNorm(hidden_dim*2) → Dropout
    → Linear(hidden_dim*2→hidden_dim) → ReLU → Dropout → (B, hidden_dim)

sequence of L steps → 2-layer BiLSTM
  → cat[forward at t=T-1, backward at t=0] → (B, hidden_dim*2)
  → FC(hidden*2→64) → ReLU → Dropout → FC(64→1) → (B, 1) logit
  (4 FC layers total: 2 in projection, 2 in classifier head)

Loss: BCEWithLogitsLoss
Predict: logit > 0 → slip/drop
```

Constructor args:
```python
GraspStabilityLSTM(
    frames_per_sec=F2,    # frames per second (= F1 = F2 = 1 by default)
    ft_dim=FT_DIM,        # 6  (F2*6, with F2=1)
    gripper_dim=GR_DIM,   # 2  (F2*2, with F2=1)
    hidden_dim=256,
    lstm_layers=2,
    dropout=0.1,
    freeze_resnet=True,
    modalities=None,      # set of {'V','T','FT','G','GF'}; None = all active
)
```

Modality masking: disabled modalities zeroed before any encoder (in `forward`).
Keys: `V`=RGB, `T`=tactile, `FT`=force-torque, `G`=gripper, `GF`=gripper_force.

### train.py

Key CLI args:
```
--root_dir       path to dataset (default ./data)
--split          object | pose | random
--test_objects   used with --split object
--test_poses     used with --split pose
--sigma          DRS target S≠/S= ratio (0.5 or 1.0)
--batch_size     default 32
--lr             default 0.01 (SGD + momentum=0.9)
--weight_decay   default 0.01
--dropout        default 0.1
--hidden_dim     default 256
--n_iters        total training iterations
--anneal_iter    iteration at which DRS activates and LR steps ×0.1
--num_workers    default 4
--modalities     e.g. --modalities V T FT  (subset to activate)
--L              max seconds per episode (default 20)
--F1             image frames per second (default 1; sets dataloader.F1 — matches module default)
--F2             sensor readings per second (default 1; sets dataloader.F2 — matches module default)
--subsample      fraction of dataset to load (e.g. 0.01 for quick tests)
--wandb_project   W&B project name (default "TEMU"; set to None to disable)
--wandb_run       W&B run name (optional)
--wandb_entity    W&B entity/team (default "mrsd-smores")
--overfit         flag: use 1 sample for train/val/test — sanity-check mode, DRS disabled
--model_save_path path for best checkpoint (default "trained_models/best_model.pt");
                  `model_latest.pt` is saved in the same directory
```

Execution flow:
1. Set `dataloader.L`, `dataloader.F1`, `dataloader.F2` before constructing `PoseItDataset`
2. Load dataset, optionally subsample (`max(4, int(N * subsample))` samples)
3. If `--overfit`: shrink to 1 sample, use it for train/val/test, disable DRS (`anneal_iter = n_iters+1`)
   Else: split → `print_dataset_stats` (per-phase pass/fail/unknown for all/train/val/test)
4. Create `DRSSampler` (inactive until `anneal_iter`)
5. Build model, criterion=`BCEWithLogitsLoss`, optimizer=SGD, scheduler=StepLR
6. Training loop: every 10 iters — evaluate, log metrics to console + W&B, then:
   - Save `best_model.pt` (by val F1, only after DRS activates; only when val_f1 improves)
   - Save rolling `model_latest.pt` in same dir (delete previous before writing)
   - Upload both to W&B via `wandb.save()` (`best_model.pt` only if it already exists)
   `model.train()` is explicitly called after every `evaluate()` call (cuDNN RNN requirement)
7. Test evaluation on `best_model.pt`

`evaluate()` returns: `(loss, acc, precision, recall, f1)` — binary classification metrics.

`batch_to_device(batch, device)`:
- Unpacks 7-tuple from default PyTorch collate
- Computes `lengths = [tac.shape[1]] * tac.shape[0]` (uniform since L is fixed)
- Returns 8-tuple including `lengths` for forward-compatibility with variable-length sequences

### sampler.py — `DRSSampler`

Partitions indices into `S=` (pose_label == label) and `S≠` groups.
Starts inactive (uniform sampling). `sampler.activate()` called at `anneal_iter`.
`sigma` = target ratio `|S≠| / |S=|` in each batch. Must be `>= r` (natural dataset ratio).
Yields variable-size batches after activation.

Key constraint: `sigma >= r` (if `sigma == r`, DRS is a no-op; if `sigma < r`, raises `ValueError`).
Sampling uses `replace=True` automatically when `batch_size > len(train_indices)`.

---

## Files

| File | Status | Notes |
|------|--------|-------|
| `dataloader.py` | ✅ Current | All bugs fixed; `uniform_random_split` guards empty splits |
| `model.py` | ✅ Current | ResNet50, modality masking, flat concat, BiLSTM |
| `train.py` | ✅ Current | `--overfit` flag added; `model.train()` bug fixed; W&B defaults set |
| `sampler.py` | ✅ Current | DRS fixed: `replace` guard, `sigma < r` check (was `<=`) |
| `README.md` | ✅ Current | Full documentation: file blurbs, all CLI params, 7 example commands, ablation matrix, W&B sweep YAML |
| `test.ipynb` | ✅ Current | 13-section interactive notebook; all DataLoaders use default collate |
| `CLAUDE.md` | ✅ This file | |

---

## Work In Progress / Next Steps

- **Full training run** not yet executed. Overfit sanity check passed (loss decreased ~0.73→0.35 over 500 iters, val_acc=100% from iter 30). Pipeline is confirmed working.
- **Next:** Run full training with `--split random --anneal_iter 300 --n_iters 600` and monitor W&B for val/f1 after DRS activates.

---

## Quick Reference Commands

```bash
# Smoke test (1% data, DRS on from start)
python train.py --split random --subsample 0.01 --anneal_iter 0 \
    --n_iters 20 --batch_size 2 --num_workers 0 \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
    --wandb_project None

# Single-sample overfit check (confirms model can learn; loss should fall toward 0)
python train.py --overfit --n_iters 500 --batch_size 1 \
    --lr 0.001 --weight_decay 0.0 --num_workers 0 \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
    --wandb_project None

# Full training run (all modalities, W&B logging to TEMU project)
python train.py --split random --anneal_iter 300 --n_iters 600 \
    --L 20 --modalities V T FT G GF \
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
- **BCEWithLogitsLoss + 1 logit** kept (not CrossEntropy + 2 logits)
- **L enforced in dataloader** (not in model.forward) — `_build_sample` drops sequences shorter than L; clips longer ones to `seconds[-L:]` (last L seconds, closest to stability event)
- **Modality masking** via zero-multiplication in `forward()` — disabled modalities still pass through encoders (zeroed input), shape is preserved
- **LSTM operates over T seconds** (one step per second, F1 frames flattened per step) — not over T×F1 individual frames
- **`grasp_label`** stored in `ds.samples` but not returned by `__getitem__` — metadata only, used for dataset stats printing
- **Bucket underfill** (0 < k < F): prints `[WARN]` and skips sample — no forward-fill
- **`--subsample`, `split_by_object`, `split_by_pose`** all confirmed working correctly — kept
- **`collate_variable_length` not used** as `collate_fn` anywhere — all DataLoaders use default PyTorch collate since `L` guarantees uniform T. `lengths` computed in `batch_to_device` from `tac.shape[1]` for forward-compatibility. Method kept in `dataloader.py` for potential future use.
- **`--F1` / `--F2` CLI args** added to `train.py` (default 1 each); set `dataloader.F1` / `dataloader.F2` before dataset construction — defaults match module defaults so no actual override unless changed

## Bugs Fixed

| Location | Bug | Fix |
|----------|-----|-----|
| `sampler.py:132` | `replace=False` crashes when `batch_size > len(train_indices)` | `replace = batch_size > len(all_indices)` |
| `sampler.py:88` | `sigma <= r` raised `ValueError` even when `sigma == r` (valid: keep_prob=1, no-op) | Changed to `sigma < r` |
| `dataloader.py:367` | `uniform_random_split` produced empty val set for small N | `n_val = max(1, ...)`, `n_train = max(1, min(..., n-n_val-1))` |
| `train.py:254` | `torch.load` missing `map_location` → crashes on CPU/GPU mismatch | Added `map_location=device` |
| `train.py:evaluate` | Division by zero when loader has no batches | Added `if n == 0: return 0,0,0,0,0` guard |
| `train.py:288` | `model.train()` not called after `evaluate()` → `RuntimeError: cudnn RNN backward can only be called in training mode` on second iteration | Added `model.train()` after every validation block |
| `model.py:136` | `lstm_out[:, -1, :]` takes backward stream at t=T which has only seen one step (backward context is full at t=0, not t=T) | Use `cat[lstm_out[:,-1,:h], lstm_out[:,0,h:]]` — forward at T + backward at 0 |
| `model.py:43-45` | Constructor defaults `frames_per_sec=2, ft_dim=12, gripper_dim=4` didn't match dataloader constants (F1=1, FT_DIM=6, GR_DIM=2) | Fixed to `frames_per_sec=1, ft_dim=6, gripper_dim=2` |
| `dataloader.py:463` | `lengths_b.tolist()` called on a Python list → `AttributeError` | Removed `.tolist()` |
| `model.py:projection` | `nn.BatchNorm1d(hidden_dim*2)` applied to `(B, T, hidden_dim*2)` — BN treats `dim=1=T` as the channel axis, normalizing over `(batch, features)` per timestep instead of over `(batch, timesteps)` per feature | Replaced with `nn.LayerNorm(hidden_dim*2)`, which normalizes over `dim=-1` independently for each `(B, T)` position |
