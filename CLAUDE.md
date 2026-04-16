# CLAUDE.md

Persistent context for Claude Code. **Update this at the end of each session.**

---

## Project

**TactileEncoderForManipulation** — CMU 11-777 course project.
Predicts slip/drop during robotic grasping from multimodal sensors (RGB, GelSight tactile, force-torque, gripper state).

**Active model:** `MBT/mbt_model.py` — Multimodal Bottleneck Transformer (primary, use this)
**Legacy model:** `model.py` — `GraspStabilityLSTM` (ResNet50 + BiLSTM; **not the active model**)
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

**Pretrained weights:** `/ocean/projects/cis260031p/shared/pretrained/`
T3 large: `models/t3_large/encoders/gs_black.pth` + `models/t3_large/trunk.pth`

---

## Active Architecture: MBT (`MBT/`)

### MBT/mbt_model.py — `MBTGraspStability`

Based on "Attention Bottlenecks for Multimodal Fusion" (Nagrani et al., NeurIPS 2021).

**5 modalities:** T (tactile/GelSight via T3-large), V (RGB via ViT-Base/16), FT (force-torque), G (gripper state), GF (gripper force scalar).

**Architecture summary:**
```
Layers 0..Lf-1  →  unimodal processing per stream (default Lf=8)
Layers Lf..11   →  bottleneck fusion: each stream sees shared Nb bottleneck tokens

Visual streams (T, V):
  ViViT-style tokenisation: patch-embed video frames + factored spatial+temporal pos embeddings
  RGB: ViT-Base/16 (768-d, frozen + AdaptFormer adapters in parallel with FFN)
  Tactile: T3-large (1024-d, 3 encoder + 9 trunk = 12 blocks)
           → unimodal runs at 1024-d
           → Linear(1024→768) projects to fusion dim at the Lf boundary
           → fusion layers use lightweight trainable blocks (not T3 blocks) at 768-d

Temporal streams (FT, G):
  Linear(raw_dim→768) + LayerNorm → CLS token prepended → learned pos embed
  → LightweightTransformerBlock (4 heads, 2× MLP, fully trainable)

Static stream (GF):
  Linear(1→768) + LayerNorm → single token
  → MLP refinement (no self-attention — 1-token attention is a no-op)

Bottleneck fusion (layers Lf..11):
  BottleneckFusionBlock: each stream attends to shared Nb bottleneck tokens
  Nb per-stream bottleneck outputs are averaged → new shared bottleneck state

Classification:
  Per-modality CLS token → per-modality LayerNorm → Linear head → logit
  Final prediction = mean of all active modality logits → (B, num_classes)
```

**Constructor args (defaults):**
```python
MBTGraspStability(
    frames_per_sec=1,
    ft_dim=6,            # FT_DIM from dataloader
    gripper_dim=2,       # GR_DIM from dataloader
    max_timesteps=20,    # L
    num_bottlenecks=4,   # Nb (paper default)
    fusion_layer=8,      # Lf: unimodal 0..7, fusion 8..11 (paper default)
    max_visual_frames=8, # subsample T*F1 frames to this many for ViT
    adapter_dim=64,      # AdaptFormer hidden dim; 0 = no adapter
    dropout=0.1,
    freeze_vit=True,
    modalities=None,     # set of {'V','T','FT','G','GF'}; None = all
    num_classes=1,
    pretrained_dir=None, # defaults to /ocean/projects/cis260031p/shared/pretrained
    t3_encoder_domain='gs_black',  # or 'gs_tag'
)
```

**Modality masking:** inactive modalities zeroed in `forward()` before processing (shapes preserved).

**Key classes:**
- `Adapter` — AdaptFormer bottleneck (down→GELU→up, zero-init → starts as no-op)
- `UnimodalBlock` — frozen ViT block + optional Adapter in parallel with FFN; has `forward_with_bottleneck`
- `LightweightTransformerBlock` — fully trainable pre-norm transformer (4 heads, 2× MLP); has `forward_with_bottleneck`
- `BottleneckFusionBlock` — one N-modality fusion layer; averages per-stream bottleneck updates

**Input tensor shapes to `forward()`:**
```
tactile:       (B, T, F1, 3, 224, 224)
rgb:           (B, T, F1, 3, 224, 224)
ft:            (B, T, FT_DIM)
gripper:       (B, T, GR_DIM)
gripper_force: (B, 1)
```

---

### MBT/mbt_train.py

**Key differences from `train.py` (LSTM):**
- AdamW optimizer (not SGD); cosine LR decay after `anneal_iter`
- Gradient accumulation (`--grad_accum`, default 8; effective batch = micro × accum)
- Mixed precision (`torch.autocast` + `torch.GradScaler`)
- Differential LR: adapter params + tactile fusion blocks at `lr × 0.1`
- DRS activation controlled by separate `--drs_iter` (can be set very high to disable)
- Best model saved by val F1; rolling `model_latest.pt` also saved
- `--L 9` in the job script (not 20) — shorter episodes for MBT

**CLI args (key ones):**
```
--root_dir           dataset path (default ./data)
--split              object | pose | random
--test_objects / --test_poses
--sigma              DRS ratio (default 0.5)
--drs_iter           iteration DRS activates (default 400; set >> n_iters to disable)
--batch_size         micro-batch size (default 4, for V100)
--grad_accum         gradient accumulation steps (default 8; effective = 4×8=32)
--lr                 peak LR (default 1e-4)
--weight_decay       default 0.01
--dropout            default 0.1
--num_bottlenecks    default 4
--fusion_layer       Lf (default 8)
--max_visual_frames  default 8
--adapter_dim        AdaptFormer dim (default 64)
--t3_encoder_domain  gs_black | gs_tag (default gs_black)
--pretrained_dir     T3 weights dir (default shared/pretrained)
--n_iters            default 600
--anneal_iter        cosine decay start (default 300; set > n_iters to disable)
--F1                 image frames per second (default 1)
--F2                 sensor readings per second (default 1)
--L                  max seconds per episode (default 20; job.sh uses 9)
--modalities         active modalities (default: V T FT G GF)
--wandb_project      default "TEMU"; set None to disable
--wandb_entity       default "mrsd-smores"
--model_save_path    default "trained_models/best_mbt_model.pt"
--overfit            1-sample sanity check mode
--subsample          fraction of dataset to load
```

**Execution flow:**
1. Validate modality keys; print effective batch size
2. Set `dataloader.L/F1/F2`; load `PoseItDataset`; optionally subsample
3. Split → `print_dataset_stats`; create `DRSSampler` (deferred)
4. Compute `pos_weight` from train label distribution for `BCEWithLogitsLoss`
5. Build `MBTGraspStability`, print trainable/total param count
6. Differential-LR AdamW + cosine LR lambda + `GradScaler`
7. Training loop with gradient accumulation; DRS activates at `drs_iter`; evaluate every 10 iters
8. Test evaluation on best checkpoint

---

### encoders.py

**`T3TactileEncoder`** (used by MBT):
- Loads T3-large (304M) from `pretrained_dir/models/t3_large/`
- Architecture: 3-block ViT encoder (1024-d) + 9-block trunk (CLS pooling) = 12 blocks total
- Config inferred from checkpoint (embed_dim, depth, num_heads)
- Fallback: tries `gs_tag` if `gs_black` missing (and vice versa)
- In MBT: encoder blocks 0..Lf-1 used as unimodal; projection Linear(1024→768) at fusion boundary
- `freeze=True` by default; `.train()` keeps frozen parts in `.eval()` mode

**`CLIPRGBEncoder`** (legacy, used by old `model.py` variant):
- CLIP ViT-L/14 (LAION-2B), open_clip, 768-d per image; frozen by default
- Not used by MBT (MBT uses timm ViT-Base/16 directly)

---

### dataloader.py — `PoseItDataset`

Module-level constants (mbt_train.py sets these before constructing dataset):
```python
F1     = 1        # image frames sampled per second
F2     = 1        # sensor readings sampled per second
FT_DIM = 6        # F2 * 6
GR_DIM = 2        # F2 * 2
L      = 20       # max seconds per episode (job.sh overrides to 9 for MBT)
phase  = 'grasp+pose'
```

`_build_sample` output per sample (all stored in `ds.samples`):
- `tactile`: `(T, F1, 3, 224, 224)` — GelSight, baseline-subtracted (frame − t_grasp first frame)
- `rgb`: `(T, F1, 3, 224, 224)` — RGB camera
- `ft`: `(T, FT_DIM)` — force-torque, flat per second
- `gripper`: `(T, GR_DIM)` — gripper state, flat per second
- `gripper_force`: `(1,)` — static scalar from folder name (`F<N>`)
- `label`: scalar long — 0=pass, 1=slip/drop (stability phase, **training target**)
- `pose_label`: scalar long — 0=pass, 1=slip/drop (pose phase, used by DRS)
- `grasp_label`: int — 0/1/-1 (grasp phase, **not returned by `__getitem__`**, metadata only)
- `object`, `pose_idx`, `force`, `sample_dir` — metadata

`__getitem__` returns 7-tuple: `(tactile, rgb, ft, gripper, gripper_force, label, pose_label)`

Key behaviors:
- Episodes clipped to last `L` seconds (closest to stability event); episodes shorter than L are dropped
- Buckets with `0 < k < F` frames/readings: `[WARN]` + skip sample
- Buckets with `k == 0`: return zeros / black frames (silent)
- `collate_variable_length` exists but is **not used** (L guarantees uniform T; default collate works)
- Sensor standardization: optional `sensor_stats` dict; compute from train indices via `compute_sensor_stats()`

Split functions: `split_by_object`, `split_by_pose`, `uniform_random_split`

---

### sampler.py — `DRSSampler`

Partitions indices into `S=` (pose_label == label) and `S≠` (pose_label ≠ label).
- Starts inactive (uniform sampling). Call `sampler.activate()` at `drs_iter`.
- `sigma` = target ratio `|S≠|/|S=|` per batch; must be `>= r` (natural ratio)
- When `sigma == r`: keep_prob=1.0 (no-op, still valid)
- `replace=True` automatically when `batch_size > len(indices)` (small datasets)
- Used with `DataLoader(dataset, batch_sampler=sampler)` — not `batch_size=`

---

## Files

| File | Role | Notes |
|------|------|-------|
| `MBT/mbt_model.py` | **Active model** | MBTGraspStability: 5-stream bottleneck transformer |
| `MBT/mbt_train.py` | **Active trainer** | AdamW, cosine LR, grad accum, mixed precision |
| `MBT/job.sh` | SLURM job | V100-32, 8h, GPU-shared; L=9, adapter_dim=128, n_iters=1500 |
| `dataloader.py` | Shared dataloader | PoseItDataset, DRS-aware splits, sensor stats |
| `encoders.py` | Encoders | T3TactileEncoder (MBT uses this), CLIPRGBEncoder (legacy) |
| `sampler.py` | DRS sampler | Deferred resampling for class imbalance |
| `model.py` | Legacy model | GraspStabilityLSTM — ResNet50 + BiLSTM; **not active** |
| `run_training.sbatch` | Legacy SLURM | Runs `train.py` (LSTM); not active |

---

## Quick Reference Commands

```bash
# MBT overfit sanity check (confirms MBT pipeline works end-to-end)
python MBT/mbt_train.py --overfit --n_iters 100 --lr 0.001 --num_workers 0 \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
    --wandb_project None

# MBT smoke test (1% data, no DRS)
python MBT/mbt_train.py --split random --subsample 0.01 \
    --n_iters 50 --batch_size 2 --grad_accum 2 --num_workers 0 \
    --drs_iter 99999 \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
    --wandb_project None

# MBT full training run (matches job.sh config)
python MBT/mbt_train.py \
    --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
    --split random --modalities V T FT G GF \
    --adapter_dim 128 --num_bottlenecks 4 --fusion_layer 8 \
    --n_iters 1500 --anneal_iter 1000 --drs_iter 99999 \
    --batch_size 4 --grad_accum 8 --lr 1e-4 --L 9

# Submit SLURM job (from mlee12's copy — adjust cd path as needed)
sbatch MBT/job.sh

# Dataloader smoke test
python dataloader.py /ocean/projects/cis260031p/shared/dataset/Gelsight
```

---

## Key Design Decisions (MBT)

- **Tactile at native 1024-d during unimodal; project to 768-d at fusion boundary.**
  T3's blocks are 1024-d and can't share 768-d bottleneck tokens directly. Solution: use T3 blocks for unimodal layers 0..Lf-1, then project with `Linear(1024→768)`, and use lightweight trainable blocks for the tactile stream in fusion layers Lf..11.
- **AdaptFormer adapters on RGB (frozen ViT-Base).** Added in parallel with the FFN in each UnimodalBlock. Zero-init so they start as a no-op. `adapter_dim=64` default (128 in job.sh).
- **ViViT-style factored positional embeddings.** Spatial pos from ViT pretrained; trainable temporal embed added per-frame. CLS token gets spatial pos but no temporal pos.
- **Per-modality classifier heads, mean-pooled logit.** Each modality's CLS → its own head; final = mean of all active logits.
- **Differential LR.** Adapter params + tactile fusion stream blocks at `lr × 0.1` (slower update for near-frozen components).
- **L=9 in job.sh** (not the dataloader default of 20) — shorter episodes work better for MBT's fixed token budget.
- **DRS disabled in job.sh** (`--drs_iter 99999`) — class balance addressed instead via `pos_weight` in BCEWithLogitsLoss.

---

## Legacy Architecture: `model.py` — `GraspStabilityLSTM`

**Not the active model.** Kept for reference/ablation. ResNet50 (frozen, 2048-d) × 2 encoders for tactile + RGB, flat concat with FT/gripper/gf, 2-layer BiLSTM, binary logit. Trained with `train.py` + SGD. See old CLAUDE.md entries for full details.
