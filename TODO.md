# TODO

Open issues identified during code review (2026-03-10).

> **Already addressed (not repeated below):**
> - Broadcasting bug (`logits (B,1)` vs `label (B,)`) → fixed with `.squeeze(1)` in train loop and `evaluate()`
> - Temporal BatchNorm1d on `(B,T,F)` sequences → replaced with `LayerNorm`
> - Exploding gradients → `clip_grad_norm_(max_norm=1.0)` added
> - GRU used in LSTM-named class → changed to `nn.LSTM`

---

## Bugs

### 1. `--overfit` crashes at test evaluation — `train.py:351`
`best_model.pt` is only saved when `sampler.is_active and val_f1 > best_val_f1`.
In overfit mode, `anneal_iter = n_iters + 1`, so DRS never activates → `sampler.is_active`
is always `False` → `best_model.pt` is never written → `torch.load(args.model_save_path)`
crashes with `FileNotFoundError`.

**Fix options (pick one):**
- Skip the `sampler.is_active` gate in overfit mode, or
- Fall back to `latest_path` at test time if `best_model.pt` doesn't exist, or
- Save `best_model.pt` unconditionally in overfit mode

---

### 2. Test evaluation crashes if `best_model.pt` was never saved — `train.py:351`
More general case of #1: if `val_f1` is always 0 (e.g. model predicts all-negative and DRS
never activates in time), `best_model.pt` is never written.

**Fix:** Add a guard before loading:
```python
if not os.path.exists(args.model_save_path):
    print("[WARN] best_model.pt not found — using latest checkpoint")
    args.model_save_path = latest_path
model.load_state_dict(torch.load(args.model_save_path, map_location=device))
```

---

### 3. `frames_per_sec=F2` should be `F1` — `train.py:250`
```python
model = GraspStabilityLSTM(
    frames_per_sec=F2,   # ← wrong: should be F1 (image frames/sec)
```
`frames_per_sec` controls the vision embedding width (`frames_per_sec * 2048 * 2`).
It should be `F1` (image frames per second), not `F2` (sensor readings per second).
Currently harmless because `F1 == F2 == 1` always, but wrong in principle and will
silently corrupt the model if `--F1 != --F2`.

**Fix:** Change to `frames_per_sec=args.F1` (use the CLI arg directly, not the stale import).

---

### 4. Stale `FT_DIM` / `GR_DIM` if `--F2` is non-default — `train.py:30,251-252`
```python
from dataloader import (..., F2, FT_DIM, GR_DIM)  # values captured at import time
# ...
_dl.F2 = args.F2   # updates the module, but local FT_DIM/GR_DIM are NOT updated
model = GraspStabilityLSTM(ft_dim=FT_DIM, gripper_dim=GR_DIM, ...)  # stale!
```
If `--F2 2` is passed, `_dl.FT_DIM` should be 12 but `FT_DIM` in train.py is still 6.
The model input dim and dataloader output dim will mismatch → silent shape error.

**Fix:** After setting `_dl.F2 = args.F2`, recompute locally:
```python
FT_DIM = _dl.F2 * 6
GR_DIM = _dl.F2 * 2
```
Or just use `_dl.FT_DIM` / `_dl.GR_DIM` directly at the model construction site.

---

## Cleanup

### 5. `_PHASE_BOUNDS` dead code — `dataloader.py:64-69`
```python
_PHASE_BOUNDS = {
    'grasp':     ('grasping',  'pose'),
    'pose':      ('pose',      'stability'),
    'stability': ('stability', 'retract'),
}
```
This dict is defined but never referenced anywhere. Remove it.

---

## Architecture Improvements

### 8. Modality drowning — FT and gripper signals drowned by vision embeddings
**Problem:** The projection MLP concatenates a `4096-d` vision block (2×2048) with a `6-d` FT
vector and a `2-d` gripper vector into a single 4105-d input. During backpropagation, the
high-dimensional vision gradient dominates, and the LSTM learns to mathematically ignore the
low-dimensional FT/gripper signals — even though those signals are often the most diagnostic
for slip detection.

**Reference:** Lee et al. (2019), *Making Sense of Vision and Touch* — heterogeneous modalities
must be projected to a common latent space before fusion. [https://arxiv.org/abs/1810.10191](https://arxiv.org/abs/1810.10191)

**Fix:** In `model.py`, add small MLPs to project FT and gripper into a higher-dimensional space
before concatenation:
```python
self.ft_proj      = nn.Sequential(nn.Linear(ft_dim,      256), nn.ReLU())
self.gripper_proj = nn.Sequential(nn.Linear(gripper_dim, 128), nn.ReLU())
# then concat [tac_emb (F1*2048), rgb_emb (F1*2048), ft_proj (256), grip_proj (128), gf (1)]
```
Adjust `pre_lstm_dim` in the constructor accordingly.

---

### 9. DRS causes covariate shift — reconsider mid-training distribution change
**Problem:** Activating `DRSSampler` at `anneal_iter` abruptly changes the batch distribution.
The representations built by the frozen ResNets and the LSTM during natural-distribution training
are suddenly forced to adapt to an artificial distribution. Per Kang et al. (2019), resampling
mid-training damages the quality of the learned feature space.

**Reference:** Kang et al. (2019), *Decoupling Representation and Classifier for Long-Tailed
Recognition* — the optimal strategy is to train representations on the natural (unmodified)
distribution, then freeze the backbone and adjust only the final classifier.
[https://arxiv.org/abs/1910.09217](https://arxiv.org/abs/1910.09217)

**Recommended approach (τ-normalization):**
1. Train entire model end-to-end on natural distribution (no DRS).
2. Freeze everything except the final `nn.Linear(64 → 1)` layer.
3. Apply τ-normalization: scale the classifier weight vector by
   `w_normalized = w / ||w||^τ` with `τ ∈ [0, 1]` to geometrically re-center the
   decision boundary toward the minority class without corrupting representations.

This replaces the DRS sampler entirely for handling class imbalance.

---

## Future Work

### 10. Migrate from BiLSTM to Multimodal Transformer (MulT)
**Motivation:** The current architecture assumes temporal alignment across modalities (vision @
~10 Hz, FT @ ~70 Hz, GelSight @ ~22 Hz are all downsampled to 1 Hz). A cross-modal Transformer
can handle unaligned multimodal streams natively via cross-attention — allowing, e.g., the tactile
stream to attend to temporally-shifted visual features.

**Reference:** Tsai et al. (2019), *Multimodal Transformer for Unaligned Multimodal Language
Sequences (MulT)*. [https://aclanthology.org/P19-1656/](https://aclanthology.org/P19-1656/)

**Migration path:**
- Replace the BiLSTM with a stack of cross-modal attention blocks (T→V, V→T, FT→T, etc.)
- Each modality gets its own positional encoding
- The projection MLP per-second stays; LSTM is replaced by Transformer encoder layers

---

## Documentation / Stale Files

### 6. `README.md` needs updating
The following CLI args were removed or changed and README still references the old API:
- Removed: `--drs_iter`, `--lstm_layers`, `--unidirectional`
- Changed: `--sigma` default 0.5 → 1.0; optimizer SGD → AdamW; scheduler StepLR → CosineAnnealingLR
- Example commands in README may have invalid flags

### 7. `test.ipynb` may reference stale model constructor args
`GraspStabilityLSTM` no longer accepts `bidirectional` or `lstm_layers` kwargs.
Any notebook cell instantiating the model with those args will crash.
