# TODO

Open issues identified during code review.

> **Already addressed (not repeated below):**
> - Broadcasting bug (`logits (B,1)` vs `label (B,)`) → fixed with `.squeeze(1)` in train loop and `evaluate()`
> - Temporal BatchNorm1d on `(B,T,F)` sequences → replaced with `LayerNorm`
> - Exploding gradients → `clip_grad_norm_(max_norm=1.0)` added
> - GRU used in LSTM-named class → noted, intentionally kept as GRU
> - `--overfit` crashes at test eval → **fixed**: removed `sampler.is_active` gate; `best_model.pt` now saved on any `val_f1` improvement
> - Test eval crashes if `best_model.pt` never saved → **fixed**: dual test eval with `os.path.exists` guard per checkpoint
> - `frames_per_sec=F2` should be `F1` in `train.py` and `test.ipynb` → **fixed**
> - Stale `FT_DIM`/`GR_DIM` if `--F2` non-default → **fixed**: recomputed after `_dl.F2 = args.F2`
> - `test.ipynb` had no F1/F2 config; `_dl.F1`/`_dl.F2` never set → **fixed**: `F1_CFG`/`F2_CFG`/`HIDDEN_DIM` in Section 0, override in Section 1
> - `test.ipynb` `hidden_dim=512` hardcoded → **fixed**: replaced with `HIDDEN_DIM` config var

---

## Cleanup

### 1. `_PHASE_BOUNDS` dead code — `dataloader.py:64-69`
```python
_PHASE_BOUNDS = {
    'grasp':     ('grasping',  'pose'),
    'pose':      ('pose',      'stability'),
    'stability': ('stability', 'retract'),
}
```
This dict is defined but never referenced anywhere. Remove it.

---

## Documentation

### 2. `README.md` needs updating
The README still references old/removed CLI args and outdated defaults:
- `--drs_iter` now exists (was previously `--anneal_iter` for DRS)
- `--lstm_layers` and `--unidirectional` now exist
- `--sigma` default is 0.5 (not 1.0)
- Optimizer is SGD + StepLR (not AdamW + CosineAnnealingLR)
- Example commands may have invalid flags

---

## Architecture Improvements

### 3. Modality drowning — FT and gripper signals drowned by vision embeddings
**Problem:** The projection MLP concatenates a `4096-d` vision block (2×2048) with a `6-d` FT
vector and a `2-d` gripper vector into a single 4105-d input. During backpropagation, the
high-dimensional vision gradient dominates, and the GRU learns to mathematically ignore the
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

### 4. DRS causes covariate shift — reconsider mid-training distribution change
**Problem:** Activating `DRSSampler` at `drs_iter` abruptly changes the batch distribution.
The representations built by the frozen ResNets and the GRU during natural-distribution training
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

### 5. Migrate from BiGRU to Multimodal Transformer (MulT)
**Motivation:** The current architecture assumes temporal alignment across modalities (vision @
~10 Hz, FT @ ~70 Hz, GelSight @ ~22 Hz are all downsampled to 1 Hz). A cross-modal Transformer
can handle unaligned multimodal streams natively via cross-attention — allowing, e.g., the tactile
stream to attend to temporally-shifted visual features.

**Reference:** Tsai et al. (2019), *Multimodal Transformer for Unaligned Multimodal Language
Sequences (MulT)*. [https://aclanthology.org/P19-1656/](https://aclanthology.org/P19-1656/)

**Migration path:**
- Replace the BiGRU with a stack of cross-modal attention blocks (T→V, V→T, FT→T, etc.)
- Each modality gets its own positional encoding
- The projection MLP per-second stays; GRU is replaced by Transformer encoder layers
