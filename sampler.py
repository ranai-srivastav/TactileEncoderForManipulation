"""
Deferred-Resampling (DRS) Sampler for PoseIt

Training examples are split into two groups:
  S=  : examples where pose_label == label  (majority, ~80%)
  S≠  : examples where pose_label != label  (minority, ~20%)

A desired sampling ratio σ > r is set (r = |S≠| / |S=|).
Each batch B is constructed from a pre-sampled batch B̃ as follows:
  1. Sample B̃ of fixed size |B̃| uniformly at random.
  2. For each x ∈ B̃ ∩ S=, keep x with probability r/σ  (drop otherwise).
  3. B = kept S= examples  ∪  (B̃ ∩ S≠)

This makes the ratio of S≠ to S= updates equal to σ in expectation.

DRS is DEFERRED: it does not activate until you call sampler.activate().
Before activation, the sampler behaves like a standard random batch sampler.

Notes
-----
- Use batch_sampler=sampler (not batch_size=) with DataLoader.
- When working with a Subset, pass the underlying dataset and the
  subset's indices via the `indices` argument.
- After DRS activates, the effective batch size will be smaller than
  `batch_size` on average (by a factor of roughly σ/(σ+1) for a
  balanced B̃ split), which is expected and matches the paper.
"""

import numpy as np
import torch
from torch.utils.data import Sampler
from typing import Iterator, List, Optional


class DRSSampler(Sampler):
    """
    dataset : PoseItDataset
        The full dataset. Needs `.samples` list where each entry has
        'label' (shake label) and 'pose_label' keys.
    sigma : float
        Desired S≠ / S= ratio after resampling. Must be > r.
        Paper uses σ ∈ {0.5, 1.0}.
    batch_size : int
        Size of the pre-sampled batch B̃ (before DRS thinning).
        Paper uses 200.
    indices : list[int] or None
        If using a Subset, pass the subset's indices here so the sampler
        only draws from those examples.
    seed : int or None
        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        dataset,
        sigma: float = 1.0,
        batch_size: int = 200,
        indices: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        self.dataset   = dataset
        self.sigma     = sigma
        self.batch_size = batch_size
        self.rng       = np.random.default_rng(seed)
        self._active   = False   # DRS is deferred until activate() is called

        # Use provided indices (Subset case) or all indices
        all_indices = indices if indices is not None else list(range(len(dataset)))

        # Partition into S= and S≠
        s_eq, s_neq = [], []
        for i in all_indices:
            s = dataset.samples[i]
            label      = int(s['label'])
            pose_label = int(s['pose_label'])
            if label == pose_label:
                s_eq.append(i)
            else:
                s_neq.append(i)

        self.s_eq  = np.array(s_eq,  dtype=np.int64)
        self.s_neq = np.array(s_neq, dtype=np.int64)
        self.all_indices = np.array(all_indices, dtype=np.int64)

        r = len(self.s_neq) / max(len(self.s_eq), 1)
        self.r = r

        if len(self.s_neq) > 0 and sigma <= r:
            raise ValueError(
                f"sigma ({sigma}) must be > r ({r:.4f}). "
                "Increase sigma or check your dataset split."
            )

        self.keep_prob = r / sigma  # probability of keeping an S= example

        n_total = len(all_indices)
        # Estimate number of batches per epoch using the full pool size
        self._n_batches = max(1, n_total // batch_size)

        print(
            f"[DRSSampler] |S=|={len(self.s_eq)}, |S≠|={len(self.s_neq)}, "
            f"r={r:.4f}, σ={sigma}, keep_prob(S=)={self.keep_prob:.4f}, "
            f"batch_size={batch_size}, deferred=True"
        )

    def activate(self):
        """Enable DRS resampling. Call this at the LR annealing step."""
        if not self._active:
            self._active = True
            print("[DRSSampler] DRS activated.")

    def deactivate(self):
        """Disable DRS (revert to uniform random sampling)."""
        if self._active:
            self._active = False
            print("[DRSSampler] DRS deactivated.")

    @property
    def is_active(self) -> bool:
        return self._active

    def __len__(self) -> int:
        return self._n_batches

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self._n_batches):
            yield self._sample_batch()

    def _sample_batch(self) -> List[int]:
        # Step 1: Draw B̃ uniformly at random
        b_tilde_idx = self.rng.choice(
            self.all_indices, size=self.batch_size, replace=False
        )

        if not self._active:
            # Pre-activation: return B̃ as-is (standard random batching)
            return b_tilde_idx.tolist()

        # Step 2 & 3: Apply DRS thinning
        s_neq_set = set(self.s_neq.tolist())

        b_eq  = []   # S= examples that survive thinning  →  B=
        b_neq = []   # S≠ examples (always kept)          →  B̃ ∩ S≠

        for idx in b_tilde_idx:
            if idx in s_neq_set:
                b_neq.append(idx)
            else:
                # Keep with probability r/σ
                if self.rng.random() < self.keep_prob:
                    b_eq.append(idx)

        batch = b_eq + b_neq

        # Safety: if thinning left us with an empty batch (rare edge case),
        # fall back to the full B̃
        if len(batch) == 0:
            return b_tilde_idx.tolist()

        return batch