import torch
import torch.nn.functional as F

def compute_confidence_scores(logit_tac, logit_rgb, logit_prop, targets):
    """
    Compute per-modality confidence scores s^u for each sample in the batch.

    s^u_i = sigmoid(logit^u_i) if label=1, else 1 - sigmoid(logit^u_i).

    Args:
        logit_tac:  (B, 1) raw logits from tactile head
        logit_rgb:  (B, 1) raw logits from RGB head
        logit_prop: (B, 1) raw logits from proprioceptive head
        targets:    (B,)   ground truth binary labels (0 or 1)

    Returns:
        s_tac, s_rgb, s_prop: each (B,) confidence scores in [0, 1]
    """
    def _score(logit, targets):
        if logit is None:
            return None
        prob = torch.sigmoid(logit.squeeze(1))
        s = torch.where(targets == 1, prob, 1.0 - prob)
        return s
    
    return _score(logit_tac, targets), _score(logit_rgb, targets), _score(logit_prop, targets)


def compute_discrepancy_ratios(s_tac, s_rgb, s_prop):
    """
    Compute discrepancy ratio rho^u for each modality (Eq. 9, extended to 3 modalities).

    rho^u = sum(s^u) / mean(sum of other modalities)
    rho^u > 1 → this modality is dominating.
    rho^u < 1 → this modality is under-optimized.
    rho^u = 0.0 → this modality is inactive (was None).

    Returns:
        rho_tac, rho_rgb, rho_prop: scalar floats
    """
    eps = 1e-8

    # Count how many modalities are active
    active_scores = [s for s in [s_tac, s_rgb, s_prop] if s is not None]
    if len(active_scores) < 2:
        # Can't compute a meaningful ratio with only 1 (or 0) active modalities
        # Just return the raw sum for the active one, 0.0 for inactive
        return (
            s_tac.sum().item()  if s_tac  is not None else 0.0,
            s_rgb.sum().item()  if s_rgb  is not None else 0.0,
            s_prop.sum().item() if s_prop is not None else 0.0,
        )

    # Sum confidence scores across the batch for each active modality
    # Inactive modalities get 0.0 so they don't pollute the ratios
    sum_tac  = s_tac.sum()  if s_tac  is not None else torch.tensor(0.0)
    sum_rgb  = s_rgb.sum()  if s_rgb  is not None else torch.tensor(0.0)
    sum_prop = s_prop.sum() if s_prop is not None else torch.tensor(0.0)

    # For each modality, rho = its sum / average sum of the other active modalities
    # We only divide by active others — inactive ones (sum=0) are excluded from the average
    def _rho(my_sum, other_sums):
        active_others = [s for s in other_sums if s is not None]
        if not active_others:
            return 0.0
        avg_others = sum(active_others) / len(active_others)
        return (my_sum / (avg_others + eps)).item()

    rho_tac  = _rho(sum_tac,  [sum_rgb  if s_rgb  is not None else None,
                            sum_prop if s_prop is not None else None]) if s_tac  is not None else 0.0
    rho_rgb  = _rho(sum_rgb,  [sum_tac  if s_tac  is not None else None,
                                sum_prop if s_prop is not None else None]) if s_rgb  is not None else 0.0
    rho_prop = _rho(sum_prop, [sum_tac  if s_tac  is not None else None,
                                sum_rgb  if s_rgb  is not None else None]) if s_prop is not None else 0.0

    return rho_tac, rho_rgb, rho_prop    


def compute_ogm_coefficients(rho_tac, rho_rgb, rho_prop, alpha):
    """
    Compute per-modality gradient scaling coefficients k^u (Eq. 10).

    k^u = 1 - tanh(alpha * rho^u)  if rho^u > 1  (modality is dominating → slow it down)
    k^u = 1                         otherwise      (modality is under-optimized → leave it)

    Args:
        rho_tac, rho_rgb, rho_prop: scalar floats from compute_discrepancy_ratios
        alpha: float, controls how aggressively to modulate

    Returns:
        k_tac, k_rgb, k_prop: scalar floats in (0, 1]
    """
    import math

    def _k(rho):
        if rho == 0.0:
            # inactive modality
            return 1.0
        if rho > 1:
            return 1.0 - math.tanh(alpha * rho)
        return 1.0

    return _k(rho_tac), _k(rho_rgb), _k(rho_prop)