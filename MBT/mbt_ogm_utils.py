import torch
import math


def compute_confidence_scores(logits_per_mod: dict, targets: torch.Tensor) -> dict:
    """
    Compute per-modality confidence scores s^u_i = P(correct class | modality u, sample i).

    For binary classification: sigmoid(logit) if label=1, else 1 - sigmoid(logit).

    Args:
        logits_per_mod: dict[str -> (B, 1)] raw logits per active modality
        targets:        (B,) ground truth binary labels

    Returns:
        dict[str -> (B,)] confidence scores in [0, 1]
    """
    scores = {}
    for key, logit in logits_per_mod.items():
        prob = torch.sigmoid(logit.squeeze(-1))                    # (B,)
        scores[key] = torch.where(targets == 1, prob, 1.0 - prob)
    return scores


def compute_discrepancy_ratios(scores: dict) -> dict:
    """
    Compute discrepancy ratio rho^u for each active modality (Eq. 9, N-modality extension).

    rho^u = sum(s^u) / mean(sum of all other active modalities)
    rho^u > 1  -> dominating
    rho^u < 1  -> under-optimized

    Args:
        scores: dict[str -> (B,)] from compute_confidence_scores

    Returns:
        dict[str -> float] discrepancy ratios
    """
    eps = 1e-8
    if len(scores) < 2:
        return {k: 1.0 for k in scores}

    sums  = {k: v.sum().item() for k, v in scores.items()}
    total = sum(sums.values())
    n     = len(sums)

    return {
        key: sums[key] / ((total - sums[key]) / (n - 1) + eps)
        for key in sums
    }


def compute_ogm_coefficients(rhos: dict, alpha: float) -> dict:
    """
    Compute per-modality gradient scaling coefficients k^u (Eq. 10).

    k^u = 1 - tanh(alpha * rho^u)  if rho^u > 1  (dominating -> slow down)
    k^u = 1.0                       otherwise

    Args:
        rhos:  dict[str -> float]
        alpha: float

    Returns:
        dict[str -> float] in (0, 1]
    """
    def _k(rho):
        return 1.0 - math.tanh(alpha * rho) if rho > 1.0 else 1.0

    return {k: _k(v) for k, v in rhos.items()}