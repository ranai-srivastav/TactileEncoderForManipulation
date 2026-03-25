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

    Returns:
        rho_tac, rho_rgb, rho_prop: scalar floats
    """
    eps = 1e-8
    sum_tac = s_tac.sum()
    sum_rgb = s_rgb.sum()
    sum_prop = s_prop.sum()
    
    rho_tac = (sum_tac / ((sum_rgb + sum_prop) / 2 + eps)).item()
    rho_rgb = (sum_rgb / ((sum_tac + sum_prop) / 2 + eps)).item()
    rho_prop = (sum_prop / ((sum_rgb + sum_tac) / 2 + eps)).item()
    
    return rho_tac, rho_rgb, rho_prop
    
    