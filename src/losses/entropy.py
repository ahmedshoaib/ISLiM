"""
Entropy regularization to prevent model collapse.
"""

import torch
import torch.nn.functional as F


def entropy_regularization(logits, label_mask, target_entropy=3.0):
    """Penalize overconfident (low-entropy) distributions.

    Collapsed models put all probability mass on 1-2 tokens (entropy ≈ 0).
    Healthy models spread mass (entropy ≈ 3-8). This encourages exploration.

    Adaptive: when the model is already uncertain (high entropy), penalty → 0.
    When the model is overconfident (low entropy), penalty = (target - actual)^2.
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    per_token_entropy = -(probs * log_probs).sum(dim=-1)
    shortfall = torch.clamp(target_entropy - per_token_entropy, min=0)
    mask = label_mask.float()
    return (shortfall ** 2 * mask).sum() / mask.sum().clamp(min=1)
