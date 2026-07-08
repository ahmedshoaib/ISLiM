"""
CLIP-style contrastive loss between visual and text modalities.
"""

import torch
import torch.nn.functional as F


def clip_contrastive_loss(visual_features, text_ids, text_mask, token_embeddings, logit_scale):
    """Bidirectional CLIP contrastive: mean-pool visual ↔ mean-pool text embeddings.

    Args:
        visual_features: (B, N, D) visual encoder output (N query tokens or T frames)
        text_ids:        (B, L) token IDs (pad_token_id for masked positions)
        text_mask:       (B, L) attention mask (0 = pad)
        token_embeddings: T5 shared embedding layer (nn.Embedding)
        logit_scale:     nn.Parameter (learned temperature)

    Returns:
        clip_loss: scalar loss
        _diag:     mean cosine similarity of correct pairs (diagnostic)
        _offdiag:  mean absolute similarity of incorrect pairs (diagnostic)
    """
    device = visual_features.device
    visual_pooled = visual_features.mean(dim=1).float()
    visual_pooled = F.normalize(visual_pooled, dim=-1)

    text_embeds = token_embeddings(text_ids).float()
    text_mask_3d = text_mask.unsqueeze(-1).float().to(device)
    text_pooled = (text_embeds * text_mask_3d).sum(dim=1) / text_mask_3d.sum(dim=1).clamp(min=1)
    text_pooled = F.normalize(text_pooled, dim=-1)

    sim = visual_pooled @ text_pooled.T
    logits = sim * logit_scale.exp()
    labels = torch.arange(len(logits), device=device)
    clip_loss = (F.cross_entropy(logits, labels) +
                 F.cross_entropy(logits.T, labels)) / 2

    d = sim.diag()
    off_diag_mask = 1 - torch.eye(len(sim), device=device)
    _diag = d.detach().mean().item()
    _offdiag = (sim * off_diag_mask).abs().detach().mean().item()

    return clip_loss, _diag, _offdiag
