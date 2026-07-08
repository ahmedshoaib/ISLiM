"""
Semantic similarity loss using SentenceBERT.

Computes cosine similarity between model decoder hidden states and
precomputed reference text embeddings.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SemanticLoss:
    def __init__(self, texts, device, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.device = device
        logger.info(f"Computing {len(texts)} SentenceBERT embeddings (CPU) ...")
        bert = SentenceTransformer(model_name)
        batch_size = 256
        embs = []
        for i in range(0, len(texts), batch_size):
            batch_texts = [t.lower() for t in texts[i:i + batch_size]]
            with torch.no_grad():
                e = bert.encode(batch_texts, convert_to_tensor=True,
                                show_progress_bar=False,
                                normalize_embeddings=True)
            embs.append(e)
        self.ref_embeddings = torch.cat(embs, dim=0)
        self.proj = nn.Linear(512, 384, bias=False).to(device)
        logger.info(f"  Embeddings: {self.ref_embeddings.shape}  Proj: 512->384")

    def __call__(self, decoder_hidden, indices):
        if isinstance(decoder_hidden, tuple):
            decoder_hidden = decoder_hidden[-1]
        pooled = decoder_hidden.mean(dim=1)
        proj = self.proj(pooled)
        proj = F.normalize(proj, dim=-1)
        refs = self.ref_embeddings[indices].to(proj.device)
        cos = (proj * refs).sum(dim=-1)
        return (1 - cos).mean()
