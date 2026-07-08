"""
Evaluation metrics for sign language translation.

Includes BLEU (1-4), ROUGE-L, METEOR, chrF++, WER, and semantic similarity.
"""

import numpy as np
from typing import List


def compute_bleu(preds: List[str], refs: List[str]):
    import sacrebleu
    preds = [p.lower() for p in preds]
    refs = [r.lower() for r in refs]
    result = {}
    for n in range(1, 5):
        bn = sacrebleu.metrics.BLEU(max_ngram_order=n, tokenize="13a")
        result[f"bleu{n}"] = round(bn.corpus_score(preds, [refs]).score, 2)
    return result


def compute_rouge_l(preds: List[str], refs: List[str]) -> float:
    clean = [(p.lower(), r.lower()) for p, r in zip(preds, refs)
             if isinstance(r, str) and isinstance(p, str) and r.strip() and p.strip()]
    if not clean:
        return 0.0
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return round(float(np.mean([scorer.score(r, p)["rougeL"].fmeasure * 100 for p, r in clean])), 2)
    except ImportError:
        def lcs(a, b):
            m, n = len(a), len(b)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m):
                for j in range(n):
                    dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i + 1][j], dp[i][j + 1])
            return dp[m][n]
        scores = []
        for p, r in clean:
            pt, rt = p.split(), r.split()
            l = lcs(pt, rt)
            prec, rec = l / max(len(pt), 1), l / max(len(rt), 1)
            f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
            scores.append(f1 * 100)
        return round(float(np.mean(scores)), 2)


def compute_meteor(preds: List[str], refs: List[str]) -> float:
    try:
        from nltk.translate.meteor_score import meteor_score as nltk_meteor
        scores = []
        for pred, ref in zip(preds, refs):
            try:
                scores.append(nltk_meteor([ref.split()], pred.split()))
            except Exception:
                scores.append(0.0)
        return np.mean(scores) * 100 if scores else 0.0
    except ImportError:
        return 0.0


def compute_chrf(preds: List[str], refs: List[str]) -> float:
    try:
        import sacrebleu
        chrf = sacrebleu.metrics.CHRF(word_order=2)
        corpus_score = chrf.corpus_score(preds, [refs])
        return round(corpus_score.score, 2)
    except ImportError:
        return 0.0


def compute_wer(preds: List[str], refs: List[str]) -> float:
    total_wer = 0.0
    count = 0
    for pred, ref in zip(preds, refs):
        pred_words = pred.lower().split()
        ref_words = ref.lower().split()
        wer = _edit_distance(pred_words, ref_words)
        if len(ref_words) > 0:
            total_wer += wer / len(ref_words)
        count += 1
    return (total_wer / count) * 100 if count > 0 else 0.0


def compute_semantic_similarity(preds: List[str], refs: List[str], max_samples: int = 200) -> float:
    try:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        n = min(max_samples, len(preds), len(refs))
        embeddings = model.encode(preds[:n] + refs[:n], convert_to_tensor=True)
        pred_emb = embeddings[:n]
        ref_emb = embeddings[n:]
        similarities = util.cos_sim(pred_emb, ref_emb)
        return similarities.diagonal().mean().item() * 100
    except Exception:
        return None


def _edit_distance(s1: List[str], s2: List[str]) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)
    return dp[m][n]
