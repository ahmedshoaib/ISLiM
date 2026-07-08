#!/usr/bin/env python3
"""
Word-length bucketed analysis (Table 6).

Bins test sentences by reference word count and computes per-bucket
mean/median BLEU-4 and ROUGE-L.

Usage:
    python scripts/eval/length_analysis.py --checkpoint checkpoints/best_model.pt
"""

import sys
import json
import os
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sacrebleu

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import POSE_STITCH_CSV_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WORD_BINS = [
    (1, 4),
    (5, 7),
    (8, 10),
    (11, 13),
    (14, 16),
    (17, 20),
    (21, 25),
    (26, 30),
    (31, 100),
]


def compute_sentence_bleu(pred, ref):
    if not pred.strip() or not ref.strip():
        return 0.0
    try:
        b = sacrebleu.metrics.BLEU(
            max_ngram_order=4, tokenize="13a", smooth_method="exp",
            effective_order=True,
        )
        return b.sentence_score(pred.strip().lower(), [ref.strip().lower()]).score
    except Exception:
        return 0.0


def compute_rouge_l(pred, ref):
    p, r = pred.lower().split(), ref.lower().split()
    if not p or not r:
        return 0.0
    lp, lr = len(p), len(r)
    dp = [[0] * (lr + 1) for _ in range(lp + 1)]
    for i in range(lp):
        for j in range(lr):
            dp[i + 1][j + 1] = (dp[i][j] + 1 if p[i] == r[j]
                                else max(dp[i + 1][j], dp[i][j + 1]))
    lcs_len = dp[lp][lr]
    prec = lcs_len / max(lp, 1)
    rec = lcs_len / max(lr, 1)
    return (2 * prec * rec / (prec + rec) * 100) if (prec + rec) > 0 else 0.0


def text_to_word_count(text):
    return len(text.strip().split())


def main():
    parser = argparse.ArgumentParser(description="Word-length analysis")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    global DEVICE
    if args.cpu:
        DEVICE = torch.device("cpu")

    logger.info(f"Device: {DEVICE}")

    from src.islim.model import ISLiMModel
    from src.data.dataset import PoseStitchDataset, collate_fn

    config = {
        "encoder": {
            "keypoint_dim": 4, "num_keypoints": 94,
            "temporal_channels": [376, 256, 256], "temporal_kernel": 5,
            "transformer_embed_dim": 512, "transformer_num_heads": 8,
            "transformer_num_layers": 4, "transformer_dropout": 0.2,
            "output_dim": 512, "max_frames": 128,
            "num_query_tokens": 16,
        },
        "t5": {
            "t5_model": "t5-small", "d_model": 512,
            "max_target_len": 128, "label_smoothing": 0.0,
        },
    }
    model = ISLiMModel(config)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}): {n_params:.1f}M params")

    test_ds = PoseStitchDataset("test", csv_dir=str(POSE_STITCH_CSV_DIR))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    logger.info(f"Test samples: {len(test_ds)}")

    all_preds, all_refs = [], []
    for batch in tqdm(test_loader, desc="Generating"):
        kp = batch["keypoints"].to(DEVICE)
        refs = batch["text"]
        try:
            gen = model.generate(kp, num_beams=4, repetition_penalty=1.5,
                                 no_repeat_ngram_size=3, do_sample=False)
        except Exception:
            gen = [""] * kp.size(0)
        all_preds.extend(gen)
        all_refs.extend(refs)

    buckets = {
        f"{lo}-{hi}": {"bleu4": [], "rouge_l": [], "count": 0, "pred_lens": [], "ref_lens": []}
        for lo, hi in WORD_BINS
    }

    for pred, ref in zip(all_preds, all_refs):
        wc = text_to_word_count(ref)
        for lo, hi in WORD_BINS:
            if lo <= wc <= hi:
                key = f"{lo}-{hi}"
                bleu = compute_sentence_bleu(pred, ref)
                rouge = compute_rouge_l(pred, ref)
                buckets[key]["bleu4"].append(bleu)
                buckets[key]["rouge_l"].append(rouge)
                buckets[key]["count"] += 1
                buckets[key]["pred_lens"].append(text_to_word_count(pred))
                buckets[key]["ref_lens"].append(wc)
                break

    print(f"\n{'Bucket':>10}  {'Count':>6}  {'Mean B4':>8}  {'Med B4':>7}  "
          f"{'Mean RL':>8}  {'Med RL':>7}  {'Ref len':>8}  {'Pred len':>8}")
    print("-" * 78)
    results = {}
    for lo, hi in WORD_BINS:
        key = f"{lo}-{hi}"
        b = buckets[key]
        if b["count"] == 0:
            continue
        mean_b4 = np.mean(b["bleu4"])
        med_b4 = np.median(b["bleu4"])
        mean_rl = np.mean(b["rouge_l"])
        med_rl = np.median(b["rouge_l"])
        mean_ref = np.mean(b["ref_lens"])
        mean_pred = np.mean(b["pred_lens"])
        print(f"{key:>10}  {b['count']:>6}  {mean_b4:>8.2f}  {med_b4:>7.2f}  "
              f"{mean_rl:>8.2f}  {med_rl:>7.2f}  {mean_ref:>8.1f}  {mean_pred:>8.1f}")
        results[key] = {
            "count": b["count"],
            "mean_bleu4": round(mean_b4, 2),
            "median_bleu4": round(med_b4, 2),
            "mean_rouge_l": round(mean_rl, 2),
            "median_rouge_l": round(med_rl, 2),
            "mean_ref_len": round(mean_ref, 1),
            "mean_pred_len": round(mean_pred, 1),
        }

    valid_preds, valid_refs = [], []
    for p, r in zip(all_preds, all_refs):
        if p.strip() and r.strip():
            valid_preds.append(p.lower().strip())
            valid_refs.append(r.lower().strip())
    corpus_b4 = sacrebleu.metrics.BLEU(max_ngram_order=4, tokenize="13a").corpus_score(
        valid_preds, [valid_refs]).score
    print(f"\nCorpus BLEU-4 (all valid): {corpus_b4:.2f}")

    out = {
        "checkpoint": args.checkpoint,
        "total_samples": len(test_ds),
        "valid_pairs": len(valid_preds),
        "corpus_bleu4": round(corpus_b4, 2),
        "bins": results,
    }
    out_path = Path(args.checkpoint).parent / "word_length_analysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
