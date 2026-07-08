#!/usr/bin/env python3
"""
Full evaluation script — 18 metrics on PoseStitch test set.

Computes and prints: sacreBLEU B1-B4 (corpus), PS-BLEU B1-B4,
ROUGE-L, METEOR, chrF++, SemSim, WER, sentence zero-BLEU rate.

Usage:
    python scripts/eval/evaluate_all.py --checkpoint checkpoints/best_model.pt --split test
    python scripts/eval/evaluate_all.py --checkpoint checkpoints/best_model.pt --split test --csv_dir data/splits
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import POSE_STITCH_CSV_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: str):
    from src.islim.model import ISLiMModel

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
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model = model.to(DEVICE).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Loaded checkpoint (epoch {state.get('epoch', '?')}): {n_params:.1f}M params")
    return model


def compute_all_metrics(predictions: List[str], references: List[str], label: str = "") -> Dict:
    from src.metrics import (
        compute_bleu, compute_bleu_verbose,
        compute_rouge, compute_wer, compute_meteor,
        compute_semantic_similarity, compute_chrf,
    )

    clean_preds, clean_refs = [], []
    for p, r in zip(predictions, references):
        if isinstance(r, str) and isinstance(p, str) and r.strip() and p.strip():
            clean_preds.append(p.strip().lower())
            clean_refs.append(r.strip().lower())

    n_total = len(predictions)
    n_empty = n_total - len(clean_preds)
    logger.info(f"  [{label}] {n_total} total, {n_empty} empty/invalid")

    if len(clean_preds) == 0:
        return {"error": "No valid predictions", "n_total": n_total}

    metrics = {"n_total": n_total, "n_valid": len(clean_preds)}

    t0 = time.time()
    bleu = compute_bleu(clean_preds, clean_refs)
    metrics.update({f"bleu{n}": bleu.get(f"bleu{n}", 0.0) for n in range(1, 5)})
    logger.info(f"  [{label}] sacreBLEU: B4={metrics.get('bleu4', 0):.2f} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    ps_bleu = compute_ps_bleu(clean_preds, clean_refs)
    metrics.update(ps_bleu)
    logger.info(f"  [{label}] PS-BLEU: B4={metrics.get('ps_b4', 0):.2f} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    rouge = compute_rouge(clean_preds, clean_refs)
    metrics["rouge_l"] = rouge
    logger.info(f"  [{label}] ROUGE-L: {rouge:.2f} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    chrf = compute_chrf(clean_preds, clean_refs)
    metrics["chrf"] = chrf
    logger.info(f"  [{label}] chrF++: {chrf:.2f} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    meteor = compute_meteor(clean_preds, clean_refs)
    metrics["meteor"] = meteor
    logger.info(f"  [{label}] METEOR: {meteor:.2f} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    wer = compute_wer(clean_preds, clean_refs)
    metrics["wer"] = wer
    logger.info(f"  [{label}] WER: {wer:.2f} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    semsim = compute_semantic_similarity(clean_preds, clean_refs, max_samples=500)
    if semsim is not None:
        metrics["semsim"] = round(semsim, 2)
    logger.info(f"  [{label}] SemSim: {metrics.get('semsim', 'N/A')} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    verbose = compute_bleu_verbose(clean_preds, clean_refs)
    metrics["sent_bleu_avg"] = verbose.get("sent_bleu_avg", 0.0)
    metrics["sent_bleu_zero"] = verbose.get("zero_bleu_count", 0)
    logger.info(
        f"  [{label}] SentB4 avg={metrics['sent_bleu_avg']:.2f} "
        f"zero={metrics['sent_bleu_zero']}/{n_total} ({time.time() - t0:.1f}s)"
    )

    return metrics


def compute_ps_bleu(predictions: List[str], references: List[str]) -> Dict:
    import collections
    import math

    def _get_ngrams(segment, max_order):
        counts = collections.Counter()
        for order in range(1, max_order + 1):
            for i in range(len(segment) - order + 1):
                counts[tuple(segment[i:i + order])] += 1
        return counts

    ref_tokens = [[r.split()] for r in references]
    pred_tokens = [p.split() for p in predictions]

    result = {}
    for n in range(1, 5):
        matches = [0] * n
        possible = [0] * n
        ref_len = 0
        trans_len = 0
        for refs, trans in zip(ref_tokens, pred_tokens):
            ref_len += min(len(r) for r in refs)
            trans_len += len(trans)
            merged = collections.Counter()
            for r in refs:
                merged |= _get_ngrams(r, n)
            overlap = _get_ngrams(trans, n) & merged
            for ng, c in overlap.items():
                matches[len(ng) - 1] += c
            for o in range(n):
                possible[o] += max(0, len(trans) - o)
        precs = [(matches[i] + 1) / (possible[i] + 1) for i in range(n)]
        if min(precs) <= 0:
            geo = 0.0
        else:
            geo = math.exp(sum(math.log(p) for p in precs) / n)
        bp = 1.0 if trans_len >= ref_len else math.exp(1 - ref_len / trans_len) if trans_len > 0 else 0.0
        result[f"ps_b{n}"] = round(geo * bp * 100, 2)
    return result


def main():
    parser = argparse.ArgumentParser(description="Full evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--split", default="test", choices=["test", "val", "train"])
    parser.add_argument("--csv_dir", default=None, help="CSV directory override")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    global DEVICE
    if args.cpu:
        DEVICE = torch.device("cpu")

    logger.info(f"Device: {DEVICE}")
    model = load_model(args.checkpoint)

    from src.data.dataset import PoseStitchDataset, collate_fn as ds_collate_fn
    csv_dir = args.csv_dir or str(POSE_STITCH_CSV_DIR)
    ds = PoseStitchDataset(args.split, csv_dir=csv_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=ds_collate_fn)
    logger.info(f"PoseStitch {args.split}: {len(ds)} samples, {len(loader)} batches")

    all_preds, all_refs = [], []
    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating"):
            kp = batch["keypoints"].to(DEVICE)
            refs = batch["text"]
            all_refs.extend(refs)
            try:
                gen = model.generate(kp, num_beams=4, repetition_penalty=1.5,
                                     no_repeat_ngram_size=3, do_sample=False)
            except Exception as e:
                logger.warning(f"Generate failed: {e}")
                gen = ["" for _ in range(kp.size(0))]
            all_preds.extend(gen)

    gen_time = time.time() - t0
    logger.info(f"Generated {len(all_preds)} preds in {gen_time:.1f}s ({len(all_preds) / max(1, gen_time):.1f} samp/s)")

    metrics = compute_all_metrics(all_preds, all_refs, label=args.split)

    print(f"\n{'=' * 70}")
    print(f"  ISLiM Evaluation — {args.split}")
    print(f"{'=' * 70}")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:>20s}: {v:.2f}")
        else:
            print(f"  {k:>20s}: {v}")

    print(f"\n  --- Sample Predictions ---")
    valid_pairs = [(p, r) for p, r in zip(all_preds, all_refs) if p.strip() and r.strip()]
    import random
    for i in random.sample(range(len(valid_pairs)), min(5, len(valid_pairs))):
        print(f"    GT: {valid_pairs[i][1][:120]}")
        print(f"    PR: {valid_pairs[i][0][:120]}")
        print()

    out_path = Path(args.checkpoint).parent / f"eval_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
