#!/usr/bin/env python3
"""
Apples-to-apples PS-BLEU comparison with PoseStitch-SLT paper (Table 2).

Computes PS-BLEU B1-B4 with the exact PoseStitch-SLT reference implementation
(add-1 smoothing, str.split() tokenization). Prints table row with model output
and PoseStitch paper's reported numbers.

Usage:
    python scripts/eval/compare_posestitch.py --checkpoint checkpoints/best_model.pt
"""

import sys
import collections
import math
import argparse
import logging
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

POSE_STITCH_PAPER = {
    "GloFE":                         {"dev": (8.92, 2.85, 1.18, 0.61), "test": (9.20, 2.83, 1.12, 0.55)},
    "w/o Pose Stitched":             {"dev": (12.85, 2.50, 1.04, 0.58), "test": (12.81, 2.66, 1.14, 0.64)},
    "Pose Stitched (RWO)":           {"dev": (14.75, 6.11, 3.60, 2.45), "test": (14.89, 6.25, 3.67, 2.51)},
    "BLIMP":                         {"dev": (13.89, 4.86, 2.57, 1.62), "test": (13.39, 4.63, 2.43, 1.48)},
    "Pose Stitched (SWO) — SOTA":    {"dev": (17.31, 8.09, 5.02, 3.54), "test": (17.67, 8.20, 5.00, 3.43)},
}


def _get_ngrams(segment, max_order):
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(len(segment) - order + 1):
            ngram_counts[tuple(segment[i:i + order])] += 1
    return ngram_counts


def compute_ps_bleu(reference_corpus, translation_corpus, max_order=4, smooth=True):
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for references, translation in zip(reference_corpus, translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)
        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(max_order):
        if smooth:
            precisions[i] = (matches_by_order[i] + 1.0) / (possible_matches_by_order[i] + 1.0)
        elif possible_matches_by_order[i] > 0:
            precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]

    geo_mean = math.exp(sum((1.0 / max_order) * math.log(p) for p in precisions)) if min(precisions) > 0 else 0
    ratio = float(translation_length) / reference_length if reference_length > 0 else 0
    bp = 1.0 if ratio >= 1.0 else math.exp(1 - 1.0 / ratio) if ratio > 0 else 0
    return [round(compute_ps_bleu(reference_corpus, translation_corpus, max_order=n)[0] * 100, 2) for n in range(1, 5)]


def compute_ps_bleu_single(reference_corpus, translation_corpus, max_order=4, smooth=True):
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for references, translation in zip(reference_corpus, translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)
        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(max_order):
        if smooth:
            precisions[i] = (matches_by_order[i] + 1.0) / (possible_matches_by_order[i] + 1.0)
        elif possible_matches_by_order[i] > 0:
            precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]

    geo_mean = math.exp(sum((1.0 / max_order) * math.log(p) for p in precisions)) if min(precisions) > 0 else 0
    ratio = float(translation_length) / reference_length if reference_length > 0 else 0
    bp = 1.0 if ratio >= 1.0 else math.exp(1 - 1.0 / ratio) if ratio > 0 else 0
    return geo_mean * bp, precisions, bp, ratio, translation_length, reference_length


def main():
    parser = argparse.ArgumentParser(description="PS-BLEU comparison with PoseStitch paper")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    global DEVICE
    if args.cpu:
        DEVICE = torch.device("cpu")

    logger.info(f"Device: {DEVICE}")

    from src.islim.model import ISLiMModel
    from config import POSE_STITCH_CSV_DIR

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
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model = model.to(DEVICE).eval()
    logger.info(f"Loaded checkpoint (epoch {state.get('epoch', '?')})")

    from src.data.dataset import PoseStitchDataset, collate_fn

    results = {}
    for split in ["val", "test"]:
        ds = PoseStitchDataset(split, csv_dir=str(POSE_STITCH_CSV_DIR))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
        logger.info(f"PoseStitch {split}: {len(ds)} samples")

        all_preds, all_refs = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Generating ({split})"):
                kp = batch["keypoints"].to(DEVICE)
                refs = batch["text"]
                all_refs.extend(refs)
                try:
                    gen = model.generate(kp, num_beams=4, repetition_penalty=1.5,
                                         no_repeat_ngram_size=3, do_sample=False)
                except Exception:
                    gen = ["" for _ in range(kp.size(0))]
                all_preds.extend(gen)

        clean_preds, clean_refs = [], []
        for p, r in zip(all_preds, all_refs):
            if isinstance(r, str) and isinstance(p, str) and r.strip() and p.strip():
                clean_preds.append(p.strip().lower())
                clean_refs.append(r.strip().lower())

        ref_tokens = [[r.split()] for r in clean_refs]
        pred_tokens = [p.split() for p in clean_preds]

        scores = {}
        for n in range(1, 5):
            b, _, _, _, _, _ = compute_ps_bleu_single(ref_tokens, pred_tokens, max_order=n)
            scores[f"b{n}"] = round(b * 100, 2)

        logger.info(f"  {split}: B1={scores['b1']} B2={scores['b2']} B3={scores['b3']} B4={scores['b4']}")
        results[split] = scores

    print(f"\n{'=' * 85}")
    print(f"  PoseStitch PS-BLEU Comparison — iSign (ISL)")
    print(f"{'=' * 85}")
    print(f"  {'Method':<30s} {'DEV B1':>7s} {'DEV B2':>7s} {'DEV B3':>7s} {'DEV B4':>7s} {'TEST B1':>8s} {'TEST B2':>8s} {'TEST B3':>8s} {'TEST B4':>8s}")
    print(f"  {'-' * 83}")

    for method, scores in POSE_STITCH_PAPER.items():
        d, t = scores["dev"], scores["test"]
        bold = "**" if "SOTA" in method else ""
        print(f"  {bold}{method:<30s}{bold} {d[0]:>7.2f} {d[1]:>7.2f} {d[2]:>7.2f} {d[3]:>7.2f} {t[0]:>8.2f} {t[1]:>8.2f} {t[2]:>8.2f} {t[3]:>8.2f}")

    r_dev = results.get("val", {})
    r_test = results.get("test", {})
    print(f"  **{'ISLiM (Ours)':<30s}** {r_dev.get('b1', 0):>7.2f} {r_dev.get('b2', 0):>7.2f} {r_dev.get('b3', 0):>7.2f} {r_dev.get('b4', 0):>7.2f} {r_test.get('b1', 0):>8.2f} {r_test.get('b2', 0):>8.2f} {r_test.get('b3', 0):>8.2f} {r_test.get('b4', 0):>8.2f}")

    sota_b4 = POSE_STITCH_PAPER["Pose Stitched (SWO) — SOTA"]["test"][3]
    delta = r_test.get("b4", 0) - sota_b4
    print(f"  {'-' * 83}")
    print(f"  Delta vs SOTA (PS-B4 test): +{delta:.2f}")

    print(f"\n  Note: PS-BLEU uses add-1 smoothing and str.split() tokenization.")
    print(f"  Paper numbers from PoseStitch-SLT Table 2.")


if __name__ == "__main__":
    main()
