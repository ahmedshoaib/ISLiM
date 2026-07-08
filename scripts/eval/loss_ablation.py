#!/usr/bin/env python3
"""
Loss function ablation (Table 5).

Runs 5 training configurations from scratch on SpaMo-OF 10K for 10 epochs each:
1. CE only
2. CE + CLIP
3. CE + Semantic
4. CE + Entropy
5. CE + all three

Reports BLEU-4, ROUGE-L, and qualitative behavior for each.

Usage:
    python scripts/eval/loss_ablation.py --split spamof --epochs 10
    python scripts/eval/loss_ablation.py --skip 1
    python scripts/eval/loss_ablation.py --only 3
"""

import sys
import os
import json
import time
import subprocess
import argparse
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import CHECKPOINT_DIR

TRAIN_SCRIPT = str(_PROJECT_ROOT / "scripts" / "train.py")

EXPERIMENTS = [
    {
        "name": "ce_only",
        "desc": "CE baseline",
        "flags": [],
    },
    {
        "name": "ce_clip",
        "desc": "CE + CLIP contrastive",
        "flags": ["--clip_weight", "1.0"],
    },
    {
        "name": "ce_sem",
        "desc": "CE + Semantic similarity",
        "flags": ["--sem_weight", "0.05"],
    },
    {
        "name": "ce_struct",
        "desc": "CE + Entropy regularization",
        "flags": ["--struct_weight", "0.01"],
    },
    {
        "name": "ce_all4",
        "desc": "CE + CLIP + Sem + Entropy",
        "flags": [
            "--clip_weight", "1.0",
            "--sem_weight", "0.05",
            "--struct_weight", "0.01",
        ],
    },
]

BASE_FLAGS = [
    "--epochs", "10",
    "--batch_size", "4",
    "--grad_accum", "4",
    "--cool_down", "0",
    "--label_smoothing", "0",
    "--no_amp",
    "--num_workers", "0",
]


def find_python():
    for candidate in [
        os.path.join(os.path.dirname(sys.executable), "python3"),
        os.path.join(os.path.dirname(sys.executable), "python"),
        sys.executable,
        "python3",
        "python",
    ]:
        if os.path.isfile(candidate):
            return candidate
    return sys.executable


def run_experiment(exp, base_exp_dir):
    exp_name = f"ablation_{exp['name']}"
    cmd = [find_python(), TRAIN_SCRIPT, "--split", "spamof",
           "--exp_name", exp_name] + BASE_FLAGS + exp["flags"]

    log_file = base_exp_dir / f"{exp_name}.log"
    print(f"\n{'=' * 60}")
    print(f"  Ablation: {exp['desc']}")
    print(f"  Exp dir:  {exp_name}")
    print(f"  Command:  {' '.join(cmd)}")
    print(f"  Log:      {log_file}")
    print(f"{'=' * 60}")

    t0 = time.time()
    with open(log_file, "w") as log:
        proc = subprocess.run(
            cmd, cwd=str(_PROJECT_ROOT),
            stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"  FAILED after {elapsed:.0f}s (exit code {proc.returncode})")
        print(f"  Last 10 lines of log:")
        with open(log_file) as f:
            lines = f.readlines()
            for line in lines[-10:]:
                print(f"    {line.rstrip()}")
        return False

    print(f"  OK — completed in {elapsed:.0f}s")
    return True


def load_history(exp_name):
    hist_path = CHECKPOINT_DIR / exp_name / "history.json"
    if not hist_path.exists():
        print(f"  WARNING: No history found at {hist_path}")
        return None
    with open(hist_path) as f:
        return json.load(f)


def analyze_history(history, exp):
    if not history:
        return None

    last = history[-1]
    if last.get("bleu4", 0) == 0 and last.get("bleu1", 0) == 0:
        if all(e.get("bleu4", 0) == 0 for e in history[-3:]):
            return {
                "name": exp["name"],
                "desc": exp["desc"],
                "epochs": len(history),
                "best_epoch": 0,
                "best_bleu1": 0, "best_bleu2": 0, "best_bleu3": 0, "best_bleu4": 0,
                "best_rouge": 0,
                "last_bleu4": 0, "last_loss": 0,
                "unique_pct": 0,
                "trend": "COLLAPSED",
            }

    best = max(history, key=lambda x: x.get("bleu4", 0))
    first_half = history[:len(history) // 2]
    second_half = history[len(history) // 2:]
    avg_first = sum(e.get("bleu4", 0) for e in first_half) / max(1, len(first_half))
    avg_second = sum(e.get("bleu4", 0) for e in second_half) / max(1, len(second_half))
    trending = "up" if avg_second > avg_first + 0.05 else ("down" if avg_second < avg_first - 0.05 else "flat")

    return {
        "name": exp["name"],
        "desc": exp["desc"],
        "epochs": len(history),
        "best_epoch": best["epoch"],
        "best_bleu1": best.get("bleu1", 0), "best_bleu2": best.get("bleu2", 0),
        "best_bleu3": best.get("bleu3", 0), "best_bleu4": best.get("bleu4", 0),
        "best_rouge": best.get("rouge_l", 0),
        "last_bleu4": last.get("bleu4", 0), "last_loss": last.get("train_loss", 0),
        "unique_pct": last.get("n_valid", 0),
        "trend": trending,
    }


def print_table(results):
    valid = [r for r in results if r is not None]
    if not valid:
        print("\nNo successful experiments to compare.")
        return

    print(f"\n{'=' * 95}")
    print(f"  4-Loss Ablation Results (10 epochs each, SpaMo-OF 10K from scratch)")
    print(f"{'=' * 95}")
    hdr = f"  {'Experiment':<22s} {'B1':>5s} {'B2':>5s} {'B3':>5s} {'B4':>5s} {'ROUGE':>6s} {'LastLoss':>8s} {'Trend':>6s}"
    print(hdr)
    print(f"  {'-' * 75}")
    for r in valid:
        if r["trend"] == "COLLAPSED":
            print(f"  {r['desc']:<22s} {'--':>5s} {'--':>5s} {'--':>5s} {'--':>5s} {'--':>6s} {'--':>8s} {'COLLAPSED  (expected - needs anti-collapse loss)':>20s}")
        else:
            print(f"  {r['desc']:<22s} {r['best_bleu1']:>5.2f} {r['best_bleu2']:>5.2f} "
                  f"{r['best_bleu3']:>5.2f} {r['best_bleu4']:>5.2f} {r['best_rouge']:>6.2f} "
                  f"{r['last_loss']:>8.4f} {r['trend']:>6s}")
    print(f"  {'-' * 75}")

    baseline = next((r for r in valid if r["name"] == "ce_only"), None)
    if baseline:
        print(f"\n  Deltas from CE-only baseline (BLEU-4={baseline['best_bleu4']:.2f}):")
        for r in valid:
            if r["name"] == "ce_only":
                continue
            delta = r["best_bleu4"] - baseline["best_bleu4"]
            sign = "+" if delta >= 0 else ""
            print(f"    {r['desc']:<40s} {sign}{delta:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Loss ablation study")
    parser.add_argument("--split", default="spamof", help="Dataset split (spamof)")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per experiment")
    parser.add_argument("--skip", nargs="*", type=int, default=[], help="Experiments to skip (1-5)")
    parser.add_argument("--only", nargs="*", type=int, default=[], help="Only run these experiments (1-5)")
    args = parser.parse_args()

    to_run = []
    for i, exp in enumerate(EXPERIMENTS):
        idx = i + 1
        if args.only and idx not in args.only:
            continue
        if idx in args.skip:
            continue
        to_run.append((idx, exp))

    print(f"Running {len(to_run)} ablation experiments with {args.epochs} epochs each...")

    results = []
    for idx, exp in to_run:
        ok = run_experiment(exp, CHECKPOINT_DIR)

        if not ok:
            results.append(None)
            continue
        history = load_history(f"ablation_{exp['name']}")
        analysis = analyze_history(history, exp)
        results.append(analysis)

    print_table(results)

    summary_path = CHECKPOINT_DIR / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump([r for r in results if r is not None], f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
