#!/usr/bin/env python3
"""
ISLiM unified training entry point.

Supports both the PoseStitch split and SpaMo-OF 10K datasets.
75M keypoint-to-text SLT model with query-token encoder + T5 decoder.

Usage:
    python scripts/train.py --split posestitch --epochs 200 --losses ce+clip+sem+entropy
    python scripts/train.py --split spamof --epochs 100
    python scripts/train.py --resume --epochs 300
"""

import sys
import os
import json
import time
import random
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import CHECKPOINT_DIR, T5_MODEL_NAME
from src.islim.model import ISLiMModel
from src.data.dataset import (
    PoseStitchDataset,
    SpaMoOFDataset,
    IndexedDataset,
    collate_fn,
)
from src.losses import SemanticLoss, entropy_regularization
from src.metrics import compute_bleu, compute_rouge_l

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="ISLiM training")
    parser.add_argument("--split", type=str, default="posestitch",
                        choices=["posestitch", "spamof"],
                        help="Which dataset to use")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--query_tokens", type=int, default=16,
                        help="Number of learnable query tokens (0 = full-frame)")
    parser.add_argument("--clip_weight", type=float, default=1.0)
    parser.add_argument("--clip_ramp", type=float, default=0.1)
    parser.add_argument("--sem_weight", type=float, default=0.05)
    parser.add_argument("--sem_ramp", type=float, default=0.005)
    parser.add_argument("--struct_weight", type=float, default=0.01)
    parser.add_argument("--cool_down", type=int, default=60,
                        help="Seconds between epochs")
    parser.add_argument("--exp_name", default=None,
                        help="Experiment name (auto-generated if not set)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable mixed precision")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers")
    parser.add_argument("--t5_model", default=T5_MODEL_NAME,
                        help="T5 model name (default: t5-small)")
    args = parser.parse_args()

    if args.exp_name is None:
        args.exp_name = f"islim_{args.split}_{args.epochs}e_{args.lr}lr_{args.query_tokens}qt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    logger.info(
        f"Device: {device}  LR: {args.lr}  Cooldown: {args.cool_down}s  AMP: {use_amp}"
        f"  Sem: {'on' if args.sem_weight > 0 else 'off'}"
        f"  CLIP: {'on' if args.clip_weight > 0 else 'off'}"
        f"  Split: {args.split}  Query tokens: {args.query_tokens}"
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    if args.split == "posestitch":
        train_ds = PoseStitchDataset("train")
        dev_ds = PoseStitchDataset("val")
    else:
        train_ds = SpaMoOFDataset("train")
        dev_ds = SpaMoOFDataset("dev")

    train_ds_id = IndexedDataset(train_ds)
    dev_ds_id = IndexedDataset(dev_ds)

    train_loader = DataLoader(
        train_ds_id, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_ds_id, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )
    logger.info(f"Train: {len(train_ds)}  Dev: {len(dev_ds)}  Batches: {len(train_loader)}")

    config = {
        "encoder": {
            "keypoint_dim": 4, "num_keypoints": 94,
            "temporal_channels": [376, 256, 256], "temporal_kernel": 5,
            "transformer_embed_dim": 512, "transformer_num_heads": 8,
            "transformer_num_layers": 4, "transformer_dropout": args.dropout,
            "output_dim": 512, "max_frames": 128,
            "num_query_tokens": args.query_tokens,
        },
        "t5": {
            "t5_model": args.t5_model, "d_model": 512,
            "max_target_len": 128, "label_smoothing": args.label_smoothing,
        },
    }
    model = ISLiMModel(config).to(device)
    total = sum(p.numel() for p in model.parameters())
    qp = sum(p.numel() for n, p in model.named_parameters() if "query_tokens" in n)
    logger.info(f"ISLiM: {total / 1e6:.1f}M params  (query tokens: {qp}, N={args.query_tokens})")

    sem_loss_fn = None
    if args.sem_weight > 0:
        if args.split == "posestitch":
            texts = train_ds.df["text"].tolist()
        else:
            texts = train_ds.df["text_ps"].tolist()
        sem_loss_fn = SemanticLoss(texts, device)
    logger.info(f"Sem loss: {'enabled (weight=' + str(args.sem_weight) + ')' if sem_loss_fn else 'disabled'}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader) // args.grad_accum
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    exp_dir = CHECKPOINT_DIR / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_bleu4 = 0.0
    history = []
    if args.resume and (exp_dir / "latest.pt").exists():
        ckpt = torch.load(exp_dir / "latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_bleu4 = ckpt.get("best_bleu4", 0.0)
        if (exp_dir / "history.json").exists():
            with open(exp_dir / "history.json") as f:
                history = json.load(f)
        logger.info(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        sem_weight = min(args.sem_weight, epoch * args.sem_ramp) if sem_loss_fn else 0.0
        clip_weight = min(args.clip_weight, epoch * args.clip_ramp) if args.clip_weight > 0 else 0.0
        struct_weight = args.struct_weight
        use_clip = args.clip_weight > 0
        use_sem = sem_loss_fn and sem_weight > 0
        use_struct = struct_weight > 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"E{epoch}")):
            kp = batch["keypoints"].to(device)
            refs = batch["text"]
            if not any(isinstance(r, str) and r.strip() for r in refs):
                continue

            try:
                if use_clip:
                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            ce_loss, clip_loss, hidden, logits = model.forward_with_clip(kp, refs)
                    else:
                        ce_loss, clip_loss, hidden, logits = model.forward_with_clip(kp, refs)
                    loss = ce_loss + clip_weight * clip_loss
                    if use_struct:
                        _, label_mask = model._encode_target(refs, device)
                        ent_loss = entropy_regularization(logits, label_mask[:, :logits.size(1)])
                        loss = loss + struct_weight * ent_loss
                    if use_sem:
                        batch_indices = batch["row_idx"].tolist()
                        sem_loss = sem_loss_fn(hidden, batch_indices)
                        loss = loss + sem_weight * sem_loss
                elif use_sem or use_struct:
                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            ce_loss, hidden, logits = model.forward_with_hidden(kp, refs)
                    else:
                        ce_loss, hidden, logits = model.forward_with_hidden(kp, refs)
                    loss = ce_loss
                    if use_struct:
                        _, label_mask = model._encode_target(refs, device)
                        ent_loss = entropy_regularization(logits, label_mask[:, :logits.size(1)])
                        loss = loss + struct_weight * ent_loss
                    if use_sem:
                        batch_indices = batch["row_idx"].tolist()
                        sem_loss = sem_loss_fn(hidden, batch_indices)
                        loss = loss + sem_weight * sem_loss
                else:
                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            loss = model(kp, refs)
                    else:
                        loss = model(kp, refs)
            except Exception as e:
                logger.warning(f"Fwd fail: {e}")
                continue

            loss = loss / args.grad_accum
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total_loss += loss.item()

            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if batch_idx % 500 == 0:
                clip_info = ""
                if use_clip:
                    d = getattr(model, "_last_clip_diag", None)
                    o = getattr(model, "_last_clip_offdiag", None)
                    if d is not None:
                        clip_info = f"  clip_diag={d:.3f} offdiag={o:.3f}"
                logger.info(f"  E{epoch} batch {batch_idx}: loss={loss.item() * args.grad_accum:.4f}{clip_info}")

        avg_loss = total_loss / max(1, len(train_loader))
        logger.info(f"E{epoch} done: loss={avg_loss:.4f}  time={time.time() - t0:.0f}s  "
                     f"sem={sem_weight:.3f}  clip={clip_weight:.3f}  struct={struct_weight:.4f}")
        time.sleep(10)

        val_res = validate(model, dev_loader, epoch, device)
        val_res["train_loss"] = round(avg_loss, 4)
        history.append(val_res)
        logger.info(
            f"  Val: B1={val_res['bleu1']} B2={val_res['bleu2']} B3={val_res['bleu3']} "
            f"B4={val_res['bleu4']}  rouge={val_res['rouge_l']}"
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_bleu4": best_bleu4,
            "val_results": val_res,
        }
        torch.save(ckpt_data, exp_dir / "latest.pt")
        if val_res["bleu4"] > best_bleu4:
            best_bleu4 = val_res["bleu4"]
            torch.save(ckpt_data, exp_dir / "best_model.pt")
            logger.info(f"  * New best BLEU-4: {best_bleu4:.2f}")
        with open(exp_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if args.cool_down > 0:
            logger.info(f"  Cooling {args.cool_down}s ...")
            time.sleep(args.cool_down)

    logger.info(f"Done. Best BLEU-4: {best_bleu4:.2f}")


@torch.no_grad()
def validate(model, loader, epoch, device):
    model.eval()
    all_preds, all_refs = [], []
    for batch in tqdm(loader, desc=f"Val E{epoch}"):
        kp = batch["keypoints"].to(device)
        refs = batch["text"]
        try:
            gen = model.generate(kp, num_beams=4, repetition_penalty=1.5,
                                 no_repeat_ngram_size=3, do_sample=False)
        except Exception:
            gen = ["" for _ in range(kp.size(0))]
        all_preds.extend(gen)
        all_refs.extend(refs)

    cp, cr = [], []
    for p, r in zip(all_preds, all_refs):
        if isinstance(r, str) and isinstance(p, str) and r.strip() and p.strip():
            cp.append(p)
            cr.append(r)
    if len(cp) >= 3:
        print(f"\n  --- Val E{epoch} Samples ---")
        for idx in random.sample(range(len(cp)), min(3, len(cp))):
            print(f"    GT: {cr[idx][:120]}")
            print(f"    PR: {cp[idx][:120]}")
            print()
    bleu = compute_bleu(cp, cr)
    rouge = compute_rouge_l(cp, cr)
    return {
        "epoch": epoch,
        "bleu1": bleu["bleu1"], "bleu2": bleu["bleu2"],
        "bleu3": bleu["bleu3"], "bleu4": bleu["bleu4"],
        "rouge_l": rouge, "n_valid": len(cp),
    }


if __name__ == "__main__":
    main()
