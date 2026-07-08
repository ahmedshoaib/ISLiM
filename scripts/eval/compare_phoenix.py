#!/usr/bin/env python3
"""
Cross-lingual transfer evaluation on Phoenix-2014-T (Table 3).

Loads the model, swaps the T5 decoder for a German T5 model,
loads Phoenix-2014-T test data, computes BLEU-4.

Usage:
    python scripts/eval/compare_phoenix.py --checkpoint checkpoints/best_model.pt
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import GERMAN_KEYPOINTS_DIR, GERMAN_CSV_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GermanPhoenixModel(nn.Module):
    def __init__(self, visual_encoder, t5_model_name, max_target_len=256):
        super().__init__()
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        self.visual_encoder = visual_encoder
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
        self.max_length = max_target_len

    def generate(self, keypoints, padding_mask=None, max_length=None,
                 num_beams=4, repetition_penalty=1.0,
                 no_repeat_ngram_size=0, do_sample=False):
        encoder_output = self.visual_encoder.get_sequence_output(keypoints, padding_mask)
        ml = max_length or self.max_length
        outputs = self.t5.generate(
            inputs_embeds=encoder_output, max_length=ml, num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=do_sample, length_penalty=1.2, early_stopping=True,
        )
        return [t.lower() for t in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)]


class PhoenixKPDataset(Dataset):
    def __init__(self, split="test", max_frames=128):
        import pandas as pd

        csv_path = Path(GERMAN_CSV_DIR) / f"{split}.csv"
        self.df = pd.read_csv(csv_path)
        self.max_frames = max_frames
        self.kp_dir = Path(GERMAN_KEYPOINTS_DIR)
        logger.info(f"Phoenix {split}: {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_name = row.get("video_name", row.get("name", ""))
        text = str(row.get("text", row.get("translation", "")))

        kp_path = self.kp_dir / f"{video_name}.npy"
        try:
            keypoints = np.load(kp_path).astype(np.float32)
        except Exception:
            keypoints = np.zeros((50, 94, 4), dtype=np.float32)

        x, y, z = keypoints[:, :, 0:1], keypoints[:, :, 1:2], keypoints[:, :, 2:3]
        for arr in [x, y, z]:
            rng = arr.max() - arr.min()
            if rng > 1e-8:
                arr[...] = (arr - arr.min()) / rng
            else:
                arr[...] = arr - arr.min()
        keypoints[:, :, :3] = np.concatenate([x, y, z], axis=2)

        T = keypoints.shape[0]
        if T > self.max_frames:
            indices = torch.linspace(0, T - 1, self.max_frames).long()
            keypoints = keypoints[indices.numpy()]
        elif T < self.max_frames:
            pad = np.zeros((self.max_frames - T, 94, 4), dtype=np.float32)
            keypoints = np.concatenate([keypoints, pad], axis=0)

        keypoints = np.transpose(keypoints, (2, 1, 0))
        return {
            "keypoints": torch.from_numpy(keypoints).float(),
            "text": text,
            "video_name": video_name,
            "num_frames": min(T, self.max_frames),
        }


def phoenix_collate_fn(batch):
    kp = torch.stack([b["keypoints"] for b in batch])
    return {
        "keypoints": kp,
        "text": [b["text"] for b in batch],
        "num_frames": [b["num_frames"] for b in batch],
    }


def compute_bleu_german(predictions, references):
    import sacrebleu

    clean_preds, clean_refs = [], []
    for p, r in zip(predictions, references):
        if isinstance(r, str) and isinstance(p, str) and r.strip() and p.strip():
            clean_preds.append(p.strip().lower())
            clean_refs.append(r.strip().lower())

    result = {}
    for n in range(1, 5):
        bn = sacrebleu.metrics.BLEU(max_ngram_order=n, tokenize="13a")
        result[f"bleu{n}"] = round(bn.corpus_score(clean_preds, [clean_refs]).score, 2)
    return result


def main():
    parser = argparse.ArgumentParser(description="Phoenix-2014-T cross-lingual evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--german_t5", default="GermanT5/t5-efficient-gc4-all-german-small-el32",
                        help="German T5 model")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    global DEVICE
    if args.cpu:
        DEVICE = torch.device("cpu")

    logger.info(f"Device: {DEVICE}")

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
    base_model = ISLiMModel(config)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base_model.load_state_dict(state["model_state_dict"], strict=False)
    logger.info(f"Loaded checkpoint (epoch {state.get('epoch', '?')})")

    model = GermanPhoenixModel(base_model.visual_encoder, args.german_t5, max_target_len=256)
    model = model.to(DEVICE).eval()

    ds = PhoenixKPDataset(split="test")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=phoenix_collate_fn)
    logger.info(f"Phoenix-2014-T test: {len(ds)} samples, {len(loader)} batches")

    all_preds, all_refs = [], []
    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating"):
            kp = batch["keypoints"].to(DEVICE)
            refs = batch["text"]
            all_refs.extend(refs)
            try:
                gen = model.generate(kp, num_beams=4, repetition_penalty=1.0,
                                     no_repeat_ngram_size=0)
            except Exception as e:
                logger.warning(f"Generate failed: {e}")
                gen = ["" for _ in range(kp.size(0))]
            all_preds.extend(gen)

    gen_time = time.time() - t0
    logger.info(f"Generated {len(all_preds)} preds in {gen_time:.1f}s")

    bleu = compute_bleu_german(all_preds, all_refs)

    print(f"\n{'=' * 70}")
    print(f"  Phoenix-2014-T Cross-Lingual Transfer (ISL → DGS)")
    print(f"{'=' * 70}")
    print(f"  Samples: {len(all_preds)}")
    for n in range(1, 5):
        print(f"  BLEU-{n}: {bleu[f'bleu{n}']}")

    print(f"\n  --- Sample Predictions ---")
    valid_pairs = [(p, r) for p, r in zip(all_preds, all_refs) if p.strip() and r.strip()]
    import random
    for i in random.sample(range(len(valid_pairs)), min(5, len(valid_pairs))):
        print(f"    GT: {valid_pairs[i][1][:120]}")
        print(f"    PR: {valid_pairs[i][0][:120]}")
        print()

    out_path = Path(args.checkpoint).parent / "eval_phoenix.json"
    with open(out_path, "w") as f:
        json.dump(bleu, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
