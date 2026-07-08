#!/usr/bin/env python3
"""
Prepare test data for mobile/Android testing.

Outputs:
  mobile/android_test/app/src/main/assets/test_data/
    keypoints.pt   - torch tensor (N, 4, 94, 128)
    references.txt - one reference per line (lowercased)
    vocab.json     - T5 tokenizer vocab for decoding

Usage:
    python scripts/mobile/prepare_data.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import POSE_STITCH_CSV_DIR
from src.data.dataset import PoseStitchDataset

ASSETS = _PROJECT_ROOT / "mobile" / "android_test" / "app" / "src" / "main" / "assets" / "test_data"
ASSETS.mkdir(parents=True, exist_ok=True)

ds = PoseStitchDataset("test", csv_dir=str(POSE_STITCH_CSV_DIR))
print(f"Loaded {len(ds)} test samples")

all_kp = []
all_refs = []
for i in range(len(ds)):
    item = ds[i]
    kp = item["keypoints"]
    all_kp.append(kp)
    all_refs.append(item["text"].lower())

tensor = torch.stack(all_kp)
torch.save(tensor, str(ASSETS / "keypoints.pt"))
print(f"Saved keypoints: {tensor.shape}")

with open(ASSETS / "references.txt", "w") as f:
    for ref in all_refs:
        f.write(ref + "\n")
print(f"Saved {len(all_refs)} references")

from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
vocab = {id: token for token, id in tokenizer.get_vocab().items()}
vocab[tokenizer.pad_token_id] = "<pad>"
vocab[tokenizer.eos_token_id] = "</s>"
vocab_sorted = {str(k): v for k, v in sorted(vocab.items())}
with open(ASSETS / "vocab.json", "w") as f:
    json.dump(vocab_sorted, f)
print(f"Saved vocab: {len(vocab_sorted)} tokens")
print(f"Done. Files in {ASSETS}")
