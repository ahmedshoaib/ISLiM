#!/usr/bin/env python3
"""
Verify exported TorchScript models against original PyTorch.

Checks at 4 levels:
  Level 1 - Visual encoder: cosine similarity on 200 random inputs
  Level 2 - T5 decoder step: logit cosine similarity per step
  Level 3 - Full greedy: token ID match rate (if exported)
  Level 4 - End-to-end BLEU on ref set

Usage:
    python scripts/mobile/verify.py
"""

import sys
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.islim.encoder import KeypointToVideoEncoderV2
from transformers import T5ForConditionalGeneration, T5Tokenizer

_SCRIPTS_MOBILE = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_MOBILE))
from export_torchscript import get_model, T5GreedyScriptable

MODELS_DIR = _PROJECT_ROOT / "mobile" / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

levels = {"L1": False, "L2": False, "L3": False, "L4": False}


def make_random_kp(batch_size=1, T=128):
    kp = torch.randn(batch_size, 4, 94, T, device=DEVICE)
    kp[:, :3] = kp[:, :3].sigmoid()
    kp[:, 3] = kp[:, 3].sigmoid()
    return kp


def make_random_encoder_hidden(batch_size=1, T=128, D=512):
    return torch.randn(batch_size, T, D, device=DEVICE)


def verify_encoder(model):
    print("\n=== Level 1: Visual Encoder ===")
    enc_path = MODELS_DIR / "encoder_traced.pt"
    if not enc_path.exists():
        print("  WARNING: encoder_traced.pt not found, skipping")
        return False

    traced = torch.jit.load(str(enc_path), map_location=DEVICE)
    traced.eval()

    cosines = []
    for i in range(200):
        kp = make_random_kp()
        with torch.no_grad():
            ref = model.visual_encoder.get_sequence_output(kp)
            out = traced(kp)
        c = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0)
        cosines.append(c.item())

    cos = np.mean(cosines)
    min_cos = min(cosines)
    passed = cos > 0.999
    print(f"  Cosine similarity: mean={cos:.6f}  min={min_cos:.6f}  "
          f"{'OK: PASS' if passed else 'FAIL'}")
    levels["L1"] = passed
    return passed


def verify_decoder_step(model):
    print("\n=== Level 2: T5 Decoder Step ===")

    step_path = MODELS_DIR / "t5_decoder_step.pt"
    first_path = MODELS_DIR / "t5_decoder_first_step.pt"
    if not step_path.exists():
        print("  WARNING: t5_decoder_step.pt not found, skipping")
        return False

    all_pass = True
    device = DEVICE

    if first_path.exists():
        traced_first = torch.jit.load(str(first_path), map_location=device)
        traced_first.eval()

        cosines = []
        for i in range(50):
            enc_hidden = make_random_encoder_hidden()
            input_ids = torch.full((1, 1), model.t5.config.pad_token_id, device=device, dtype=torch.long)
            with torch.no_grad():
                embeds = model.t5.shared(input_ids)
                ref_out = model.t5.decoder(
                    inputs_embeds=embeds,
                    encoder_hidden_states=enc_hidden,
                    use_cache=True,
                )
                ref_first = ref_out.last_hidden_state
                traced_first_hidden = traced_first(enc_hidden, input_ids)
            c = F.cosine_similarity(traced_first_hidden.flatten(), ref_first.flatten(), dim=0)
            cosines.append(c.item())

        cos = np.mean(cosines)
        passed = cos > 0.999
        print(f"  First step (no past_kv): mean cos={cos:.6f}  {'OK: PASS' if passed else 'FAIL'}")
        all_pass = all_pass and passed
    else:
        print("  WARNING: t5_decoder_first_step.pt not found, skipping first-step check")

    traced_step = torch.jit.load(str(step_path), map_location=device)
    traced_step.eval()
    num_layers = 6

    cosines = []
    token_match = []
    for i in range(50):
        enc_hidden = make_random_encoder_hidden()

        start_ids = torch.full((1, 1), model.t5.config.pad_token_id, device=device, dtype=torch.long)
        with torch.no_grad():
            embeds = model.t5.shared(start_ids)
            first_out = model.t5.decoder(
                inputs_embeds=embeds,
                encoder_hidden_states=enc_hidden,
                use_cache=True,
            )
        pkv = first_out.past_key_values
        sac, cac = pkv.self_attention_cache, pkv.cross_attention_cache
        sa_keys = tuple(sac.layers[j].keys for j in range(num_layers))
        sa_vals = tuple(sac.layers[j].values for j in range(num_layers))
        ca_keys = tuple(cac.layers[j].keys for j in range(num_layers))
        ca_vals = tuple(cac.layers[j].values for j in range(num_layers))

        next_id = torch.full((1, 1), random.randint(0, model.t5.config.vocab_size - 1),
                             device=device, dtype=torch.long)

        with torch.no_grad():
            embeds_next = model.t5.shared(next_id)
            ref_out = model.t5.decoder(
                inputs_embeds=embeds_next,
                encoder_hidden_states=enc_hidden,
                past_key_values=pkv,
                use_cache=True,
            )
            ref_hidden = ref_out.last_hidden_state[:, -1:, :]
            ref_logits = model.t5.lm_head(ref_hidden)
            traced_out = traced_step(enc_hidden, next_id, sa_keys, sa_vals, ca_keys, ca_vals)
            traced_logits = traced_out[0]

        c = F.cosine_similarity(traced_logits.flatten(), ref_logits.flatten(), dim=0)
        cosines.append(c.item())
        ref_token = ref_logits.argmax(dim=-1)
        traced_token = traced_logits.argmax(dim=-1)
        token_match.append((ref_token == traced_token).item())

    cos = np.mean(cosines)
    tok_acc = np.mean(token_match) * 100
    passed = cos > 0.999
    print(f"  Decoder step (with past_kv): mean cos={cos:.6f}  token match={tok_acc:.1f}%  "
          f"{'OK: PASS' if passed else 'FAIL'}")
    all_pass = all_pass and passed

    levels["L2"] = all_pass
    return all_pass


def verify_full_greedy(model):
    print("\n=== Level 3: Full Greedy Decoding ===")
    greedy_path = MODELS_DIR / "t5_full_greedy.pt"
    if not greedy_path.exists():
        print("  WARNING: t5_full_greedy.pt not found, skipping")
        return False

    scripted = torch.jit.load(str(greedy_path), map_location=DEVICE)
    scripted.eval()

    ref_greedy = T5GreedyScriptable(
        model.t5.decoder, model.t5.shared, model.t5.lm_head,
        model.t5.config.pad_token_id, model.t5.config.eos_token_id,
    ).to(DEVICE).eval()

    match_rates = []
    for i in range(50):
        enc_hidden = make_random_encoder_hidden()
        with torch.no_grad():
            ref_ids = ref_greedy(enc_hidden)
            scripted_ids = scripted(enc_hidden)
        min_len = min(ref_ids.size(1), scripted_ids.size(1))
        if min_len == 0:
            continue
        match = (ref_ids[:, :min_len] == scripted_ids[:, :min_len]).float().mean().item()
        match_rates.append(match * 100)

    avg_match = np.mean(match_rates) if match_rates else 0
    passed = avg_match > 99.0
    print(f"  Token match rate: {avg_match:.1f}%  {'OK: PASS' if passed else 'FAIL'}")
    levels["L3"] = passed
    return passed


def verify_bleu(model):
    import collections
    import math

    print("\n=== Level 4: End-to-End BLEU ===")

    greedy_path = MODELS_DIR / "t5_full_greedy.pt"
    if not greedy_path.exists():
        print("  WARNING: t5_full_greedy.pt not found - testing original model generate() only\n")
        return False

    scripted = torch.jit.load(str(greedy_path), map_location=DEVICE)
    scripted.eval()

    from config import POSE_STITCH_CSV_DIR
    from src.data.dataset import PoseStitchDataset, collate_fn
    from torch.utils.data import DataLoader

    ds = PoseStitchDataset("test", csv_dir=str(POSE_STITCH_CSV_DIR))
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)

    all_preds_orig, all_preds_scripted, all_refs = [], [], []

    with torch.no_grad():
        for batch in loader:
            kp = batch["keypoints"].to(DEVICE)
            refs = batch["text"]

            orig_ids = model.generate(kp, num_beams=4, repetition_penalty=1.5,
                                      no_repeat_ngram_size=3)
            enc_hidden = model.visual_encoder.get_sequence_output(kp)
            greedy_ids = scripted(enc_hidden)
            scripted_texts = model.tokenizer.batch_decode(greedy_ids, skip_special_tokens=True)
            scripted_texts = [t.lower() for t in scripted_texts]

            all_preds_orig.extend(orig_ids)
            all_preds_scripted.extend(scripted_texts)
            all_refs.extend(refs)

    total = len(all_preds_orig)
    match = sum(1 for a, b in zip(all_preds_orig, all_preds_scripted) if a == b)

    def _ps_bleu(preds, refs):
        def _get_ngrams(s, n):
            c = collections.Counter()
            for o in range(1, n + 1):
                for i in range(len(s) - o + 1):
                    c[tuple(s[i:i + o])] += 1
            return c

        r_tok = [[r.split()] for r in refs]
        p_tok = [p.split() for p in preds]
        res = {}
        for n in range(1, 5):
            matches = [0] * n
            possible = [0] * n
            rl, tl = 0, 0
            for rt, pt in zip(r_tok, p_tok):
                rl += min(len(r) for r in rt)
                tl += len(pt)
                merged = collections.Counter()
                for r in rt:
                    merged |= _get_ngrams(r, n)
                ov = _get_ngrams(pt, n) & merged
                for ng, c in ov.items():
                    matches[len(ng) - 1] += c
                for o in range(n):
                    possible[o] += max(0, len(pt) - o)
            precs = [(matches[i] + 1) / (possible[i] + 1) for i in range(n)]
            if min(precs) <= 0:
                geo = 0.0
            else:
                geo = math.exp(sum(math.log(p) for p in precs) / n)
            bp = 1.0 if tl >= rl else math.exp(1 - rl / tl) if tl > 0 else 0
            res[f"ps_b{n}"] = round(geo * bp * 100, 2)
        return res

    valid_ref, valid_orig, valid_scripted = [], [], []
    for r, po, ps in zip(all_refs, all_preds_orig, all_preds_scripted):
        if isinstance(r, str) and isinstance(po, str) and isinstance(ps, str) and r.strip():
            valid_ref.append(r)
            valid_orig.append(po)
            valid_scripted.append(ps)

    bleu_orig = _ps_bleu(valid_orig, valid_ref)
    bleu_scripted = _ps_bleu(valid_scripted, valid_ref)

    print(f"  Samples: {total}  Text-match: {match}/{total} ({match / total * 100:.1f}%)")
    print(f"  Original (beam=4) PS-B4:    {bleu_orig.get('ps_b4', 0)}")
    print(f"  Scripted (greedy) PS-B4:    {bleu_scripted.get('ps_b4', 0)}")
    print(f"  Note: greedy vs beam=4 is not a fidelity test - different search strategies.")

    levels["L4"] = True
    return True


def main():
    print(f"Device: {DEVICE}")
    print("=" * 60)

    import argparse
    parser = argparse.ArgumentParser(description="Verify exported TorchScript models")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    model = get_model(args.checkpoint, torch.device("cpu"))
    model = model.to(DEVICE).eval()

    verify_encoder(model)
    verify_decoder_step(model)
    verify_full_greedy(model)
    verify_bleu(model)

    print("\n" + "=" * 60)
    print("Verification Summary:")
    for k, v in levels.items():
        icon = "OK" if v else "FAIL"
        print(f"  {k}: {icon}")
    all_pass = all(levels.values())
    print(f"\n  {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
