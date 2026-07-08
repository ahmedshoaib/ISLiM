#!/usr/bin/env python3
"""
Export ISLiM model (encoder + T5 decoder) to TorchScript for mobile.

Produces:
  mobile/models/
    encoder_traced.pt        - visual encoder: (B,4,94,128) -> (B,N,512)
    t5_decoder_step.pt       - single decoder step with KV cache
    t5_shared_embed.pt       - token -> embedding lookup
    t5_lm_head.pt            - hidden -> logits projection
    t5_full_greedy.pt        - combined: encoder_in -> token_ids
    t5_config.json           - T5 config for mobile-side loop
    model_state_dict.pt      - full weight dump as reference

Verification:
  scripts/mobile/verify.py - cosine sim, token match, BLEU on ref set

Usage:
    python scripts/mobile/export_torchscript.py --checkpoint checkpoints/best_model.pt
"""

import sys
import os
import json
from pathlib import Path

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import CHECKPOINT_DIR
from src.islim.encoder import KeypointToVideoEncoderV2
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from torch.utils.mobile_optimizer import optimize_for_mobile

MODELS_DIR = _PROJECT_ROOT / "mobile" / "models"


def save_mobile(module, path, optimize=True):
    path = path.with_suffix(".ptl")
    if optimize:
        try:
            module = optimize_for_mobile(module)
        except Exception as e:
            print(f"    WARNING: optimize_for_mobile failed: {e}")
    module._save_for_lite_interpreter(str(path))
    print(f"    Saved: {path}")
    return path


def get_model(checkpoint_path, device):
    from src.islim.model import ISLiMModel

    config = {
        "encoder": {
            "keypoint_dim": 4, "num_keypoints": 94,
            "temporal_channels": [376, 256, 256],
            "temporal_kernel": 5,
            "transformer_embed_dim": 512,
            "transformer_num_heads": 8,
            "transformer_num_layers": 4,
            "transformer_dropout": 0.1,
            "output_dim": 512, "max_frames": 128,
            "num_query_tokens": 16,
        },
        "t5": {
            "t5_model": "t5-small", "d_model": 512,
            "max_target_len": 128, "label_smoothing": 0.0,
        },
    }
    model = ISLiMModel(config).to(device)
    model.eval()

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    epoch = ckpt.get("epoch", "?")
    val_b4 = ckpt.get("val_results", {}).get("bleu4", "?")
    print(f"  Checkpoint epoch={epoch}  val BLEU-4={val_b4}")

    return model


class T5GreedyScriptable(nn.Module):
    def __init__(self, decoder, shared, lm_head, pad_id, eos_id):
        super().__init__()
        self.decoder = decoder
        self.shared = shared
        self.lm_head = lm_head
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.max_length = 128

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        B = encoder_hidden_states.size(0)
        device = encoder_hidden_states.device

        decoder_input_ids = torch.full((B, 1), self.pad_id, device=device, dtype=torch.long)
        all_ids = decoder_input_ids
        past_key_values = None

        for _ in range(self.max_length):
            embeds = self.shared(decoder_input_ids)
            if past_key_values is not None:
                out = self.decoder(
                    inputs_embeds=embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                out = self.decoder(
                    inputs_embeds=embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    use_cache=True,
                )
            past_key_values = out.past_key_values
            hidden = out.last_hidden_state[:, -1:, :]
            logits = self.lm_head(hidden)
            next_token = logits.argmax(dim=-1)
            all_ids = torch.cat([all_ids, next_token], dim=1)
            decoder_input_ids = next_token
            if next_token.item() == self.eos_id:
                break

        return all_ids


def export_visual_encoder(model, device):
    print("\n-- Exporting Visual Encoder --")
    example = torch.randn(1, 4, 94, 128, device=device)

    class EncoderWrapper(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, keypoints):
            return self.encoder.get_sequence_output(keypoints)

    wrapper = EncoderWrapper(model.visual_encoder).to(device).eval()
    traced = torch.jit.trace(wrapper, example, check_trace=False)
    traced_path = MODELS_DIR / "encoder_traced.pt"
    traced.save(str(traced_path))
    save_mobile(traced, MODELS_DIR / "encoder_traced.pt")
    out = traced(example)
    print(f"  Input:  {tuple(example.shape)}")
    print(f"  Output: {tuple(out.shape)}")
    print(f"  Saved:  {traced_path}")

    with torch.no_grad():
        ref = model.visual_encoder.get_sequence_output(example)
    cos = nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0)
    print(f"  Cosine sim vs original: {cos.item():.6f}")
    assert cos > 0.999, f"Visual encoder trace drift: cos={cos.item():.6f}"
    print(f"  OK: Encoder trace verified")
    return traced


def export_decoder_components(model, device):
    print("\n-- Exporting T5 Decoder Components --")

    B, T, D = 1, 128, 512
    encoder_hidden = torch.randn(B, T, D, device=device)
    input_ids = torch.full((B, 1), model.t5.config.pad_token_id, device=device, dtype=torch.long)

    print("  Exporting t5_shared_embed...")

    class SharedEmbedWrapper(nn.Module):
        def __init__(self, embed):
            super().__init__()
            self.embed = embed

        def forward(self, ids):
            return self.embed(ids)

    shared_wrapper = SharedEmbedWrapper(model.t5.shared).to(device).eval()
    traced_embed = torch.jit.trace(shared_wrapper, input_ids, check_trace=False)
    embed_path = MODELS_DIR / "t5_shared_embed.pt"
    traced_embed.save(str(embed_path))
    save_mobile(traced_embed, embed_path)
    emb_out = traced_embed(input_ids)
    print(f"    Input:  {tuple(input_ids.shape)} -> Output: {tuple(emb_out.shape)}")
    print(f"    Saved:  {embed_path}")

    print("  Exporting t5_encoder...")

    class T5EncoderWrapper(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, hidden_states):
            return self.encoder(inputs_embeds=hidden_states).last_hidden_state

    t5_enc_wrapper = T5EncoderWrapper(model.t5.encoder).to(device).eval()
    traced_t5_enc = torch.jit.trace(t5_enc_wrapper, encoder_hidden, check_trace=False)
    t5_enc_path = MODELS_DIR / "t5_encoder.pt"
    traced_t5_enc.save(str(t5_enc_path))
    save_mobile(traced_t5_enc, t5_enc_path)
    t5_enc_out = traced_t5_enc(encoder_hidden)
    print(f"    Input:  {tuple(encoder_hidden.shape)} -> Output: {tuple(t5_enc_out.shape)}")
    print(f"    Saved:  {t5_enc_path}")

    with torch.no_grad():
        ref_t5_enc = model.t5.encoder(inputs_embeds=encoder_hidden).last_hidden_state
    cos_enc = nn.functional.cosine_similarity(t5_enc_out.flatten(), ref_t5_enc.flatten(), dim=0)
    print(f"    Cosine sim vs original: {cos_enc.item():.6f}")

    print("  Exporting t5_decoder_first_step...")

    class DecoderFirstStep(nn.Module):
        def __init__(self, decoder, shared):
            super().__init__()
            self.decoder = decoder
            self.shared = shared

        def forward(self, encoder_hidden_states, input_ids):
            embeds = self.shared(input_ids)
            out = self.decoder(
                inputs_embeds=embeds,
                encoder_hidden_states=encoder_hidden_states,
                use_cache=True,
            )
            pkv = out.past_key_values
            sac = pkv.self_attention_cache
            cac = pkv.cross_attention_cache
            n = len(sac.layers)
            out_sa_keys = tuple(sac.layers[i].keys for i in range(n))
            out_sa_vals = tuple(sac.layers[i].values for i in range(n))
            out_ca_keys = tuple(cac.layers[i].keys for i in range(n))
            out_ca_vals = tuple(cac.layers[i].values for i in range(n))
            return out.last_hidden_state, out_sa_keys, out_sa_vals, out_ca_keys, out_ca_vals

    first_step = DecoderFirstStep(model.t5.decoder, model.t5.shared).to(device).eval()

    try:
        traced_first = torch.jit.trace(first_step, (encoder_hidden, input_ids), check_trace=False)
        first_path = MODELS_DIR / "t5_decoder_first_step.pt"
        traced_first.save(str(first_path))
        save_mobile(traced_first, first_path)
        print(f"    Saved: {first_path}")
    except Exception as e:
        print(f"    WARNING: First step trace failed: {e}")

    print("  Exporting t5_decoder_step...")
    from transformers.cache_utils import EncoderDecoderCache, DynamicCache

    class DecoderStepRaw(nn.Module):
        def __init__(self, decoder, shared, lm_head):
            super().__init__()
            self.decoder = decoder
            self.shared = shared
            self.lm_head = lm_head

        def forward(self, encoder_hidden_states, input_ids,
                    sa_keys, sa_vals, ca_keys, ca_vals):
            num_layers = len(sa_keys)
            sa_cache = DynamicCache()
            ca_cache = DynamicCache()
            for i in range(num_layers):
                sa_cache.update(sa_keys[i], sa_vals[i], i)
                ca_cache.update(ca_keys[i], ca_vals[i], i)
            pkv = EncoderDecoderCache(sa_cache, ca_cache)

            embeds = self.shared(input_ids)
            out = self.decoder(
                inputs_embeds=embeds,
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=pkv,
                use_cache=True,
            )
            pkv_out = out.past_key_values
            sac = pkv_out.self_attention_cache
            cac = pkv_out.cross_attention_cache
            n = len(sac.layers)
            out_sa_keys = tuple(sac.layers[i].keys for i in range(n))
            out_sa_vals = tuple(sac.layers[i].values for i in range(n))
            out_ca_keys = tuple(cac.layers[i].keys for i in range(n))
            out_ca_vals = tuple(cac.layers[i].values for i in range(n))

            hidden = out.last_hidden_state[:, -1:, :]
            logits = self.lm_head(hidden)
            return logits, out_sa_keys, out_sa_vals, out_ca_keys, out_ca_vals

    step_raw = DecoderStepRaw(model.t5.decoder, model.t5.shared, model.t5.lm_head).to(device).eval()

    with torch.no_grad():
        embeds = model.t5.shared(input_ids)
        first_out = model.t5.decoder(
            inputs_embeds=embeds,
            encoder_hidden_states=encoder_hidden,
            use_cache=True,
        )
    pkv_ref = first_out.past_key_values
    sa_keys = tuple(pkv_ref.self_attention_cache.layers[i].keys for i in range(6))
    sa_vals = tuple(pkv_ref.self_attention_cache.layers[i].values for i in range(6))
    ca_keys = tuple(pkv_ref.cross_attention_cache.layers[i].keys for i in range(6))
    ca_vals = tuple(pkv_ref.cross_attention_cache.layers[i].values for i in range(6))
    next_ids = torch.full((B, 1), 3, device=device, dtype=torch.long)

    try:
        traced_step = torch.jit.trace(
            step_raw,
            (encoder_hidden, next_ids, sa_keys, sa_vals, ca_keys, ca_vals),
            check_trace=False,
        )
        step_path = MODELS_DIR / "t5_decoder_step.pt"
        traced_step.save(str(step_path))
        save_mobile(traced_step, step_path)
        print(f"    Saved: {step_path}")
    except Exception as e:
        print(f"    WARNING: Step trace failed: {e}")

    print("  Exporting t5_lm_head...")
    hidden = torch.randn(B, 1, D, device=device)

    class LMHeadWrapper(nn.Module):
        def __init__(self, lm_head):
            super().__init__()
            self.lm_head = lm_head

        def forward(self, x):
            return self.lm_head(x)

    lm_mod = LMHeadWrapper(model.t5.lm_head).to(device).eval()
    traced_lm = torch.jit.trace(lm_mod, hidden, check_trace=False)
    lm_path = MODELS_DIR / "t5_lm_head.pt"
    traced_lm.save(str(lm_path))
    save_mobile(traced_lm, lm_path)
    print(f"    Input: {tuple(hidden.shape)} -> Output: ({B}, 1, {model.t5.lm_head.out_features})")
    print(f"    Saved: {lm_path}")

    return {"shared_embed": embed_path, "lm_head": lm_path}


def export_full_greedy(model, device):
    print("\n-- Exporting Full Greedy Decoder --")

    greedy_mod = T5GreedyScriptable(
        model.t5.decoder, model.t5.shared, model.t5.lm_head,
        model.t5.config.pad_token_id, model.t5.config.eos_token_id,
    ).to(device).eval()

    try:
        scripted = torch.jit.script(greedy_mod)
        greedy_path = MODELS_DIR / "t5_full_greedy.pt"
        scripted.save(str(greedy_path))
        print(f"  OK: Full greedy script successful")
        print(f"  Saved: {greedy_path}")
        return scripted
    except Exception as e:
        print(f"  WARNING: Full greedy scripting failed: {e}")
        return None


def export_config(model):
    config = {
        "vocab_size": model.t5.config.vocab_size,
        "d_model": model.t5.config.d_model,
        "d_kv": model.t5.config.d_kv,
        "num_layers": model.t5.config.num_layers,
        "num_decoder_layers": model.t5.config.num_decoder_layers,
        "num_heads": model.t5.config.num_heads,
        "d_ff": model.t5.config.d_ff,
        "pad_token_id": model.t5.config.pad_token_id,
        "eos_token_id": model.t5.config.eos_token_id,
        "decoder_start_token_id": model.t5.config.decoder_start_token_id,
        "max_target_len": 128,
    }
    config_path = MODELS_DIR / "t5_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n  Config saved: {config_path}")
    return config


def export_full_state_dict(model):
    sd_path = MODELS_DIR / "model_state_dict.pt"
    torch.save(model.state_dict(), sd_path)
    print(f"  Reference state dict saved: {sd_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export model to TorchScript for mobile")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    print(f"Device: {device} (forced CPU for mobile-compatible export)")
    print(f"Checkpoint: {args.checkpoint}")

    model = get_model(args.checkpoint, device)

    export_full_state_dict(model)
    export_config(model)
    export_visual_encoder(model, device)
    export_decoder_components(model, device)
    export_full_greedy(model, device)

    print("\n-- Export Summary --")
    for f in sorted(MODELS_DIR.iterdir()):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name:30s} {size_mb:8.2f} MB")
    print(f"\n  OK: Export complete. Models in {MODELS_DIR}/")
    print(f"\n  For mobile, use the 6 components + manual loop:")
    print(f"    1. encoder_traced.pt       keypoints -> vis_enc_out")
    print(f"    2. t5_encoder.pt           vis_enc_out -> t5_enc_out")
    print(f"    3. t5_shared_embed.pt      token_id -> embedding")
    print(f"    4. t5_decoder_first_step   1st decoder step -> (hidden, cache)")
    print(f"    5. t5_decoder_step.pt      subsequent steps -> (logits, cache)")
    print(f"    6. t5_lm_head.pt           hidden -> logits")


if __name__ == "__main__":
    main()
