"""
ISLiM Model — visual encoder + T5-small decoder for sign language translation.

Architecture:
  ISLiMEncoder with N query tokens → T5 decoder.
  N=0: full-frame mode (128 tokens to T5).  N=16: compressed (16 tokens to T5).

Training supports four loss modes:
  forward()          → CE only, returns scalar loss
  forward_with_hidden() → CE + hidden states + logits
  forward_with_clip()   → CE + CLIP contrastive + hidden states + logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer
import inspect


class ISLiMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        from src.islim.encoder import ISLiMEncoder

        visual_cfg = config["encoder"].copy()
        visual_cfg["output_dim"] = config["t5"]["d_model"]
        self.visual_encoder = ISLiMEncoder(visual_cfg)
        self.t5 = T5ForConditionalGeneration.from_pretrained(config["t5"]["t5_model"])
        self.tokenizer = T5Tokenizer.from_pretrained(config["t5"]["t5_model"])
        self.max_length = config["t5"].get("max_target_len", 128)
        self.label_smoothing = config["t5"].get("label_smoothing", 0.0)
        t5_params = set(inspect.signature(self.t5.forward).parameters.keys())
        self._supports_ls = "label_smoothing" in t5_params or "kwargs" in t5_params
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

        self.num_query_tokens = visual_cfg.get("num_query_tokens", 16)

    def _encode_target(self, target_texts, device):
        labels = self.tokenizer(
            target_texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ids = labels.input_ids.to(device)[:, : self.max_length]
        mask = labels.attention_mask.to(device)[:, : self.max_length]
        return ids, mask

    def _t5_forward(self, encoder_output, label_ids, label_mask,
                    output_hidden_states=False):
        label_ids_ce = label_ids.masked_fill(label_mask == 0, -100)
        kw = {
            "inputs_embeds": encoder_output,
            "labels": label_ids_ce,
        }
        if output_hidden_states:
            kw["output_hidden_states"] = True
        if self._supports_ls and self.label_smoothing > 0:
            kw["label_smoothing"] = self.label_smoothing
        return self.t5(**kw)

    def forward(self, keypoints, target_texts, padding_mask=None):
        encoder_output = self.visual_encoder.get_sequence_output(keypoints, padding_mask)
        label_ids, label_mask = self._encode_target(target_texts, keypoints.device)
        outputs = self._t5_forward(encoder_output, label_ids, label_mask)
        return outputs.loss

    def forward_with_hidden(self, keypoints, target_texts, padding_mask=None):
        encoder_output = self.visual_encoder.get_sequence_output(keypoints, padding_mask)
        label_ids, label_mask = self._encode_target(target_texts, keypoints.device)
        outputs = self._t5_forward(encoder_output, label_ids, label_mask,
                                   output_hidden_states=True)
        return outputs.loss, outputs.decoder_hidden_states, outputs.logits

    def forward_with_clip(self, keypoints, target_texts, padding_mask=None):
        encoder_output = self.visual_encoder.get_sequence_output(keypoints, padding_mask)
        label_ids, label_mask = self._encode_target(target_texts, keypoints.device)
        outputs = self._t5_forward(encoder_output, label_ids, label_mask,
                                   output_hidden_states=True)
        ce_loss = outputs.loss

        device = keypoints.device
        visual_pooled = encoder_output.mean(dim=1).float()
        visual_pooled = F.normalize(visual_pooled, dim=-1)

        with torch.no_grad():
            text_ids = label_ids.clone()
            text_ids[label_mask == 0] = self.tokenizer.pad_token_id
        text_embeds = self.t5.shared(text_ids).float()
        text_mask_3d = label_mask.unsqueeze(-1).float().to(device)
        text_pooled = (text_embeds * text_mask_3d).sum(dim=1) / text_mask_3d.sum(dim=1).clamp(min=1)
        text_pooled = F.normalize(text_pooled, dim=-1)

        sim = visual_pooled @ text_pooled.T
        logits = sim * self.logit_scale.exp()
        labels_clip = torch.arange(len(logits), device=device)
        clip_loss = (F.cross_entropy(logits, labels_clip) +
                     F.cross_entropy(logits.T, labels_clip)) / 2

        d = sim.diag()
        off_diag_mask = 1 - torch.eye(len(sim), device=device)
        self._last_clip_diag = d.detach().mean().item()
        self._last_clip_offdiag = (sim * off_diag_mask).abs().detach().mean().item()

        return ce_loss, clip_loss, outputs.decoder_hidden_states, outputs.logits

    def generate(self, keypoints, padding_mask=None, max_length=None,
                 num_beams=4, repetition_penalty=1.5,
                 no_repeat_ngram_size=3, do_sample=False):
        encoder_output = self.visual_encoder.get_sequence_output(keypoints, padding_mask)
        ml = max_length or self.max_length
        outputs = self.t5.generate(
            inputs_embeds=encoder_output,
            max_length=ml,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=do_sample,
            length_penalty=1.2,
            early_stopping=True,
        )
        return [t.lower() for t in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)]
