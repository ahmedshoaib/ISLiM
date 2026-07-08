"""
ISLiM Visual Encoder for sign language keypoint sequences.

Two components:
  VisualEncoder — core encoder with optional learnable query tokens
  ISLiMEncoder   — wrapper that handles padding/truncation of variable-length keypoints

When num_query_tokens=0: full-frame mode — Conv1D×2 → 4-layer Transformer → all T tokens
When num_query_tokens=N>0: compressed mode — prepends N query tokens before Transformer,
                            T5 receives only N tokens (~8× compression at N=16)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class VisualEncoder(nn.Module):
    """
    Token-prepend visual encoder for sign language keypoints.

    Architecture:
      1. Linear projection (flattened keypoints → temporal channels)
      2. TemporalConv1D layers (motion features)
      3. Optional learnable query tokens prepended (no positional encoding)
      4. Transformer encoder (queries attend to all frames)
      5. Slice queries off as compressed representation
    """

    def __init__(
        self,
        keypoint_dim=4,
        num_keypoints=94,
        temporal_channels=(376, 256, 256),
        temporal_kernel=5,
        transformer_embed_dim=512,
        transformer_num_heads=8,
        transformer_num_layers=4,
        transformer_dropout=0.1,
        output_dim=512,
        num_query_tokens=16,
    ):
        super().__init__()

        self.keypoint_dim = keypoint_dim
        self.num_keypoints = num_keypoints
        self.transformer_embed_dim = transformer_embed_dim
        self.num_query_tokens = num_query_tokens

        flat_dim = num_keypoints * keypoint_dim
        self.keypoint_proj = nn.Linear(flat_dim, temporal_channels[0])

        self.temporal_convs = nn.ModuleList()
        in_ch = temporal_channels[0]
        for out_ch in temporal_channels[1:]:
            self.temporal_convs.append(
                TemporalConv1D(in_ch, out_ch, temporal_kernel, transformer_dropout)
            )
            in_ch = out_ch

        self.transformer_proj = nn.Linear(temporal_channels[-1], transformer_embed_dim)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, 1000, transformer_embed_dim) * 0.1
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_embed_dim,
            nhead=transformer_num_heads,
            dim_feedforward=transformer_embed_dim * 4,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_num_layers
        )

        if num_query_tokens > 0:
            self.query_tokens = nn.Parameter(
                torch.randn(1, num_query_tokens, transformer_embed_dim) * 0.1
            )

        self.output_proj = nn.Linear(transformer_embed_dim, output_dim)
        self.layer_norm = nn.LayerNorm(transformer_embed_dim)

    def _prepare_input(self, keypoints):
        if keypoints.dim() == 4 and keypoints.shape[1] == 4:
            keypoints = keypoints.permute(0, 3, 2, 1)

        B, T, K, D = keypoints.shape
        x = keypoints.reshape(B, T, -1)
        x = self.keypoint_proj(x)
        x = x.transpose(1, 2)
        for conv in self.temporal_convs:
            x = conv(x)
        x = x.transpose(1, 2)
        x = self.transformer_proj(x)

        if x.shape[1] <= self.pos_embedding.shape[1]:
            x = x + self.pos_embedding[:, : x.shape[1], :]

        return x

    def _prepend_queries(self, x, mask=None):
        B = x.shape[0]
        if self.num_query_tokens > 0:
            queries = self.query_tokens.expand(B, -1, -1)
            x = torch.cat([queries, x], dim=1)
            if mask is not None:
                query_mask = torch.zeros(
                    B, self.num_query_tokens, device=mask.device, dtype=mask.dtype
                )
                mask = torch.cat([query_mask, mask], dim=1)
        return x, mask

    def _encode(self, x, mask=None):
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.layer_norm(x)
        return x

    def forward(self, keypoints, mask=None):
        x = self._prepare_input(keypoints)
        x, mask = self._prepend_queries(x, mask)
        x = self._encode(x, mask)
        if self.num_query_tokens > 0:
            pooled = x[:, : self.num_query_tokens, :].mean(dim=1)
        else:
            pooled = x.mean(dim=1)
        return self.output_proj(pooled)

    def get_sequence_output(self, keypoints, mask=None):
        x = self._prepare_input(keypoints)
        x, mask = self._prepend_queries(x, mask)
        x = self._encode(x, mask)
        if self.num_query_tokens > 0:
            out = x[:, : self.num_query_tokens, :]
        else:
            out = x
        return self.output_proj(out)

    def get_full_output(self, keypoints, mask=None):
        x = self._prepare_input(keypoints)
        x, mask = self._prepend_queries(x, mask)
        x = self._encode(x, mask)
        return self.output_proj(x)


class ISLiMEncoder(nn.Module):
    """
    Wrapper that handles variable-length keypoint sequences:
    permutation, padding/truncation to max_frames, and delegates to VisualEncoder.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_frames = config.get("max_frames", 128)
        self.num_query_tokens = config.get("num_query_tokens", 16)

        self.encoder = VisualEncoder(
            keypoint_dim=config.get("keypoint_dim", 4),
            num_keypoints=config.get("num_keypoints", 94),
            temporal_channels=config.get("temporal_channels", [376, 256, 256]),
            temporal_kernel=config.get("temporal_kernel", 5),
            transformer_embed_dim=config.get("transformer_embed_dim", 512),
            transformer_num_heads=config.get("transformer_num_heads", 8),
            transformer_num_layers=config.get("transformer_num_layers", 4),
            transformer_dropout=config.get("transformer_dropout", 0.1),
            output_dim=config.get("output_dim", 512),
            num_query_tokens=self.num_query_tokens,
        )

    def _normalize_frames(self, keypoints, padding_mask=None):
        if keypoints.dim() == 4 and keypoints.shape[1] == 4:
            keypoints = keypoints.permute(0, 3, 2, 1)

        B, T, K, D = keypoints.shape
        if T > self.max_frames:
            indices = torch.linspace(0, T - 1, self.max_frames).long().to(keypoints.device)
            keypoints = keypoints[:, indices, :, :]
        elif T < self.max_frames:
            pad_size = self.max_frames - T
            padding = torch.zeros(B, pad_size, K, D).to(keypoints.device)
            keypoints = torch.cat([keypoints, padding], dim=1)
            if padding_mask is not None:
                pad_mask = torch.ones(B, pad_size).to(padding_mask.device)
                padding_mask = torch.cat([padding_mask, pad_mask], dim=1)

        return keypoints, padding_mask

    def forward(self, keypoints, padding_mask=None):
        keypoints, padding_mask = self._normalize_frames(keypoints, padding_mask)
        return self.encoder(keypoints, padding_mask)

    def get_sequence_output(self, keypoints, padding_mask=None):
        keypoints, padding_mask = self._normalize_frames(keypoints, padding_mask)
        return self.encoder.get_sequence_output(keypoints, padding_mask)

    def get_full_output(self, keypoints, padding_mask=None):
        keypoints, padding_mask = self._normalize_frames(keypoints, padding_mask)
        return self.encoder.get_full_output(keypoints, padding_mask)
