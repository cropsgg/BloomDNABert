"""
Bloom-Guided Positional Cross-Attention (BGPCA) Module

A novel architecture that bridges probabilistic data structures (Bloom filters)
with neural attention mechanisms (DNABERT-2 transformer) through position-aware
cross-modal attention.

Key Innovation:
    Instead of collapsing Bloom filter hits into summary statistics (losing
    positional information) and pooling DNABERT hidden states into a single
    vector (losing token-level context), BGPCA preserves spatial correspondence
    between both modalities and uses Bloom signals as ATTENTION BIASES in a
    cross-attention mechanism.

    This creates a "knowledge bridge" where:
    - The Bloom filter's O(1) pathogenic pattern detection tells the model
      WHERE to look (positional prior)
    - DNABERT's contextual representations tell the model WHAT is there
      (semantic understanding)
    - Cross-attention learns HOW to combine these complementary signals

Architecture:
    1. PositionalBloomEncoder: Multi-scale 1D CNN encodes raw Bloom signals
       into dense per-position embeddings, capturing local hit patterns.

    2. BloomGuidedCrossAttention: Cross-attention where Bloom positional
       encodings serve as Keys (what to attend to), DNABERT hidden states
       serve as Queries and Values (what to extract), and Bloom activation
       magnitudes create additive attention biases (structural prior).

    3. MutationAwarePooling: Instead of mean pooling, uses the Bloom signal
       to learn position-wise importance weights, naturally focusing on
       mutation-relevant regions.

    4. GatedCrossModalFusion: Learnable sigmoid gate that dynamically
       balances trust between Bloom pattern matching (high confidence on
       known variants) and DNABERT contextual understanding (generalizes
       to novel variants).

    5. Monte Carlo Dropout: Enables epistemic uncertainty estimation by
       running multiple stochastic forward passes at inference time.

References:
    - This architecture is novel. No prior work uses Bloom filter outputs
      as attention biases in transformer cross-attention mechanisms.
    - Related concepts: ALiBi (Press et al., 2022) for learned attention
      biases; Cross-attention in Perceiver (Jaegle et al., 2021).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class PositionalBloomEncoder(nn.Module):
    """
    Transforms raw per-position Bloom filter activation signals into
    dense embeddings using multi-scale 1D convolutions.

    The raw Bloom signal at each position is a 3-dimensional vector
    (one per k-mer scale: k=6, 8, 10) indicating the fraction of
    overlapping k-mers that match known pathogenic patterns.

    Multi-scale convolutions capture local patterns in the hit landscape
    (e.g., clusters of hits indicating a mutation hotspot vs isolated
    false positives).

    Input:  [batch, seq_len, n_scales]  (raw Bloom activation)
    Output: [batch, seq_len, d_bloom]   (dense positional embedding)
    """

    def __init__(
        self,
        n_scales: int = 3,
        d_bloom: int = 64,
        kernel_sizes: list = None,
        dropout: float = 0.1
    ):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]

        self.n_scales = n_scales
        self.d_bloom = d_bloom
        channels_per_branch = d_bloom // len(kernel_sizes)
        remainder = d_bloom - channels_per_branch * len(kernel_sizes)

        self.conv_layers = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            out_ch = channels_per_branch + (1 if i < remainder else 0)
            self.conv_layers.append(
                nn.Conv1d(n_scales, out_ch, k, padding=k // 2)
            )

        self.norm = nn.LayerNorm(d_bloom)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Learned scale importance weights
        self.scale_weights = nn.Parameter(torch.ones(n_scales))

    def forward(self, bloom_signal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bloom_signal: [batch, seq_len, n_scales]
        Returns:
            [batch, seq_len, d_bloom]
        """
        weighted = bloom_signal * F.softmax(self.scale_weights, dim=0)

        # Conv1d expects [batch, channels, length]
        x = weighted.transpose(1, 2)
        conv_outputs = [conv(x) for conv in self.conv_layers]

        # Handle potential length mismatches from convolution
        min_len = min(out.size(2) for out in conv_outputs)
        conv_outputs = [out[:, :, :min_len] for out in conv_outputs]

        x = torch.cat(conv_outputs, dim=1)  # [batch, d_bloom, seq_len]
        x = x.transpose(1, 2)               # [batch, seq_len, d_bloom]

        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x


class BloomGuidedCrossAttention(nn.Module):
    """
    Novel cross-attention mechanism where Bloom filter positional encodings
    serve as Keys and create additive attention biases, guiding the
    transformer's attention toward positions with known pathogenic patterns.

    This is the core architectural novelty of BGPCA:

        Standard cross-attention:
            Attn(Q, K, V) = softmax(QK^T / sqrt(d)) V

        Bloom-guided cross-attention:
            Attn(Q, K, V; B) = softmax(QK^T / sqrt(d) + phi(B)) V

    where phi(B) is a learned projection of the Bloom positional encoding
    that creates per-head, per-position attention biases. Positions with
    strong Bloom activation receive higher bias, naturally drawing the
    model's attention to potential mutation sites.

    Q comes from DNABERT hidden states (what to ask about)
    K comes from Bloom encodings (where to look)
    V comes from DNABERT hidden states (what to extract)
    B creates the structural prior (what we already know)
    """

    def __init__(
        self,
        d_model: int = 768,
        d_bloom: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.d_bloom = d_bloom
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_bloom, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Bloom attention bias: projects Bloom encoding to per-head scalar biases
        self.bloom_bias_proj = nn.Sequential(
            nn.Linear(d_bloom, n_heads * 2),
            nn.GELU(),
            nn.Linear(n_heads * 2, n_heads)
        )

        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        dnabert_hidden: torch.Tensor,
        bloom_encoding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            dnabert_hidden:  [batch, seq_len, d_model]
            bloom_encoding:  [batch, seq_len, d_bloom]
        Returns:
            output:       [batch, seq_len, d_model]
            attn_weights: [batch, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = dnabert_hidden.shape

        Q = self.W_q(dnabert_hidden)
        K = self.W_k(bloom_encoding)
        V = self.W_v(dnabert_hidden)

        # Reshape for multi-head: [batch, n_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # NOVEL: Bloom-derived attention bias
        # Each position's Bloom activation magnitude biases how much
        # attention flows TO that position across all query positions
        bloom_bias = self.bloom_bias_proj(bloom_encoding)   # [batch, seq_len, n_heads]
        bloom_bias = bloom_bias.permute(0, 2, 1).unsqueeze(2)  # [batch, n_heads, 1, seq_len]
        attn_scores = attn_scores + bloom_bias

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.W_o(context)
        output = self.dropout(output)
        output = self.norm(output + dnabert_hidden)

        return output, attn_weights


class MutationAwarePooling(nn.Module):
    """
    Learns position-wise importance weights guided by Bloom activation,
    then applies weighted aggregation to produce a fixed-size representation.

    Unlike mean pooling (which treats all positions equally) or CLS pooling
    (which relies on a single token), this module learns to focus on
    positions most relevant to pathogenicity assessment. The Bloom signal
    provides a strong inductive bias: positions with known pathogenic
    k-mer hits should contribute more to the final representation.
    """

    def __init__(self, d_model: int = 768, d_bloom: int = 64):
        super().__init__()

        self.importance_net = nn.Sequential(
            nn.Linear(d_model + d_bloom, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        bloom_encoding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states:  [batch, seq_len, d_model]
            bloom_encoding: [batch, seq_len, d_bloom]
        Returns:
            pooled: [batch, d_model]
            importance_weights: [batch, seq_len] (for interpretability)
        """
        combined = torch.cat([hidden_states, bloom_encoding], dim=-1)
        importance = self.importance_net(combined).squeeze(-1)  # [batch, seq_len]
        importance_weights = F.softmax(importance, dim=-1)

        pooled = torch.bmm(
            importance_weights.unsqueeze(1),
            hidden_states
        ).squeeze(1)

        return pooled, importance_weights


class GatedCrossModalFusion(nn.Module):
    """
    Dynamically balances contributions from the Bloom-guided cross-attention
    path and the original Bloom summary features.

    When the Bloom filter has strong, unambiguous signal (many k-mer hits
    clustered around a known mutation site), the gate learns to trust the
    pattern-matching path. When the signal is weak or ambiguous, the gate
    shifts weight to the DNABERT contextual representation, which can
    generalize to novel variants not in the Bloom filter.
    """

    def __init__(self, d_model: int = 768, bloom_summary_dim: int = 18):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(d_model + bloom_summary_dim, 256),
            nn.GELU(),
            nn.Linear(256, d_model),
            nn.Sigmoid()
        )

        self.bloom_proj = nn.Sequential(
            nn.Linear(bloom_summary_dim, 128),
            nn.GELU(),
            nn.Linear(128, d_model)
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        cross_attn_repr: torch.Tensor,
        bloom_summary: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            cross_attn_repr: [batch, d_model] from cross-attention + pooling
            bloom_summary:   [batch, bloom_summary_dim] flat Bloom features
        Returns:
            fused: [batch, d_model]
        """
        gate_input = torch.cat([cross_attn_repr, bloom_summary], dim=-1)
        g = self.gate(gate_input)

        bloom_proj = self.bloom_proj(bloom_summary)

        fused = g * cross_attn_repr + (1 - g) * bloom_proj
        return self.norm(fused)


class BloomGuidedClassifier(nn.Module):
    """
    Complete Bloom-Guided Positional Cross-Attention (BGPCA) classifier.

    End-to-end architecture:
        DNA Sequence
            |
            v
        [Bloom Filter] ---> Per-position signal [seq_len, 3]
            |                      |
            v                      v
        [Summary 18-dim]    [PositionalBloomEncoder]
            |                      |
            |               [seq_len, d_bloom]
            |                      |
            |     [DNABERT-2] ---> Per-token hidden [seq_len, 768]
            |           |                    |
            |           v                    v
            |    [BloomGuidedCrossAttention] x N layers
            |                    |
            |           [seq_len, 768]
            |                    |
            |       [MutationAwarePooling]
            |                    |
            |              [768-dim]
            |                    |
            v                    v
          [GatedCrossModalFusion]
                    |
              [768-dim fused]
                    |
           [Classification Head]
                    |
                [logit]
    """

    def __init__(
        self,
        d_model: int = 768,
        d_bloom: int = 64,
        n_bloom_scales: int = 3,
        bloom_summary_dim: int = 18,
        n_heads: int = 4,
        n_cross_attn_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        self.d_model = d_model
        self.d_bloom = d_bloom
        self.bloom_summary_dim = bloom_summary_dim
        self.n_cross_attn_layers = n_cross_attn_layers

        self.bloom_encoder = PositionalBloomEncoder(
            n_scales=n_bloom_scales,
            d_bloom=d_bloom,
            dropout=dropout
        )

        self.cross_attn_layers = nn.ModuleList([
            BloomGuidedCrossAttention(
                d_model=d_model,
                d_bloom=d_bloom,
                n_heads=n_heads,
                dropout=dropout
            )
            for _ in range(n_cross_attn_layers)
        ])

        self.pooling = MutationAwarePooling(d_model=d_model, d_bloom=d_bloom)

        self.fusion = GatedCrossModalFusion(
            d_model=d_model,
            bloom_summary_dim=bloom_summary_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        dnabert_hidden_states: torch.Tensor,
        bloom_positional_signal: torch.Tensor,
        bloom_summary_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            dnabert_hidden_states:   [batch, seq_len, d_model]
            bloom_positional_signal: [batch, seq_len, n_scales]
            bloom_summary_features:  [batch, bloom_summary_dim]
        Returns:
            logits: [batch, 1]
        """
        bloom_encoding = self.bloom_encoder(bloom_positional_signal)

        hidden = dnabert_hidden_states
        all_attn_weights = []
        for cross_attn in self.cross_attn_layers:
            hidden, attn_weights = cross_attn(hidden, bloom_encoding)
            all_attn_weights.append(attn_weights)

        pooled, importance_weights = self.pooling(hidden, bloom_encoding)

        fused = self.fusion(pooled, bloom_summary_features)

        logits = self.classifier(fused)

        return logits

    def forward_with_interpretability(
        self,
        dnabert_hidden_states: torch.Tensor,
        bloom_positional_signal: torch.Tensor,
        bloom_summary_features: torch.Tensor
    ) -> dict:
        """
        Forward pass that also returns interpretability signals.

        Returns dict with:
            - logits: classification logits
            - cross_attn_weights: attention weights from each BGCA layer
            - position_importance: learned position importance weights
            - gate_values: fusion gate activations
        """
        bloom_encoding = self.bloom_encoder(bloom_positional_signal)

        hidden = dnabert_hidden_states
        all_attn_weights = []
        for cross_attn in self.cross_attn_layers:
            hidden, attn_weights = cross_attn(hidden, bloom_encoding)
            all_attn_weights.append(attn_weights.detach())

        pooled, importance_weights = self.pooling(hidden, bloom_encoding)

        # Capture gate values
        gate_input = torch.cat([pooled, bloom_summary_features], dim=-1)
        gate_values = self.fusion.gate(gate_input).detach()

        fused = self.fusion(pooled, bloom_summary_features)
        logits = self.classifier(fused)

        return {
            'logits': logits,
            'cross_attn_weights': all_attn_weights,
            'position_importance': importance_weights.detach(),
            'gate_values': gate_values,
            'bloom_encoding': bloom_encoding.detach()
        }

    def predict_with_uncertainty(
        self,
        dnabert_hidden_states: torch.Tensor,
        bloom_positional_signal: torch.Tensor,
        bloom_summary_features: torch.Tensor,
        n_samples: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo dropout for epistemic uncertainty estimation.

        Runs multiple stochastic forward passes with dropout enabled
        and computes mean prediction and standard deviation.

        High uncertainty indicates the model is unsure, which is
        critical for clinical applications where false confidence
        is dangerous.

        Returns:
            mean_prediction: [batch, 1] mean probability
            uncertainty:     [batch, 1] standard deviation
        """
        was_training = self.training
        self.train()

        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                logits = self.forward(
                    dnabert_hidden_states,
                    bloom_positional_signal,
                    bloom_summary_features
                )
                prob = torch.sigmoid(logits)
                predictions.append(prob)

        if not was_training:
            self.eval()

        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)

        return mean_pred, uncertainty
