"""Tests for BGPCA architecture components."""

import pytest
import torch
import numpy as np
from bloom_dnabert.bloom_attention_bridge import (
    PositionalBloomEncoder,
    BloomGuidedCrossAttention,
    MutationAwarePooling,
    GatedCrossModalFusion,
    BloomGuidedClassifier,
)


BATCH = 2
SEQ_LEN = 16
D_MODEL = 768
D_BLOOM = 64
N_SCALES = 3
N_HEADS = 4
BLOOM_SUMMARY_DIM = 18


@pytest.fixture
def sample_inputs():
    """Create random inputs mimicking real data shapes."""
    return {
        'hidden': torch.randn(BATCH, SEQ_LEN, D_MODEL),
        'bloom_sig': torch.rand(BATCH, SEQ_LEN, N_SCALES),
        'bloom_sum': torch.rand(BATCH, BLOOM_SUMMARY_DIM),
    }


class TestPositionalBloomEncoder:
    def test_output_shape(self):
        encoder = PositionalBloomEncoder(n_scales=N_SCALES, d_bloom=D_BLOOM)
        x = torch.rand(BATCH, SEQ_LEN, N_SCALES)
        out = encoder(x)
        assert out.shape == (BATCH, SEQ_LEN, D_BLOOM)

    def test_gradient_flows(self):
        encoder = PositionalBloomEncoder(n_scales=N_SCALES, d_bloom=D_BLOOM)
        x = torch.rand(BATCH, SEQ_LEN, N_SCALES, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_scale_weights_learnable(self):
        encoder = PositionalBloomEncoder(n_scales=N_SCALES, d_bloom=D_BLOOM)
        assert encoder.scale_weights.requires_grad


class TestBloomGuidedCrossAttention:
    def test_output_shape(self):
        attn = BloomGuidedCrossAttention(
            d_model=D_MODEL, d_bloom=D_BLOOM, n_heads=N_HEADS
        )
        hidden = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        bloom_enc = torch.randn(BATCH, SEQ_LEN, D_BLOOM)
        output, weights = attn(hidden, bloom_enc)
        assert output.shape == (BATCH, SEQ_LEN, D_MODEL)
        assert weights.shape == (BATCH, N_HEADS, SEQ_LEN, SEQ_LEN)

    def test_attention_weights_sum_to_one(self):
        attn = BloomGuidedCrossAttention(
            d_model=D_MODEL, d_bloom=D_BLOOM, n_heads=N_HEADS
        )
        attn.eval()  # Disable dropout so weights stay normalized
        hidden = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        bloom_enc = torch.randn(BATCH, SEQ_LEN, D_BLOOM)
        with torch.no_grad():
            _, weights = attn(hidden, bloom_enc)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_bloom_bias_effect(self):
        """Verify that non-zero Bloom signal changes attention output."""
        attn = BloomGuidedCrossAttention(
            d_model=D_MODEL, d_bloom=D_BLOOM, n_heads=N_HEADS
        )
        attn.eval()
        hidden = torch.randn(1, SEQ_LEN, D_MODEL)

        bloom_zero = torch.zeros(1, SEQ_LEN, D_BLOOM)
        bloom_hot = torch.randn(1, SEQ_LEN, D_BLOOM) * 5.0

        with torch.no_grad():
            out_zero, _ = attn(hidden, bloom_zero)
            out_hot, _ = attn(hidden, bloom_hot)

        # Outputs should differ when Bloom signal differs
        assert not torch.allclose(out_zero, out_hot, atol=1e-3)


class TestMutationAwarePooling:
    def test_output_shape(self):
        pool = MutationAwarePooling(d_model=D_MODEL, d_bloom=D_BLOOM)
        hidden = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        bloom_enc = torch.randn(BATCH, SEQ_LEN, D_BLOOM)
        pooled, importance = pool(hidden, bloom_enc)
        assert pooled.shape == (BATCH, D_MODEL)
        assert importance.shape == (BATCH, SEQ_LEN)

    def test_importance_sums_to_one(self):
        pool = MutationAwarePooling(d_model=D_MODEL, d_bloom=D_BLOOM)
        hidden = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        bloom_enc = torch.randn(BATCH, SEQ_LEN, D_BLOOM)
        _, importance = pool(hidden, bloom_enc)
        sums = importance.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestGatedCrossModalFusion:
    def test_output_shape(self):
        fusion = GatedCrossModalFusion(
            d_model=D_MODEL, bloom_summary_dim=BLOOM_SUMMARY_DIM
        )
        cross_repr = torch.randn(BATCH, D_MODEL)
        bloom_sum = torch.randn(BATCH, BLOOM_SUMMARY_DIM)
        out = fusion(cross_repr, bloom_sum)
        assert out.shape == (BATCH, D_MODEL)


class TestBloomGuidedClassifier:
    @pytest.fixture
    def model(self):
        return BloomGuidedClassifier(
            d_model=D_MODEL,
            d_bloom=D_BLOOM,
            n_bloom_scales=N_SCALES,
            bloom_summary_dim=BLOOM_SUMMARY_DIM,
            n_heads=N_HEADS,
            n_cross_attn_layers=2,
            dropout=0.2
        )

    def test_forward_shape(self, model, sample_inputs):
        logits = model(
            sample_inputs['hidden'],
            sample_inputs['bloom_sig'],
            sample_inputs['bloom_sum']
        )
        assert logits.shape == (BATCH, 1)

    def test_forward_with_interpretability(self, model, sample_inputs):
        result = model.forward_with_interpretability(
            sample_inputs['hidden'],
            sample_inputs['bloom_sig'],
            sample_inputs['bloom_sum']
        )
        assert 'logits' in result
        assert 'cross_attn_weights' in result
        assert 'position_importance' in result
        assert 'gate_values' in result
        assert result['logits'].shape == (BATCH, 1)
        assert len(result['cross_attn_weights']) == 2

    def test_mc_dropout_uncertainty(self, model, sample_inputs):
        mean_pred, uncertainty = model.predict_with_uncertainty(
            sample_inputs['hidden'],
            sample_inputs['bloom_sig'],
            sample_inputs['bloom_sum'],
            n_samples=5
        )
        assert mean_pred.shape == (BATCH, 1)
        assert uncertainty.shape == (BATCH, 1)
        assert torch.all(mean_pred >= 0) and torch.all(mean_pred <= 1)
        assert torch.all(uncertainty >= 0)

    def test_gradient_flow_end_to_end(self, model, sample_inputs):
        """Verify gradients flow through entire BGPCA architecture."""
        hidden = sample_inputs['hidden'].requires_grad_(True)
        logits = model(hidden, sample_inputs['bloom_sig'], sample_inputs['bloom_sum'])
        loss = logits.sum()
        loss.backward()
        assert hidden.grad is not None
        assert hidden.grad.abs().sum() > 0

    def test_save_load_roundtrip(self, model, sample_inputs, tmp_path):
        """Verify model can be saved and loaded with identical outputs."""
        model.eval()
        with torch.no_grad():
            original_out = model(
                sample_inputs['hidden'],
                sample_inputs['bloom_sig'],
                sample_inputs['bloom_sum']
            )

        # Build config the same way as BloomGuidedPipeline.save()
        config = {
            'd_model': model.d_model,
            'd_bloom': model.d_bloom,
            'n_bloom_scales': model.bloom_encoder.n_scales,
            'bloom_summary_dim': model.bloom_summary_dim,
            'n_heads': model.cross_attn_layers[0].n_heads,
            'n_cross_attn_layers': model.n_cross_attn_layers,
            'dropout': model.classifier[2].p,
        }

        path = tmp_path / "model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': config,
            'architecture': 'BGPCA'
        }, path)

        model2 = BloomGuidedClassifier(**config)
        state = torch.load(path, weights_only=False)
        model2.load_state_dict(state['model_state_dict'])
        model2.eval()

        with torch.no_grad():
            loaded_out = model2(
                sample_inputs['hidden'],
                sample_inputs['bloom_sig'],
                sample_inputs['bloom_sum']
            )

        assert torch.allclose(original_out, loaded_out, atol=1e-6)
