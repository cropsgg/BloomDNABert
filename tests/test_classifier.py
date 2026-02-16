"""Tests for HybridClassifier and pipeline input validation."""

import pytest
import torch
import numpy as np
from bloom_dnabert.classifier import HybridClassifier


class TestHybridClassifier:
    @pytest.fixture
    def model(self):
        return HybridClassifier(dnabert_dim=768, bloom_dim=18)

    def test_forward_shape(self, model):
        dnabert = torch.randn(4, 768)
        bloom = torch.randn(4, 18)
        out = model(dnabert, bloom)
        assert out.shape == (4, 1)

    def test_predict_proba_range(self, model):
        dnabert = torch.randn(4, 768)
        bloom = torch.randn(4, 18)
        probs = model.predict_proba(dnabert, bloom)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)

    def test_uses_layernorm_not_batchnorm(self, model):
        """Verify Phase 2.9 fix: LayerNorm instead of BatchNorm1d."""
        assert hasattr(model, 'ln1')
        assert hasattr(model, 'ln2')
        assert isinstance(model.ln1, torch.nn.LayerNorm)
        assert isinstance(model.ln2, torch.nn.LayerNorm)
        assert not hasattr(model, 'bn1')
        assert not hasattr(model, 'bn2')

    def test_single_sample_inference(self, model):
        """LayerNorm should work with batch_size=1 (BatchNorm would fail in eval)."""
        model.eval()
        dnabert = torch.randn(1, 768)
        bloom = torch.randn(1, 18)
        out = model(dnabert, bloom)
        assert out.shape == (1, 1)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self, model):
        dnabert = torch.randn(2, 768, requires_grad=True)
        bloom = torch.randn(2, 18, requires_grad=True)
        out = model(dnabert, bloom)
        out.sum().backward()
        assert dnabert.grad is not None
        assert bloom.grad is not None


class TestBaselineSaveLoad:
    def test_save_includes_all_params(self):
        """Verify fix: save() stores hidden_dim and dropout."""
        from bloom_dnabert.classifier import HybridClassifierPipeline
        pipeline = HybridClassifierPipeline()
        pipeline.model = HybridClassifier(
            dnabert_dim=768, bloom_dim=18, hidden_dim=128, dropout=0.5
        )
        pipeline.trained = True

        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
        try:
            pipeline.save(path)
            save_dict = torch.load(path, weights_only=False)
            config = save_dict['model_config']
            assert config['hidden_dim'] == 128
            assert config['dropout'] == 0.5
            assert 'architecture' in save_dict
        finally:
            os.unlink(path)


class TestInputValidation:
    """Test input validation on pipeline classes (without full ML components)."""

    def test_too_short_sequence(self):
        from bloom_dnabert.classifier import HybridClassifierPipeline
        pipeline = HybridClassifierPipeline()
        pipeline.trained = True
        pipeline.model = HybridClassifier()
        with pytest.raises(ValueError, match="too short"):
            pipeline.predict("ATCG")

    def test_too_long_sequence(self):
        from bloom_dnabert.classifier import HybridClassifierPipeline
        pipeline = HybridClassifierPipeline()
        pipeline.trained = True
        pipeline.model = HybridClassifier()
        long_seq = "A" * 6000
        with pytest.raises(ValueError, match="too long"):
            pipeline.predict(long_seq)

    def test_invalid_characters(self):
        from bloom_dnabert.classifier import HybridClassifierPipeline
        pipeline = HybridClassifierPipeline()
        pipeline.trained = True
        pipeline.model = HybridClassifier()
        with pytest.raises(ValueError, match="Invalid characters"):
            pipeline.predict("ATCGATCGXYZ123")

    def test_empty_sequence(self):
        from bloom_dnabert.classifier import HybridClassifierPipeline
        pipeline = HybridClassifierPipeline()
        pipeline.trained = True
        pipeline.model = HybridClassifier()
        with pytest.raises(ValueError, match="non-empty"):
            pipeline.predict("")

    def test_valid_sequence_accepted(self):
        from bloom_dnabert.classifier import HybridClassifierPipeline
        pipeline = HybridClassifierPipeline()
        seq = pipeline._validate_sequence("atcgatcgatcgatcg")
        assert seq == "ATCGATCGATCGATCG"

    def test_n_characters_accepted(self):
        from bloom_dnabert.classifier import HybridClassifierPipeline
        pipeline = HybridClassifierPipeline()
        seq = pipeline._validate_sequence("ATCGNNNNATCGATCG")
        assert "N" in seq
