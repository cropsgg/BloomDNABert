"""
Bloom-Enhanced DNABERT for Sickle Cell Variant Classification

A hybrid system combining Bloom filters for fast pathogenic k-mer detection
with DNABERT-2 embeddings for variant classification.

Includes the novel Bloom-Guided Positional Cross-Attention (BGPCA)
architecture that bridges probabilistic data structures with neural
attention mechanisms for position-aware cross-modal fusion.
"""

from .bloom_filter import MultiScaleBloomFilter
from .dnabert_wrapper import DNABERTWrapper
from .classifier import HybridClassifier
from .visualizer import AttentionVisualizer
from .bloom_attention_bridge import (
    BloomGuidedClassifier,
    PositionalBloomEncoder,
    BloomGuidedCrossAttention,
    MutationAwarePooling,
    GatedCrossModalFusion,
)

__version__ = "2.0.0"

__all__ = [
    "MultiScaleBloomFilter",
    "DNABERTWrapper",
    "HybridClassifier",
    "AttentionVisualizer",
    "BloomGuidedClassifier",
    "PositionalBloomEncoder",
    "BloomGuidedCrossAttention",
    "MutationAwarePooling",
    "GatedCrossModalFusion",
]
