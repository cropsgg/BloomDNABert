"""
Bloom-Enhanced DNABERT for Sickle Cell Variant Classification

A hybrid system combining Bloom filters for fast pathogenic k-mer detection
with DNABERT-2 embeddings for variant classification.
"""

from .bloom_filter import MultiScaleBloomFilter
from .dnabert_wrapper import DNABERTWrapper
from .classifier import HybridClassifier
from .visualizer import AttentionVisualizer

__version__ = "1.0.0"

__all__ = [
    "MultiScaleBloomFilter",
    "DNABERTWrapper",
    "HybridClassifier",
    "AttentionVisualizer",
]
