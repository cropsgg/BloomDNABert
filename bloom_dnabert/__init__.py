"""Deprecated compatibility shim — use ``bloom_seq`` instead."""

from __future__ import annotations

import warnings

warnings.warn(
    "Package `bloom_dnabert` is deprecated; import from `bloom_seq` and use "
    "entry-point plugins (see PLUGINS.md). This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from bloom_seq.bridge import (
    BloomGuidedClassifier,
    BloomGuidedCrossAttention,
    GatedCrossModalFusion,
    MutationAwarePooling,
    PositionalBloomEncoder,
)
from bloom_seq.errors import DataSourceError
from bloom_seq.pipeline import (
    BloomGuidedPipeline,
    HybridClassifier,
    HybridClassifierPipeline,
)
from bloom_seq.plugins.dnabert2.wrapper import DNABERTWrapper
from bloom_seq.plugins.multiscale_bloom import MultiScaleBloomFilter
from bloom_seq.viz import AttentionVisualizer

__version__ = "2.1.0"

__all__ = [
    "MultiScaleBloomFilter",
    "DataSourceError",
    "DNABERTWrapper",
    "HybridClassifier",
    "HybridClassifierPipeline",
    "BloomGuidedPipeline",
    "AttentionVisualizer",
    "BloomGuidedClassifier",
    "PositionalBloomEncoder",
    "BloomGuidedCrossAttention",
    "MutationAwarePooling",
    "GatedCrossModalFusion",
]
