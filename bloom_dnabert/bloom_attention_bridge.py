"""Deprecated shim."""

from bloom_seq.bridge import (
    BloomGuidedClassifier,
    BloomGuidedCrossAttention,
    GatedCrossModalFusion,
    MutationAwarePooling,
    PositionalBloomEncoder,
)

__all__ = [
    "BloomGuidedClassifier",
    "PositionalBloomEncoder",
    "BloomGuidedCrossAttention",
    "MutationAwarePooling",
    "GatedCrossModalFusion",
]
