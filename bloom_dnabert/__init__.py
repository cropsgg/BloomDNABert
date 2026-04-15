import importlib
from typing import Any, List

__version__ = "2.0.0"

_LAZY = {
    "MultiScaleBloomFilter": (".bloom_filter", "MultiScaleBloomFilter"),
    "DNABERTWrapper": (".dnabert_wrapper", "DNABERTWrapper"),
    "HybridClassifier": (".classifier", "HybridClassifier"),
    "AttentionVisualizer": (".visualizer", "AttentionVisualizer"),
    "load_settings": (".settings", "load_settings"),
    "BloomGuidedClassifier": (".bloom_attention_bridge", "BloomGuidedClassifier"),
    "PositionalBloomEncoder": (".bloom_attention_bridge", "PositionalBloomEncoder"),
    "BloomGuidedCrossAttention": (".bloom_attention_bridge", "BloomGuidedCrossAttention"),
    "MutationAwarePooling": (".bloom_attention_bridge", "MutationAwarePooling"),
    "GatedCrossModalFusion": (".bloom_attention_bridge", "GatedCrossModalFusion"),
}

__all__: List[str] = [
    "load_settings",
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


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        mod_path, attr = _LAZY[name]
        mod = importlib.import_module(mod_path, __name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__))
