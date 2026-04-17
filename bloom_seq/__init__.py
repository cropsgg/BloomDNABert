"""
bloom_seq: plugin-oriented framework for biosequence encoders + pattern indexes.

Reference plugins (DNABERT-2, multi-scale Bloom, HBB/ClinVar) ship in ``bloom_seq.plugins``.
The legacy ``bloom_dnabert`` package re-exports the same public API with a deprecation warning.
"""

from bloom_seq.errors import (
    AlphabetMismatchError,
    DataSourceError,
    PluginLoadError,
    PluginNotFoundError,
)
from bloom_seq.registry import (
    alphabet_plugins,
    backbones,
    data_sources,
    list_all_plugins,
    pattern_indexes,
    plausibility,
)

__version__ = "2.1.0"

__all__ = [
    "AlphabetMismatchError",
    "DataSourceError",
    "PluginLoadError",
    "PluginNotFoundError",
    "alphabet_plugins",
    "backbones",
    "data_sources",
    "list_all_plugins",
    "pattern_indexes",
    "plausibility",
]
