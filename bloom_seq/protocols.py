"""Typed protocols for backbones, pattern indexes, and data sources.

Implementations may be duck-typed; use ``typing.Protocol`` for static checks.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np

from bloom_seq.alphabets import Alphabet

TaskKind = Literal["binary", "multiclass", "regression"]

TokenSpan = Tuple[int, int]
TokenSpans = List[TokenSpan]


@runtime_checkable
class Backbone(Protocol):
    """Sequence encoder (e.g. HuggingFace DNA/RNA/protein LM)."""

    name: str
    hidden_size: int
    num_layers: int
    num_heads: int
    max_length: int
    alphabet: Alphabet
    supports_attention: bool
    supports_hidden_states: bool

    def get_embedding(self, sequence: str, pool_method: str = "mean") -> np.ndarray:
        ...

    def get_attention_weights(self, sequence: str, layer: int = -1) -> np.ndarray:
        ...

    def get_token_level_outputs(self, sequence: str) -> Dict[str, Any]:
        ...


@runtime_checkable
class PatternIndex(Protocol):
    """Fast approximate k-mer / pattern lookup (e.g. multi-scale Bloom)."""

    alphabet: Alphabet
    k_sizes: List[int]
    n_scales: int

    @property
    def feature_dim(self) -> int:
        """Length of ``get_feature_vector`` output."""
        ...

    def get_feature_vector(self, sequence: str) -> np.ndarray:
        ...

    def get_positional_signal(self, sequence: str) -> np.ndarray:
        ...

    def get_token_aligned_signal(
        self, sequence: str, token_spans: TokenSpans
    ) -> np.ndarray:
        ...

    def check_sequence(self, sequence: str) -> Dict[int, List[bool]]:
        ...

    def get_hit_positions(self, sequence: str, k: int = 8) -> List[Tuple[int, str]]:
        ...


@runtime_checkable
class DataSource(Protocol):
    """Labeled sequence data for training."""

    name: str
    alphabet: Alphabet

    def get_training_splits(
        self,
        val_split: float = 0.2,
        test_split: float = 0.2,
        random_state: int = 42,
        use_cache: bool = True,
        balance_majority_ratio: Optional[float] = 4.0,
    ) -> Tuple[Any, Any, Any]:
        """Return (train_df, val_df, test_df) with canonical columns."""
        ...


@runtime_checkable
class PlausibilityScorer(Protocol):
    """Optional sequence prior (e.g. trinucleotide model)."""

    alphabet: Alphabet

    def score(self, sequence: str) -> Dict[str, Any]:
        ...
