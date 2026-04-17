"""DNABERT-2 backbone plugin (HuggingFace)."""

from __future__ import annotations

from bloom_seq.alphabets import DNA_ALPHABET
from bloom_seq.plugins.dnabert2.wrapper import DNABERTWrapper


class DNABERT2Backbone(DNABERTWrapper):
    """Same as :class:`DNABERTWrapper` with framework metadata for the registry."""

    name = "dnabert2"
    alphabet = DNA_ALPHABET
    supports_attention = True
    supports_hidden_states = True
