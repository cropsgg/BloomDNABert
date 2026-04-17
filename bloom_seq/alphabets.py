"""Sequence alphabets for DNA, RNA, and protein."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Pattern


@dataclass(frozen=True)
class Alphabet:
    """Valid symbols (IUPAC-style) and optional ambiguity characters."""

    name: str
    symbols: str
    ambiguity: str = ""

    def normalize(self, seq: str) -> str:
        return seq.upper().strip()

    def strip_ambiguity(self, seq: str) -> str:
        s = self.normalize(seq)
        for ch in self.ambiguity.upper():
            s = s.replace(ch, "")
        return s

    def strict_pattern(self) -> Pattern[str]:
        """Regex for sequences containing only ``symbols`` (no ambiguity)."""
        esc = re.escape(self.symbols.upper())
        return re.compile(rf"^[{esc}]+$")

    def is_kmer_valid(self, kmer: str) -> bool:
        kmer = kmer.upper()
        allowed = set(self.symbols.upper())
        return all(c in allowed for c in kmer)


DNA_ALPHABET = Alphabet("dna", "ACGT", ambiguity="N")
RNA_ALPHABET = Alphabet("rna", "ACGU", ambiguity="N")
PROTEIN_ALPHABET = Alphabet(
    "protein",
    "ACDEFGHIKLMNPQRSTVWY",
    ambiguity="XBUZJO*",
)


def validation_pattern(alphabet: Alphabet, allow_ambiguity: bool = True) -> Pattern[str]:
    """
    Build a regex for user input validation.

    If ``allow_ambiguity``, ambiguity chars are permitted in addition to symbols.
    """
    chars = alphabet.symbols.upper()
    if allow_ambiguity and alphabet.ambiguity:
        chars += alphabet.ambiguity.upper()
    return re.compile(rf"^[{re.escape(chars)}]+$")
