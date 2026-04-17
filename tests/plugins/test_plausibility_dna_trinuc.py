"""Tests for statistical sequence plausibility vs human k-mer background."""

import importlib.util
import random
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "sequence_plausibility",
    _ROOT / "bloom_seq" / "plugins" / "plausibility_dna_trinuc" / "prior.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
assess_sequence_genomic_plausibility = _mod.assess_sequence_genomic_plausibility


def test_realistic_window_more_plausible_than_random():
    random.seed(42)
    rnd = "".join(random.choice("ACGT") for _ in range(400))
    real = (
        "TTCCTGTGAGGTAGCCTCGGGCAGCCAGGACTTCGGCCACGGTCACCTCCTCCAGGGGCAGGCCCCCCCGGGAGCTGGGCACCAGGACGCCAGGGTACATCCCCATCCGAACCGGGAGCCGGCCGGTCAGGAGGGCGGCCCTGCGGGACAAGTCACAGAGTCCCTGAGACAGACAGAAATGTGGCCTTCCCTAGAGAGAGA"
    )
    a = assess_sequence_genomic_plausibility(real)
    b = assess_sequence_genomic_plausibility(rnd)
    assert a["genomic_plausibility_score"] > b["genomic_plausibility_score"]
    assert b["probability_statistically_spurious"] > 0.5


def test_homopolymer_low_plausibility():
    out = assess_sequence_genomic_plausibility("A" * 200)
    assert out["base_diversity_score"] == 0.0
    assert out["genomic_plausibility_score"] == 0.0
    assert out["probability_statistically_spurious"] == 1.0


def test_too_short_trinucs_returns_unreliable():
    out = assess_sequence_genomic_plausibility("AAAA")
    assert out["trinucleotides_scored"] < 3
    assert out["reliability_note"]
    assert out["genomic_plausibility_score"] == 0.0
