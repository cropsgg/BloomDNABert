"""Core types and alphabet helpers (no PyTorch required)."""

from bloom_seq.alphabets import DNA_ALPHABET, PROTEIN_ALPHABET, validation_pattern
from bloom_seq.splits import sanitize_labeled_frame
import pandas as pd


def test_dna_alphabet_kmer():
    assert DNA_ALPHABET.is_kmer_valid("ACGT")
    assert not DNA_ALPHABET.is_kmer_valid("ACGX")


def test_protein_alphabet_kmer():
    assert PROTEIN_ALPHABET.is_kmer_valid("MKT")


def test_validation_pattern_accepts_ambiguity():
    pat = validation_pattern(DNA_ALPHABET, allow_ambiguity=True)
    assert pat.match("ACGTN")


def test_sanitize_labeled_frame_dna():
    df = pd.DataFrame(
        {
            "sequence": ["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT", "XX"],
            "label": [1, 0],
        }
    )
    out = sanitize_labeled_frame(df, DNA_ALPHABET, min_length=40)
    assert len(out) == 1
    assert out.iloc[0]["label"] == 1
