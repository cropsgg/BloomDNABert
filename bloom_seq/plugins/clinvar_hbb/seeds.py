"""Preset pathogenic k-mer contexts for HBB (reference DNA plugin)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bloom_seq.plugins.multiscale_bloom.index import MultiScaleBloomFilter

PATHOGENIC_SEQUENCES = [
    "CATCTGACTCCTGTGGAGAAGTCTGCC",
    "ACTCCTGTGGAG",
    "CCTGTG",
    "CCTGTGGA",
    "CCTGTGGAGA",
    "TGACTCCTGTGGAGAA",
    "CATCTGACTCCTAAGGAGAAGTCTGCC",
    "ACTCCTAAGGAG",
    "CCTAAG",
    "CCTAAGGA",
    "CCTAAGGAGA",
    "GCCCTGGGCAAGTTGGTATCAAGGTTACAAG",
]


def apply_hbb_pathogenic_preset(index: "MultiScaleBloomFilter") -> None:
    """Insert curated HBB mutant contexts into a multi-scale Bloom index."""
    for seq in PATHOGENIC_SEQUENCES:
        index.add_pathogenic_kmers(seq)
    print("Loaded pathogenic variants into Bloom filters:")
    for k in index.k_sizes:
        print(f"  k={k}: {index.pathogenic_kmer_count[k]} k-mers added")
