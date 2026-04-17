"""DNA trinucleotide plausibility prior (hg38-style background)."""

from bloom_seq.alphabets import DNA_ALPHABET
from bloom_seq.plugins.plausibility_dna_trinuc.prior import (
    assess_sequence_genomic_plausibility,
)


class DNATrinucPrior:
    name = "dna_trinuc"
    alphabet = DNA_ALPHABET

    def score(self, sequence: str):
        return assess_sequence_genomic_plausibility(sequence)


__all__ = ["DNATrinucPrior", "assess_sequence_genomic_plausibility"]
