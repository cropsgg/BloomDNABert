"""HBB / ClinVar reference data source (DNA)."""

from __future__ import annotations

from bloom_seq.alphabets import DNA_ALPHABET
from bloom_seq.plugins.clinvar_hbb.source import ClinVarDataLoader


class HBBClinVarSource:
    """Registry-facing wrapper around :class:`ClinVarDataLoader`."""

    name = "clinvar_hbb"
    alphabet = DNA_ALPHABET

    def __init__(self, cache_dir: str = "data", **kwargs):
        self._loader = ClinVarDataLoader(cache_dir=cache_dir, **kwargs)

    def get_training_splits(self, **kwargs):
        return self._loader.get_training_data(**kwargs)

    @staticmethod
    def example_sequences():
        return [
            {
                "label": "Normal HBB (Wild-type)",
                "sequence": (
                    "CACGTGGACTACCCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCC"
                    "CTGGGCAGGTTGGTATCAAGGTTACAAGACAGGTTTAAGGAGACCAATAGAAACTGGGCATGTGGAGACAGAGAAGACTCTT"
                    "GGGTTTCTGATAGGCACTGACTCTCTCTGCCTATTGGTCTATTTTCCCACCCTTAGG"
                ),
            },
            {
                "label": "Sickle Cell (HbS E6V)",
                "sequence": (
                    "CACGTGGTCTACCCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCC"
                    "CTGGGCAGGTTGGTATCAAGGTTACAAGACAGGTTTAAGGAGACCAATAGAAACTGGGCATGTGGAGACAGAGAAGACTCTT"
                    "GGGTTTCTGATAGGCACTGACTCTCTCTGCCTATTGGTCTATTTTCCCACCCTTAGG"
                ),
            },
            {
                "label": "HbC Disease (E6K)",
                "sequence": (
                    "CACGTGAAGTACCCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCC"
                    "CTGGGCAGGTTGGTATCAAGGTTACAAGACAGGTTTAAGGAGACCAATAGAAACTGGGCATGTGGAGACAGAGAAGACTCTT"
                    "GGGTTTCTGATAGGCACTGACTCTCTCTGCCTATTGGTCTATTTTCCCACCCTTAGG"
                ),
            },
            {
                "label": "Random Benign Variant",
                "sequence": (
                    "CACGTGGACTACCCCTGAGGAGAAGTCTGCCGTTACTACCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCC"
                    "CTGGGCAGGTTGGTATCAAGGTTACAAGACAGGTTTAAGGAGACCAATAGAAACTGGGCATGTGGAGACAGAGAAGACTCTT"
                    "GGGTTTCTGATAGGCACTGACTCTCTCTGCCTATTGGTCTATTTTCCCACCCTTAGG"
                ),
            },
        ]


__all__ = ["ClinVarDataLoader", "HBBClinVarSource"]
