"""Tests for pan-gene ClinVar helpers (no FASTA required)."""

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "clinvar_pan",
    _ROOT / "bloom_dnabert" / "clinvar_pan.py",
)
_pan = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_pan)


def test_clinical_significance_to_label():
    assert _pan.clinical_significance_to_label("Pathogenic") == 1
    assert _pan.clinical_significance_to_label("Likely pathogenic") == 1
    assert _pan.clinical_significance_to_label("Benign") == 0
    assert _pan.clinical_significance_to_label("Likely benign") == 0
    assert _pan.clinical_significance_to_label("Uncertain significance") is None
    assert _pan.clinical_significance_to_label("Conflicting classifications of pathogenicity") is None
    assert _pan.clinical_significance_to_label("Pathogenic/Likely benign") is None


def test_normalize_clinvar_chrom():
    assert _pan.normalize_clinvar_chrom("11") == "chr11"
    assert _pan.normalize_clinvar_chrom("chrX") == "chrX"
    assert _pan.normalize_clinvar_chrom("MT") == "chrM"
