"""Tests for ClinVarDataLoader and data quality."""

from pathlib import Path

import pytest

from bloom_dnabert.data_loader import ClinVarDataLoader
from bloom_dnabert.settings import load_settings


@pytest.fixture
def app_settings(tmp_path):
    root = Path(__file__).resolve().parent.parent
    s = load_settings(root / "config" / "default.yaml")
    return s.model_copy(
        update={
            "data": s.data.model_copy(
                update={"cache_dir": tmp_path.resolve()},
            )
        }
    )


@pytest.fixture
def data_loader(app_settings):
    return ClinVarDataLoader(app_settings)


class TestReferenceSequence:
    def test_hbb_reference_starts_with_atg(self, data_loader):
        assert data_loader.HBB_REFERENCE[:3] == "ATG"

    def test_sickle_cell_codon_is_gag(self, data_loader):
        ref = data_loader.HBB_REFERENCE
        codon = ref[18:21]
        assert codon == "GAG", f"Codon 7 should be GAG, got {codon}"

    def test_sickle_cell_position_is_adenine(self, data_loader):
        assert data_loader.HBB_REFERENCE[19] == "A"

    def test_hbc_position_is_guanine(self, data_loader):
        assert data_loader.HBB_REFERENCE[18] == "G"


class TestSynonymousChecking:
    def test_known_synonymous_change(self, data_loader):
        assert data_loader._is_synonymous(5, "A")
        assert data_loader._is_synonymous(5, "C")
        assert data_loader._is_synonymous(5, "T")

    def test_known_nonsynonymous_change(self, data_loader):
        assert not data_loader._is_synonymous(19, "T")

    def test_intronic_position_is_not_coding(self, data_loader):
        assert data_loader._is_synonymous(100, "A")


class TestDatasetGeneration:
    def test_generates_all_variant_types(self, data_loader):
        df = data_loader._generate_synthetic_dataset()
        variant_types = set(df["variant_type"].unique())
        assert "HbS" in variant_types
        assert "HbC" in variant_types
        assert "HbE" in variant_types

    def test_all_sequences_unique(self, data_loader):
        df = data_loader._generate_synthetic_dataset()
        labeled = df[df["label"] != -1]
        unique_ratio = labeled["sequence"].nunique() / len(labeled)
        assert unique_ratio > 0.9, f"Only {unique_ratio:.1%} of sequences are unique"

    def test_sickle_cell_mutation_present(self, data_loader):
        df = data_loader._generate_synthetic_dataset()
        hbs = df[df["variant_type"] == "HbS"]
        assert len(hbs) > 0
        assert all(hbs["alt"] == "T")
        assert all(hbs["position"] == 19)

    def test_hbc_mutation_at_correct_position(self, data_loader):
        df = data_loader._generate_synthetic_dataset()
        hbc = df[df["variant_type"] == "HbC"]
        assert len(hbc) > 0
        assert all(hbc["position"] == 18)
        assert all(hbc["alt"] == "A")

    def test_benign_variants_are_synonymous(self, data_loader):
        df = data_loader._generate_synthetic_dataset()
        benign_syn = df[df["variant_type"] == "Benign_synonymous"]
        for _, row in benign_syn.iterrows():
            pos = row["position"]
            alt = row["alt"]
            if pos < data_loader.EXON1_END:
                assert data_loader._is_synonymous(pos, alt), (
                    f"Benign variant at {pos} ({alt}) is not synonymous"
                )

    def test_version_matches_config(self, data_loader):
        df = data_loader._generate_synthetic_dataset()
        v = data_loader.settings.data.cache_format_version
        assert all(df["version"] == v)


class TestClinVarParsing:
    def test_parse_pathogenic_record(self, data_loader):
        doc = {
            "uid": "12345",
            "title": "NM_000518.5(HBB):c.20A>T (p.Glu7Val)",
            "clinical_significance": {"description": "Pathogenic"},
            "variation_set": [{"cdna_change": "c.20A>T"}],
            "protein_change": "p.Glu7Val",
        }
        result = data_loader._parse_clinvar_record(doc)
        assert result is not None
        assert result["label"] == 1
        assert result["variant_type"] == "ClinVar_pathogenic"
        assert result["hgvs_c"] == "c.20A>T"

    def test_parse_benign_record(self, data_loader):
        doc = {
            "uid": "67890",
            "title": "NM_000518.5(HBB):c.9T>C (p.His3His)",
            "clinical_significance": {"description": "Benign"},
            "variation_set": [{"cdna_change": "c.9T>C"}],
            "protein_change": "p.His3His",
        }
        result = data_loader._parse_clinvar_record(doc)
        assert result is not None
        assert result["label"] == 0
        assert result["variant_type"] == "ClinVar_benign"

    def test_skip_conflicting_significance(self, data_loader):
        doc = {
            "uid": "99999",
            "title": "Some variant",
            "clinical_significance": {
                "description": "Conflicting interpretations of pathogenicity"
            },
            "variation_set": [],
            "protein_change": "",
        }
        result = data_loader._parse_clinvar_record(doc)
        assert result is None

    def test_parse_malformed_record_returns_none(self, data_loader):
        result = data_loader._parse_clinvar_record({})
        assert result is None


class TestTrainValTestSplit:
    def test_returns_three_splits(self, data_loader):
        train_df, val_df, test_df = data_loader.get_training_data()
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0

    def test_no_vus_in_splits(self, data_loader):
        train_df, val_df, test_df = data_loader.get_training_data()
        for split in [train_df, val_df, test_df]:
            assert all(split["label"] != -1)

    def test_no_sequence_leakage(self, data_loader):
        train_df, val_df, test_df = data_loader.get_training_data()
        train_seqs = set(train_df["sequence"])
        val_seqs = set(val_df["sequence"])
        test_seqs = set(test_df["sequence"])
        assert len(train_seqs & val_seqs) == 0, "Train/val overlap detected"
        assert len(train_seqs & test_seqs) == 0, "Train/test overlap detected"
        assert len(val_seqs & test_seqs) == 0, "Val/test overlap detected"

    def test_stratified_split(self, data_loader):
        train_df, val_df, test_df = data_loader.get_training_data()
        for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
            labels = set(split["label"].unique())
            assert 0 in labels, f"{name} split missing benign examples"
            assert 1 in labels, f"{name} split missing pathogenic examples"
