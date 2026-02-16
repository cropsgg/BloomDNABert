"""Tests for ClinVarDataLoader and data quality."""

import pytest
import numpy as np
import pandas as pd
from bloom_dnabert.data_loader import ClinVarDataLoader, CODON_TABLE


@pytest.fixture
def data_loader(tmp_path):
    return ClinVarDataLoader(cache_dir=str(tmp_path), random_seed=42)


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
        # Third position of codon 2 (GTG at pos 3-5): G at pos 5
        # GTG -> GTA/GTC/GTT all encode Val
        assert data_loader._is_synonymous(5, 'A')  # GTG -> GTA = Val
        assert data_loader._is_synonymous(5, 'C')  # GTG -> GTC = Val
        assert data_loader._is_synonymous(5, 'T')  # GTG -> GTT = Val

    def test_known_nonsynonymous_change(self, data_loader):
        # Position 19 (A in GAG): A -> T = GAG -> GTG = Glu -> Val (sickle cell)
        assert not data_loader._is_synonymous(19, 'T')

    def test_intronic_position_is_not_coding(self, data_loader):
        # Positions beyond exon 1 are intronic
        assert data_loader._is_synonymous(100, 'A')


class TestDatasetGeneration:
    def test_generates_all_variant_types(self, data_loader):
        df = data_loader._generate_synthetic_dataset()
        variant_types = set(df['variant_type'].unique())
        assert 'HbS' in variant_types
        assert 'HbC' in variant_types
        assert 'HbE' in variant_types

    def test_all_sequences_unique(self, data_loader):
        df = data_loader._generate_synthetic_dataset()
        labeled = df[df['label'] != -1]
        # Most sequences should be unique (background SNPs add diversity)
        unique_ratio = labeled['sequence'].nunique() / len(labeled)
        assert unique_ratio > 0.9, f"Only {unique_ratio:.1%} of sequences are unique"

    def test_sickle_cell_mutation_present(self, data_loader):
        df = data_loader._generate_synthetic_dataset()
        hbs = df[df['variant_type'] == 'HbS']
        assert len(hbs) > 0
        assert all(hbs['alt'] == 'T')
        assert all(hbs['position'] == 19)

    def test_hbc_mutation_at_correct_position(self, data_loader):
        """Verify Phase 1 fix: HbC at position 18, not 19."""
        df = data_loader._generate_synthetic_dataset()
        hbc = df[df['variant_type'] == 'HbC']
        assert len(hbc) > 0
        assert all(hbc['position'] == 18)
        assert all(hbc['alt'] == 'A')

    def test_benign_variants_are_synonymous(self, data_loader):
        df = data_loader._generate_synthetic_dataset()
        benign_syn = df[df['variant_type'] == 'Benign_synonymous']
        for _, row in benign_syn.iterrows():
            pos = row['position']
            alt = row['alt']
            if pos < data_loader.EXON1_END:
                assert data_loader._is_synonymous(pos, alt), (
                    f"Benign variant at {pos} ({alt}) is not synonymous"
                )

    def test_version_2_marker(self, data_loader):
        df = data_loader._generate_synthetic_dataset()
        assert all(df['version'] == 2)


class TestClinVarParsing:
    def test_parse_pathogenic_record(self, data_loader):
        """ClinVar parser should extract pathogenic variants."""
        doc = {
            'uid': '12345',
            'title': 'NM_000518.5(HBB):c.20A>T (p.Glu7Val)',
            'clinical_significance': {'description': 'Pathogenic'},
            'variation_set': [{'cdna_change': 'c.20A>T'}],
            'protein_change': 'p.Glu7Val',
        }
        result = data_loader._parse_clinvar_record(doc)
        assert result is not None
        assert result['label'] == 1
        assert result['variant_type'] == 'ClinVar_pathogenic'
        assert result['hgvs_c'] == 'c.20A>T'

    def test_parse_benign_record(self, data_loader):
        """ClinVar parser should extract benign variants."""
        doc = {
            'uid': '67890',
            'title': 'NM_000518.5(HBB):c.9T>C (p.His3His)',
            'clinical_significance': {'description': 'Benign'},
            'variation_set': [{'cdna_change': 'c.9T>C'}],
            'protein_change': 'p.His3His',
        }
        result = data_loader._parse_clinvar_record(doc)
        assert result is not None
        assert result['label'] == 0
        assert result['variant_type'] == 'ClinVar_benign'

    def test_skip_conflicting_significance(self, data_loader):
        """Should skip variants with conflicting interpretations."""
        doc = {
            'uid': '99999',
            'title': 'Some variant',
            'clinical_significance': {'description': 'Conflicting interpretations of pathogenicity'},
            'variation_set': [],
            'protein_change': '',
        }
        result = data_loader._parse_clinvar_record(doc)
        assert result is None

    def test_parse_malformed_record_returns_none(self, data_loader):
        """Malformed records should return None, not crash."""
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
            assert all(split['label'] != -1)

    def test_no_sequence_leakage(self, data_loader):
        train_df, val_df, test_df = data_loader.get_training_data()
        train_seqs = set(train_df['sequence'])
        val_seqs = set(val_df['sequence'])
        test_seqs = set(test_df['sequence'])
        assert len(train_seqs & val_seqs) == 0, "Train/val overlap detected"
        assert len(train_seqs & test_seqs) == 0, "Train/test overlap detected"
        assert len(val_seqs & test_seqs) == 0, "Val/test overlap detected"

    def test_stratified_split(self, data_loader):
        train_df, val_df, test_df = data_loader.get_training_data()
        # Each split should have both pathogenic and benign
        for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
            labels = set(split['label'].unique())
            assert 0 in labels, f"{name} split missing benign examples"
            assert 1 in labels, f"{name} split missing pathogenic examples"
