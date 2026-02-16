"""Tests for MultiScaleBloomFilter."""

import pytest
import numpy as np
from bloom_dnabert.bloom_filter import MultiScaleBloomFilter


@pytest.fixture
def bloom_filter():
    bf = MultiScaleBloomFilter(capacity=100000, error_rate=0.001)
    bf.load_hbb_pathogenic_variants()
    return bf


class TestBloomFilterInit:
    def test_creates_three_scale_filters(self):
        bf = MultiScaleBloomFilter()
        assert bf.k_sizes == [6, 8, 10]
        assert len(bf.filters) == 3

    def test_custom_capacity_and_error_rate(self):
        bf = MultiScaleBloomFilter(capacity=50000, error_rate=0.01)
        assert len(bf.filters) == 3


class TestPathogenicKmers:
    def test_add_single_kmer(self):
        bf = MultiScaleBloomFilter()
        bf.add_pathogenic_kmer("ATCGATCG")
        assert bf.pathogenic_kmer_count[8] == 1

    def test_add_pathogenic_kmers_from_sequence(self, bloom_filter):
        for k in [6, 8, 10]:
            assert bloom_filter.pathogenic_kmer_count[k] > 0

    def test_sickle_cell_kmer_detected(self, bloom_filter):
        # The sickle cell mutation creates the k-mer CCTGTG
        assert "CCTGTG" in bloom_filter.filters[6]

    def test_hbc_kmer_detected(self, bloom_filter):
        assert "CCTAAG" in bloom_filter.filters[6]


class TestCheckSequence:
    def test_returns_dict_with_all_k_sizes(self, bloom_filter):
        seq = "ATCGATCGATCGATCGATCG"
        result = bloom_filter.check_sequence(seq)
        assert set(result.keys()) == {6, 8, 10}

    def test_sickle_cell_sequence_has_hits(self, bloom_filter):
        sickle_seq = "CATCTGACTCCTGTGGAGAAGTCTGCC"
        result = bloom_filter.check_sequence(sickle_seq)
        assert any(result[6])

    def test_random_sequence_fewer_hits(self, bloom_filter):
        random_seq = "AAAAAAAAAAAAAAAAAAAAA"
        result = bloom_filter.check_sequence(random_seq)
        total_hits = sum(sum(hits) for hits in result.values())
        # Random sequence should have very few or no hits
        assert total_hits <= 5


class TestFeatureVector:
    def test_feature_vector_shape(self, bloom_filter):
        seq = "CACGTGGACTACCCCTGAGGAGAAGTCTGCC"
        fv = bloom_filter.get_feature_vector(seq)
        assert fv.shape == (18,)
        assert fv.dtype == np.float32

    def test_feature_vector_deterministic(self, bloom_filter):
        seq = "CACGTGGACTACCCCTGAGGAGAAGTCTGCC"
        fv1 = bloom_filter.get_feature_vector(seq)
        fv2 = bloom_filter.get_feature_vector(seq)
        np.testing.assert_array_equal(fv1, fv2)


class TestPositionalSignal:
    def test_positional_signal_shape(self, bloom_filter):
        seq = "CACGTGGACTACCCCTGAGGAG"
        signal = bloom_filter.get_positional_signal(seq)
        assert signal.shape == (len(seq), 3)
        assert signal.dtype == np.float32

    def test_positional_signal_range(self, bloom_filter):
        seq = "CACGTGGACTACCCCTGAGGAGAAGTCTGCC"
        signal = bloom_filter.get_positional_signal(seq)
        assert np.all(signal >= 0.0)
        assert np.all(signal <= 1.0)


class TestTokenAlignedSignal:
    def test_token_aligned_signal_shape(self, bloom_filter):
        seq = "CACGTGGACTACCCCTGAGGAG"
        token_spans = [(0, 3), (3, 6), (6, 10), (10, 15), (15, 22)]
        signal = bloom_filter.get_token_aligned_signal(seq, token_spans)
        assert signal.shape == (5, 3)

    def test_empty_spans(self, bloom_filter):
        seq = "CACGTGGACTAC"
        signal = bloom_filter.get_token_aligned_signal(seq, [])
        assert signal.shape == (0, 3)
