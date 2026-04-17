"""Multi-scale Bloom pattern index (DNA/RNA alphabet-aware)."""

from bloom_seq.plugins.multiscale_bloom.index import MultiScaleBloomFilter

MultiScaleBloomIndex = MultiScaleBloomFilter

__all__ = ["MultiScaleBloomFilter", "MultiScaleBloomIndex"]
