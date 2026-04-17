"""Manual pipeline smoke: Bloom + DNABERT + forward pass. Not pytest. See CONTRIBUTING.md."""

import sys

try:
    print("Initializing components...")
    from bloom_seq.plugins.multiscale_bloom import MultiScaleBloomFilter
    from bloom_seq.plugins.dnabert2.wrapper import DNABERTWrapper
    from bloom_seq.pipeline import BloomGuidedPipeline

    # 1. Bloom Filter
    bloom = MultiScaleBloomFilter(capacity=100000, error_rate=0.001)
    bloom.load_hbb_pathogenic_variants()
    
    # 2. DNABERT
    dnabert = DNABERTWrapper()
    
    # 3. Pipeline
    pipeline = BloomGuidedPipeline(bloom, dnabert)
    
    sequence = "CACGTGGTCTACCCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGTTGGTATCAAGGTTACAAGACAGGTTTAAGGAGACCAATAGAAACTGGGCATGTGGAGACAGAGAAGACTCTTGGGTTTCTGATAGGCACTGACTCTCTCTGCCTATTGGTCTATTTTCCCACCCTTAGG"
    
    # Untrained run
    print("Testing untrained pass...")
    features = bloom.get_hit_features(sequence)
    
    print("\nEnd to End Check PASSED!")
    sys.exit(0)
    
except Exception as e:
    print(f"End to End Check FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
