"""Manual BGPCA training smoke (one epoch). Not collected by pytest. See CONTRIBUTING.md."""

import traceback

try:
    from bloom_seq.plugins.multiscale_bloom import MultiScaleBloomFilter
    from bloom_seq.plugins.dnabert2.wrapper import DNABERTWrapper
    from bloom_seq.pipeline import BloomGuidedPipeline
    import pandas as pd
    
    bloom = MultiScaleBloomFilter(capacity=100000, error_rate=0.001)
    bloom.load_hbb_pathogenic_variants()
    dnabert = DNABERTWrapper()
    pipeline = BloomGuidedPipeline(bloom, dnabert)
    
    seqs = ["CATCTGACTCCTGTGGAGAAGTCTGCC"] * 4
    labels = [1, 1, 1, 1]
    
    print("Testing train...")
    pipeline.train(seqs, labels, epochs=1)
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()
