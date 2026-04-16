import os
import sys
import traceback
os.environ["HF_HOME"] = r"D:\BloomDNABert\.cache\huggingface"

try:
    from bloom_dnabert import MultiScaleBloomFilter, DNABERTWrapper
    from bloom_dnabert.classifier import BloomGuidedPipeline
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
