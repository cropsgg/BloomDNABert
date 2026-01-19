"""
Test script to verify the Bloom-Enhanced DNABERT system
"""

import sys
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')

from bloom_dnabert import MultiScaleBloomFilter, DNABERTWrapper
from bloom_dnabert.data_loader import ClinVarDataLoader

print("="*60)
print("Testing Bloom-Enhanced DNABERT System")
print("="*60)

# Test 1: Bloom Filter
print("\n1. Testing Bloom Filter...")
try:
    bloom_filter = MultiScaleBloomFilter(capacity=10000, error_rate=0.001)
    bloom_filter.load_hbb_pathogenic_variants()
    
    # Test with sickle cell sequence
    sickle_seq = "CACGTGGTCTACCCCTGAGGAG"
    features = bloom_filter.get_hit_features(sickle_seq)
    print(f"   [OK] Bloom filter initialized")
    print(f"   [OK] Feature extraction works")
    print(f"     Hit ratio (k=8): {features['hit_ratio_k8']:.3f}")
except Exception as e:
    print(f"   [ERROR] {e}")
    sys.exit(1)

# Test 2: DNABERT Wrapper
print("\n2. Testing DNABERT Wrapper...")
try:
    dnabert = DNABERTWrapper()
    test_seq = "ATCGATCGATCGATCG"
    embedding = dnabert.get_embedding(test_seq)
    print(f"   [OK] DNABERT loaded")
    print(f"   [OK] Embedding shape: {embedding.shape}")
except Exception as e:
    print(f"   [ERROR] {e}")
    sys.exit(1)

# Test 3: Data Loader
print("\n3. Testing Data Loader...")
try:
    data_loader = ClinVarDataLoader()
    train_df, test_df = data_loader.get_training_data(test_split=0.2)
    print(f"   [OK] Data loader works")
    print(f"   [OK] Training samples: {len(train_df)}")
    print(f"   [OK] Test samples: {len(test_df)}")
except Exception as e:
    print(f"   [ERROR] {e}")
    sys.exit(1)

# Test 4: Visualizer
print("\n4. Testing Visualizer...")
try:
    from bloom_dnabert import AttentionVisualizer
    visualizer = AttentionVisualizer(dnabert, bloom_filter)
    
    test_seq = "CACGTGGTCTACCCCTGAGGAGAAGTCT"
    # Just test that visualizer can be instantiated
    print(f"   [OK] Visualizer initialized")
    print(f"   [OK] Visualizer ready for heatmap generation")
except Exception as e:
    print(f"   [ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Classifier Pipeline
print("\n5. Testing Classifier Pipeline...")
try:
    from bloom_dnabert.classifier import HybridClassifierPipeline
    
    pipeline = HybridClassifierPipeline(bloom_filter, dnabert)
    
    # Extract features for one sequence
    test_seq = "CACGTGGTCTACCCCTGAGGAG"
    bloom_feat, dnabert_emb = pipeline.extract_features(test_seq)
    
    print(f"   [OK] Classifier pipeline initialized")
    print(f"   [OK] Bloom features shape: {bloom_feat.shape}")
    print(f"   [OK] DNABERT embedding shape: {dnabert_emb.shape}")
except Exception as e:
    print(f"   [ERROR] {e}")
    sys.exit(1)

print("\n" + "="*60)
print("[SUCCESS] All tests passed!")
print("="*60)
print("\nSystem is ready to use!")
print("\nTo launch the web dashboard, run:")
print("  python app.py")
