import sys
import os

if sys.platform == "win32":
    os.system("chcp 65001 > nul")

from bloom_dnabert import MultiScaleBloomFilter, DNABERTWrapper
from bloom_dnabert.data_loader import ClinVarDataLoader
from bloom_dnabert.settings import load_settings

print("=" * 60)
print("Testing Bloom-Enhanced DNABERT System")
print("=" * 60)

settings = load_settings()

print("\n1. Testing Bloom Filter...")
try:
    bc = settings.bloom
    bloom_filter = MultiScaleBloomFilter(
        capacity=bc.capacity,
        error_rate=bc.error_rate,
        k_sizes=bc.k_sizes,
    )
    bloom_filter.load_pathogenic_seeds(bc.seeds_path)
    sickle_seq = "CACGTGGTCTACCCCTGAGGAG"
    features = bloom_filter.get_hit_features(sickle_seq)
    print("   [OK] Bloom filter initialized")
    print("   [OK] Feature extraction works")
    k8 = bc.k_sizes[1] if len(bc.k_sizes) > 1 else bc.k_sizes[0]
    print(f"     Hit ratio (k={k8}): {features[f'hit_ratio_k{k8}']:.3f}")
except Exception as e:
    print(f"   [ERROR] {e}")
    sys.exit(1)

print("\n2. Testing DNABERT Wrapper...")
try:
    dc = settings.dnabert
    dnabert = DNABERTWrapper(
        model_name=dc.model_name,
        tokenizer_max_length=dc.tokenizer_max_length,
    )
    test_seq = "ATCGATCGATCGATCG"
    embedding = dnabert.get_embedding(test_seq)
    print("   [OK] DNABERT loaded")
    print(f"   [OK] Embedding shape: {embedding.shape}")
except Exception as e:
    print(f"   [ERROR] {e}")
    sys.exit(1)

print("\n3. Testing Data Loader...")
try:
    data_loader = ClinVarDataLoader(settings)
    train_df, val_df, test_df = data_loader.get_training_data()
    print("   [OK] Data loader works")
    print(f"   [OK] Training samples: {len(train_df)}")
    print(f"   [OK] Val samples: {len(val_df)}")
    print(f"   [OK] Test samples: {len(test_df)}")
except Exception as e:
    print(f"   [ERROR] {e}")
    sys.exit(1)

print("\n4. Testing Visualizer...")
try:
    from bloom_dnabert import AttentionVisualizer

    AttentionVisualizer(dnabert, bloom_filter)
    print("   [OK] Visualizer initialized")
    print("   [OK] Visualizer ready for heatmap generation")
except Exception as e:
    print(f"   [ERROR] {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n5. Testing Classifier Pipeline...")
try:
    from bloom_dnabert.classifier import HybridClassifierPipeline

    pipeline = HybridClassifierPipeline(
        bloom_filter, dnabert, settings=settings
    )
    test_seq = "CACGTGGTCTACCCCTGAGGAG"
    bloom_feat, dnabert_emb = pipeline.extract_features(test_seq)
    print("   [OK] Classifier pipeline initialized")
    print(f"   [OK] Bloom features shape: {bloom_feat.shape}")
    print(f"   [OK] DNABERT embedding shape: {dnabert_emb.shape}")
except Exception as e:
    print(f"   [ERROR] {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("[SUCCESS] All tests passed!")
print("=" * 60)
print("\nSystem is ready to use!")
print("\nTo launch the web dashboard, run:")
print("  python app.py")
