#!/usr/bin/env python3
"""
Run model training from the command line (no UI).
Use this to train the model once; then start the app and use "Analyze Sequence".
"""
import sys
from pathlib import Path

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bloom_dnabert import MultiScaleBloomFilter, DNABERTWrapper
from bloom_dnabert.classifier import HybridClassifierPipeline
from bloom_dnabert.data_loader import ClinVarDataLoader


def main():
    print("Loading data (using cache if available)...")
    data_loader = ClinVarDataLoader()
    train_df, val_df, test_df = data_loader.get_training_data(use_cache=True)

    train_sequences = train_df["sequence"].tolist()
    train_labels = train_df["label"].tolist()
    val_sequences = val_df["sequence"].tolist()
    val_labels = val_df["label"].tolist()
    test_sequences = test_df["sequence"].tolist()
    test_labels = test_df["label"].tolist()

    print("Initializing Bloom filter and DNABERT...")
    bloom_filter = MultiScaleBloomFilter(capacity=100000, error_rate=0.001)
    bloom_filter.load_hbb_pathogenic_variants()
    dnabert = DNABERTWrapper()

    # Train Baseline (faster; use this to verify pipeline)
    print("\nTraining Baseline model (concatenation + MLP)...")
    baseline = HybridClassifierPipeline(bloom_filter=bloom_filter, dnabert_wrapper=dnabert)
    history = baseline.train(
        train_sequences=train_sequences,
        train_labels=train_labels,
        val_sequences=val_sequences,
        val_labels=val_labels,
        epochs=30,
        batch_size=16,
    )
    metrics = baseline.evaluate(test_sequences, test_labels)
    print(f"  Test Accuracy: {metrics['accuracy']:.3f}, AUC-ROC: {metrics['auc_roc']:.3f}")

    print("\nTraining finished successfully (Baseline model).")
    print("To train BGPCA (novel model with uncertainty), use the 'Train Model' tab in the app.")
    print("\nStart the dashboard with:  .venv/bin/python app.py")
    print("Then use 'Analyze Sequence' to get predictions (train from the app for BGPCA).")


if __name__ == "__main__":
    main()
