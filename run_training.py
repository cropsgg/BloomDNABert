#!/usr/bin/env python3
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bloom_dnabert import MultiScaleBloomFilter, DNABERTWrapper
from bloom_dnabert.classifier import HybridClassifierPipeline
from bloom_dnabert.data_loader import ClinVarDataLoader
from bloom_dnabert.settings import load_settings


def main():
    settings = load_settings()
    print("Loading data (using cache if available)...")
    data_loader = ClinVarDataLoader(settings)
    train_df, val_df, test_df = data_loader.get_training_data(use_cache=True)

    train_sequences = train_df["sequence"].tolist()
    train_labels = train_df["label"].tolist()
    val_sequences = val_df["sequence"].tolist()
    val_labels = val_df["label"].tolist()
    test_sequences = test_df["sequence"].tolist()
    test_labels = test_df["label"].tolist()

    print("Initializing Bloom filter and DNABERT...")
    bc = settings.bloom
    bloom_filter = MultiScaleBloomFilter(
        capacity=bc.capacity,
        error_rate=bc.error_rate,
        k_sizes=bc.k_sizes,
    )
    bloom_filter.load_pathogenic_seeds(bc.seeds_path)
    dc = settings.dnabert
    dnabert = DNABERTWrapper(
        model_name=dc.model_name,
        tokenizer_max_length=dc.tokenizer_max_length,
    )

    print("\nTraining Baseline model (concatenation + MLP)...")
    baseline = HybridClassifierPipeline(
        bloom_filter=bloom_filter,
        dnabert_wrapper=dnabert,
        settings=settings,
    )
    tc = settings.training
    batch_size = int(os.environ.get("BLOOM_TRAIN_BATCH_SIZE", str(tc.batch_size)))
    _ = baseline.train(
        train_sequences=train_sequences,
        train_labels=train_labels,
        val_sequences=val_sequences,
        val_labels=val_labels,
        epochs=tc.baseline_epochs_cli,
        batch_size=batch_size,
        learning_rate=tc.learning_rate_baseline,
        weight_decay=tc.weight_decay,
        patience=tc.patience,
        max_grad_norm=tc.max_grad_norm,
    )
    metrics = baseline.evaluate(test_sequences, test_labels)
    print(f"  Test Accuracy: {metrics['accuracy']:.3f}, AUC-ROC: {metrics['auc_roc']:.3f}")

    print("\nTraining finished successfully (Baseline model).")
    print("To train BGPCA (novel model with uncertainty), use the 'Train Model' tab in the app.")
    print("\nStart the dashboard with:  python app.py")
    print("Then use 'Analyze Sequence' to get predictions (train from the app for BGPCA).")


if __name__ == "__main__":
    main()
