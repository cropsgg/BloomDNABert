# Bloom-Enhanced DNABERT for Sickle Cell Variant Classification

A hybrid system combining **Bloom filters** for fast pathogenic k-mer detection with **DNABERT-2** embeddings for variant classification, featuring the novel **Bloom-Guided Positional Cross-Attention (BGPCA)** architecture for position-aware cross-modal fusion with uncertainty estimation.

## Key Features

- **BGPCA Architecture (Novel)**: Cross-attention where Bloom filter signals serve as attention biases
- **Multi-scale Bloom Filters**: Fast O(1) lookup of known pathogenic k-mers at k=6, 8, 10
- **DNABERT-2 Integration**: 117M parameter transformer for deep sequence understanding
- **Position-Aware Fusion**: Preserves spatial correspondence between Bloom hits and DNABERT tokens
- **Uncertainty Estimation**: Monte Carlo dropout for epistemic uncertainty quantification
- **Mutation-Aware Pooling**: Bloom-guided position importance weighting
- **Gated Cross-Modal Fusion**: Dynamically balances pattern matching vs contextual understanding
- **Interpretable Visualizations**: Attention heatmaps, position importance, gate values
- **Early Stopping + Best Checkpoint**: Automatic training halt with model restoration
- **K-Fold Cross-Validation**: Reliable performance estimates for small datasets
- **Class-Weighted Loss**: Handles imbalanced pathogenic/benign ratios
- **Calibration Analysis**: ECE/MCE metrics to verify prediction reliability
- **ClinVar Integration**: Real variant data from NCBI with synthetic fallback
- **Interactive Web Dashboard**: Gradio-based UI for real-time analysis

## Novel Contribution: BGPCA Architecture

### The Problem

Existing hybrid approaches for DNA variant classification simply **concatenate** features from different sources (e.g., Bloom filter statistics + DNABERT embeddings), which causes:

1. **Positional information loss**: Bloom filters know *exactly where* pathogenic k-mer hits occur, but this is crushed into 18 scalar summary statistics
2. **No cross-modal interaction**: Bloom signals never influence what the transformer attends to
3. **Naive fusion**: The MLP must learn all modality interactions from scratch
4. **No uncertainty**: No way to express "I'm not confident about this prediction"

### The Solution: Bloom-Guided Positional Cross-Attention

BGPCA preserves per-position Bloom filter signals and uses them as **additive attention biases** in a cross-attention mechanism with DNABERT's per-token hidden states.

**Standard cross-attention:**

```
Attn(Q, K, V) = softmax(QK^T / sqrt(d)) V
```

**Bloom-guided cross-attention (novel):**

```
Attn(Q, K, V; B) = softmax(QK^T / sqrt(d) + phi(B)) V
```

where `phi(B)` is a learned projection of Bloom positional encodings that creates per-head, per-position attention biases. Positions with strong Bloom activation receive higher bias, naturally drawing the model's attention to potential mutation sites.

### Architecture Diagram

```
DNA Sequence
      |
      +---> [Bloom Filter (k=6,8,10)]
      |           |                    |
      |      Per-position signal   Summary features
      |      [seq_len, 3]          [18-dim]
      |           |                    |
      |    [PositionalBloomEncoder]    |
      |      (multi-scale 1D CNN)      |
      |      [seq_len, 64]            |
      |           |                    |
      +---> [DNABERT-2 Encoder]        |
                  |                    |
           Per-token hidden            |
           [seq_len, 768]             |
                  |                    |
      [BloomGuidedCrossAttention]      |
        Q = DNABERT tokens             |
        K = Bloom encodings            |
        V = DNABERT tokens             |
        Bias = Bloom activation        |
              x N layers               |
                  |                    |
       [MutationAwarePooling]          |
         (Bloom-weighted attention     |
          over positions)              |
                  |                    |
              [768-dim]                |
                  |                    |
       [GatedCrossModalFusion] <-------+
         g * cross_attn + (1-g) * bloom_proj
                  |
           [Classification Head]
                  |
         (logit, uncertainty)
```

### Why This Is Novel

| Aspect | Prior Work | BGPCA |
|--------|-----------|-------|
| Bloom filter role | Feature extractor (flat vector) | Attention bias generator (positional) |
| Spatial information | Lost in summarization | Preserved per-position |
| Cross-modal interaction | None (independent streams) | Cross-attention with Bloom bias |
| Fusion strategy | Concatenation | Learned gating |
| DNABERT features | Pooled embedding (768-dim) | Per-token hidden states |
| Uncertainty | Not available | Monte Carlo dropout |
| Interpretability | Basic attention maps | Position importance + gate values |

### Comparison: Baseline vs BGPCA

| Component | Baseline | BGPCA |
|-----------|----------|-------|
| Bloom features | 18-dim summary | Per-position activation signal |
| DNABERT features | Mean-pooled 768-dim | Per-token hidden states |
| Fusion | `cat(bloom, dnabert)` -> MLP | Cross-attention + gated fusion |
| Architecture | 2-layer MLP (786->256->128->1) | Bloom encoder + 2x cross-attn + pooling + gate + classifier |
| Position awareness | None | Full positional correspondence |
| Uncertainty | No | MC Dropout (20 samples) |
| Interpretability | Attention heatmap | Position importance, cross-attn weights, gate values |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch Web Dashboard

```bash
python app.py
```

The dashboard will be available at `http://localhost:7860`

### 3. Using the Dashboard

1. **Train the Model**:
   - Go to the "Train Model" tab
   - Select architecture: **BGPCA (Novel)** or Baseline
   - Adjust epochs (default: 30)
   - Click "Train Model"

2. **Analyze Sequences**:
   - Go to "Analyze Sequence" tab
   - Enter a DNA sequence or load an example
   - Click "Analyze Sequence"
   - View prediction, uncertainty, attention heatmap, and Bloom filter hits

## Architecture Components

### 1. Positional Bloom Encoder

Multi-scale 1D convolutions (kernels: 3, 5, 7) encode raw per-position Bloom filter activation signals into dense embeddings, capturing local hit patterns like mutation hotspots vs isolated false positives.

### 2. Bloom-Guided Cross-Attention

The core innovation. Cross-attention where:
- **Q** (queries) come from DNABERT token representations
- **K** (keys) come from Bloom positional encodings
- **V** (values) come from DNABERT token representations
- **Bias** comes from Bloom activation magnitude (structural prior)

The Bloom bias acts as a "spotlight" that tells the attention mechanism: "pay extra attention to these positions -- the Bloom filter detected known pathogenic patterns here."

### 3. Mutation-Aware Pooling

Instead of mean pooling (treats all positions equally), learns position-wise importance weights guided by Bloom activation. Positions overlapping pathogenic k-mer hits naturally receive higher weight.

### 4. Gated Cross-Modal Fusion

A sigmoid gate that dynamically balances:
- **Cross-attention path** (rich contextual understanding from DNABERT + Bloom spatial information)
- **Bloom summary path** (direct pattern matching features)

When Bloom has strong signal, the gate trusts pattern matching. When weak, it relies on DNABERT's generalization.

### 5. Monte Carlo Dropout Uncertainty

Multiple stochastic forward passes with dropout enabled estimate epistemic uncertainty. High uncertainty indicates the model is unsure -- critical for clinical applications.

## Training Robustness

Both architectures include production-grade training features:

- **Early stopping** with configurable patience (default 10 epochs), tracking validation loss with automatic best-model checkpoint restoration
- **Gradient clipping** (max norm 1.0) prevents gradient explosions
- **Class-weighted BCE loss** automatically compensates for imbalanced pathogenic/benign ratios
- **CosineAnnealing LR scheduler** with warm restarts for smooth convergence
- **K-fold cross-validation** (`pipeline.cross_validate(seqs, labels, n_folds=5)`) for reliable small-dataset evaluation
- **Calibration analysis** (`pipeline.calibration_analysis(seqs, labels)`) computes ECE/MCE to verify predicted probabilities match observed frequencies
- **Input validation** enforces 10-5000 bp length, ATCGN-only alphabet, and sanitizes all inputs before inference

## Python API

### Novel BGPCA Architecture

```python
from bloom_dnabert import MultiScaleBloomFilter, DNABERTWrapper
from bloom_dnabert.classifier import BloomGuidedPipeline

# Initialize
bloom_filter = MultiScaleBloomFilter()
bloom_filter.load_hbb_pathogenic_variants()
dnabert = DNABERTWrapper()

# Create BGPCA pipeline
pipeline = BloomGuidedPipeline(bloom_filter, dnabert)

# Train
pipeline.train(train_sequences, train_labels, epochs=50)

# Predict with uncertainty
result = pipeline.predict_with_uncertainty("CACGTGGTCTACCCCTGAGGAG...")
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.3f}")
print(f"Uncertainty: {result['uncertainty']:.4f} ({result['uncertainty_level']})")

# Predict with full interpretability
interp = pipeline.predict_with_interpretability("CACGTGGTCTACCCCTGAGGAG...")
print(f"Position importance shape: {interp['position_importance'].shape}")
print(f"Gate values (Bloom vs DNABERT): {interp['gate_values'].mean():.3f}")
```

### Baseline Architecture

```python
from bloom_dnabert.classifier import HybridClassifierPipeline

pipeline = HybridClassifierPipeline(bloom_filter, dnabert)
pipeline.train(train_sequences, train_labels, epochs=50)
result = pipeline.predict("CACGTGGTCTACCCCTGAGGAG...")
```

### K-Fold Cross-Validation

```python
# More reliable metrics for small datasets
results = pipeline.cross_validate(all_sequences, all_labels, n_folds=5, epochs=30)
print(f"Accuracy: {results['accuracy_mean']:.3f} +/- {results['accuracy_std']:.3f}")
print(f"AUC-ROC: {results['auc_roc_mean']:.3f} +/- {results['auc_roc_std']:.3f}")
```

### Calibration Analysis

```python
# Verify prediction reliability
cal = pipeline.calibration_analysis(test_sequences, test_labels)
print(f"Expected Calibration Error: {cal['ece']:.4f}")
print(f"Maximum Calibration Error: {cal['mce']:.4f}")
```

### Per-Position Bloom Signal

```python
# Get per-position Bloom activation
signal = bloom_filter.get_positional_signal("CACGTGGTCTACCCCTGAGGAG")
# shape: [seq_len, 3] -- one channel per k-mer scale (k=6,8,10)
```

## Project Structure

```
BloomDNABert/
+-- app.py                              # Gradio web dashboard (both architectures)
+-- requirements.txt                    # Pinned dependency versions
+-- LICENSE                             # MIT License
+-- README.md                           # This file
+-- bloom_dnabert/
|   +-- __init__.py                     # Package exports
|   +-- bloom_filter.py                 # Multi-scale Bloom filter + positional signal
|   +-- dnabert_wrapper.py              # DNABERT-2 wrapper + per-token hidden states
|   +-- bloom_attention_bridge.py       # BGPCA architecture (NOVEL)
|   +-- classifier.py                   # Both pipelines (Baseline + BGPCA)
|   +-- data_loader.py                  # ClinVar API + synthetic data generation
|   +-- visualizer.py                   # Attention heatmap generation
+-- tests/
|   +-- conftest.py                     # Test configuration and mock setup
|   +-- test_bloom_filter.py            # Bloom filter unit tests
|   +-- test_bloom_attention_bridge.py  # BGPCA architecture tests
|   +-- test_classifier.py             # Classifier + input validation tests
|   +-- test_data_loader.py            # Data quality + no-leakage tests
+-- data/
    +-- hbb_variants.csv                # ClinVar + synthetic variant dataset
```

## Scientific Background

### Sickle Cell Disease
- Most common inherited blood disorder
- Caused by HBB gene mutation (chr11:5227002 T>A)
- Point mutation at codon 6: Glu -> Val
- Results in abnormal hemoglobin (HbS)
- Affects millions worldwide

### Related Work
- **DNABERT-2**: Zhou et al. (2023) -- transformer for DNA sequences
- **Bloom Filters in Bioinformatics**: Solomon & Kingsford (2016) -- k-mer indexing
- **ALiBi**: Press et al. (2022) -- learned attention biases (related concept)
- **Perceiver**: Jaegle et al. (2021) -- cross-attention architecture (related concept)
- **MC Dropout**: Gal & Ghahramani (2016) -- uncertainty estimation
- **ClinVar**: Landrum et al. (2018) -- clinical variant database

## Limitations

- ClinVar integration supplements synthetic data; full clinical-grade training requires larger curated datasets
- DNABERT-2 max sequence length: ~2048 tokens (BPE)
- Bloom filters have false positives (tunable, currently 0.1%)
- BGPCA adds computational overhead vs baseline (cross-attention is O(n^2))
- Windows: Uses PyTorch attention (no Triton acceleration)

## Testing

Run the test suite (57 tests covering all components):

```bash
pip install pytest
python -m pytest tests/ -v
```

## Future Work

- Expand ClinVar integration to fetch full variant annotations and evidence levels
- Add support for more genes (BRCA1, BRCA2, TP53)
- Implement counting Bloom filters for frequency tracking
- Add multi-variant analysis (compound heterozygotes)
- Extend BGPCA to multi-class classification (pathogenic subtypes)
- Clinical validation studies on independent datasets
- Fairness analysis across population groups
- Explore Bloom filter as attention bias in other domains (proteomics, drug discovery)

## Citation

```bibtex
@software{bgpca_bloom_dnabert_2026,
  title={Bloom-Guided Positional Cross-Attention for DNA Variant Classification},
  author={Your Name},
  year={2026},
  note={A novel architecture bridging probabilistic data structures with neural
        attention mechanisms for position-aware cross-modal variant classification}
}
```

## Disclaimer

**This is a research prototype for educational and research purposes only.**
This system should NOT be used for clinical diagnosis or treatment decisions.
Always consult with qualified healthcare professionals and genetic counselors for medical interpretation of genetic variants.

---

**Built with**: PyTorch, Transformers, Gradio, Plotly
