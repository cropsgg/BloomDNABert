# Bloom-Enhanced DNABERT for Sickle Cell Variant Classification

A **plugin-oriented framework** for biosequence foundation models: combine **pattern indexes** (e.g. multi-scale Bloom filters) with **transformer backbones** (reference: **DNABERT-2**) using the novel **Bloom-Guided Positional Cross-Attention (BGPCA)** stack for position-aware fusion and uncertainty estimation. Third-party wheels can register new backbones, alphabets, indexes, and data sources via Python entry points—see [PLUGINS.md](PLUGINS.md) and the model ideas catalog in [MODELS.md](MODELS.md).

**This repository is a research and education codebase, not a medical device.** Outputs are not validated for clinical diagnosis or treatment. See [Disclaimer](#disclaimer) below.

**Documentation map:** This README is the main manual. [DATASETS.md](DATASETS.md) covers ClinVar sources, file formats, and build scripts. [CONTRIBUTING.md](CONTRIBUTING.md) describes development setup, tests, and root-level smoke scripts. [QUICKSTART.md](QUICKSTART.md) and [PROJECT_FILES_GUIDE.md](PROJECT_FILES_GUIDE.md) provide shorter orientation. **Plugin development:** [PLUGINS.md](PLUGINS.md), [MODELS.md](MODELS.md).

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
- **ClinVar integration**: *HBB*-focused live API, curated CSVs, or optional **large pan-gene GRCh38 SNV** windows from `scripts/build_clinvar_pan_dataset.py` (see [DATASETS.md](DATASETS.md)). Training **does not** fall back to synthetic data unless you explicitly opt in (`allow_synthetic=True` or `BLOOM_DNABERT_ALLOW_SYNTHETIC=1`) for tests or experiments.
- **Interactive web dashboard**: Gradio UI for training and sequence analysis

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
      |   [seq_len, n_scales]    [feature_dim]
      |           |                    |
      |    [PositionalBloomEncoder]    |
      |      (multi-scale 1D CNN)      |
      |      [seq_len, d_bloom]         |
      |           |                    |
      +---> [Backbone encoder]         |
                  |                    |
           Per-token hidden            |
        [seq_len, d_model]             |
                  |                    |
      [BloomGuidedCrossAttention]      |
        Q = backbone tokens            |
        K = Bloom encodings            |
        V = backbone tokens            |
        Bias = Bloom activation        |
              x N layers               |
                  |                    |
       [MutationAwarePooling]          |
         (Bloom-weighted attention     |
          over positions)              |
                  |                    |
              [d_model]                |
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
| Backbone features | Pooled embedding | Per-token hidden states |
| Uncertainty | Not available | Monte Carlo dropout |
| Interpretability | Basic attention maps | Position importance + gate values |

### Comparison: Baseline vs BGPCA

| Component | Baseline | BGPCA |
|-----------|----------|-------|
| Bloom features | Summary vector (index `feature_dim`) | Per-position activation signal |
| Backbone features | Mean-pooled hidden states | Per-token hidden states |
| Fusion | `cat(bloom, backbone)` → MLP | Cross-attention + gated fusion |
| Architecture | 2-layer MLP on concatenated features | Bloom encoder + 2× cross-attn + pooling + gate + classifier |
| Position awareness | None | Full positional correspondence |
| Uncertainty | No | MC Dropout (20 samples) |
| Interpretability | Attention heatmap | Position importance, cross-attn weights, gate values |

## Architecture overview (code map)

| Concern | Module | Role |
|--------|--------|------|
| Core protocols + registry | [bloom_seq/protocols.py](bloom_seq/protocols.py), [bloom_seq/registry.py](bloom_seq/registry.py) | Typed contracts; `importlib.metadata` entry points (`bloom_seq.*`) |
| Multi-scale Bloom filter + k-mer hits | [bloom_seq/plugins/multiscale_bloom/index.py](bloom_seq/plugins/multiscale_bloom/index.py) | O(1) pathogenic k-mer checks; positional signal for BGPCA |
| DNABERT-2 encode + hidden states | [bloom_seq/plugins/dnabert2/wrapper.py](bloom_seq/plugins/dnabert2/wrapper.py), [bloom_seq/plugins/dnabert2/backbone.py](bloom_seq/plugins/dnabert2/backbone.py) | Transformer embeddings; supports offline cache env vars |
| Baseline hybrid MLP | [bloom_seq/pipeline.py](bloom_seq/pipeline.py) (`HybridClassifierPipeline`) | Concat(Bloom summary, pooled backbone embedding) → MLP |
| BGPCA stack | [bloom_seq/bridge.py](bloom_seq/bridge.py), [bloom_seq/pipeline.py](bloom_seq/pipeline.py) (`BloomGuidedPipeline`) | Cross-attention with Bloom bias, gated fusion, MC dropout |
| Training data | [bloom_seq/plugins/clinvar_hbb/source.py](bloom_seq/plugins/clinvar_hbb/source.py) (`ClinVarDataLoader`, `DataSourceError`) | CSV / API resolution; stratified splits without sequence leakage |
| Sequence priors / plausibility | [bloom_seq/plugins/plausibility_dna_trinuc/prior.py](bloom_seq/plugins/plausibility_dna_trinuc/prior.py) | Trinucleotide context helpers (bundled JSON in plugin) |
| Web UI | [app.py](app.py) | Gradio dashboard |
| Legacy imports | [bloom_dnabert/__init__.py](bloom_dnabert/__init__.py) | Deprecation shim re-exporting `bloom_seq` |

Bloom filters are seeded from known *HBB* pathogenic k-mers for this project’s scope; using the same seeds on pan-gene CSV training is a documented design choice (see [DATASETS.md](DATASETS.md)).

## Data sources and resolution order

The loader resolves variants in this order (under default `cache_dir=` `data/`):

| Priority | File / source | Notes |
|----------|----------------|-------|
| 1 | `data/clinvar_pan_grch38_snvs.csv` | Built with `scripts/build_clinvar_pan_dataset.py` + GRCh38 FASTA (large; ignored by git by default). |
| 2 | `data/hbb_clinvar_refined.csv` | Small *HBB* exonic SNV slice; build with `scripts/build_hbb_clinvar_dataset.py` (requires network). |
| 3 | `data/hbb_variants.csv` | Cached ClinVar API pull; valid if `version >= 4`. |
| 4 | Live NCBI ClinVar (eutils) | Fetched when no usable CSV exists; subject to rate limits. |

If none of the above produce rows and synthetic mode is **off**, `ClinVarDataLoader.fetch_hbb_variants()` raises `DataSourceError` with paths and pointers to [DATASETS.md](DATASETS.md).

**First-time setup for training:** run `python scripts/build_hbb_clinvar_dataset.py` (or place a compatible CSV). The pan-genome table is optional and needs a local reference FASTA; see [DATASETS.md](DATASETS.md).

## Quick start

### 1. Install

**Option A — editable install (recommended for contributors):**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dl,ui,dev]"
```

**Core-only install** (no PyTorch / Gradio; registry + protocol tests only):

```bash
pip install -e "."
```

**Option B — requirements file only:**

```bash
pip install -r requirements.txt
pip install pytest pytest-cov   # optional, for running tests
```

Python **3.10+** is supported (CI exercises 3.11 and 3.12).

### 2. Hugging Face cache and optional `HF_HOME`

Model weights for `zhihan1996/DNABERT-2-117M` download into the Hugging Face cache. You do **not** need to set `HF_HOME` unless you want a custom location (e.g. a larger disk). Example:

```bash
export HF_HOME="$HOME/.cache/huggingface"
```

For air-gapped use, pre-populate the cache and set `HF_HUB_OFFLINE=1` as described in [bloom_seq/plugins/dnabert2/wrapper.py](bloom_seq/plugins/dnabert2/wrapper.py).

### 3. Launch the web dashboard

```bash
python app.py
```

The UI is served at `http://127.0.0.1:7860` by default. Equivalent entry point: `python launch_dashboard.py` (UTF-8 console tweaks on Windows only).

**Inference** loads Bloom + DNABERT without training data. **Training** inside the app requires real variant rows per the data table above; otherwise the loader raises `DataSourceError`.

### 4. Using the dashboard

1. **Train the model** (requires data): open the **Train Model** tab, choose **BGPCA** or **Baseline**, set epochs, click **Train Model**.
2. **Analyze sequences**: open **Analyze Sequence**, paste ATCG sequence (length limits enforced in code), run analysis, inspect probability, uncertainty, attention, and Bloom hits.

### 5. Triton compatibility (DNABERT-2)

Upstream DNABERT-2 may import Flash attention paths that expect **Triton**. This repo ships [bloom_seq/plugins/dnabert2/triton_compat.py](bloom_seq/plugins/dnabert2/triton_compat.py) and [create_triton_stub.py](create_triton_stub.py) so CPU / non-Triton PyTorch can run the model safely. You normally do not need to run the script unless your environment still pulls incompatible Flash kernels; see comments in `wrapper.py`.

A shorter checklist lives in [QUICKSTART.md](QUICKSTART.md).

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
- **Input validation** enforces 10-5000 bp length, DNA/RNA alphabet (including ambiguity symbols when configured), and sanitizes all inputs before inference

## Python API

### Novel BGPCA Architecture

```python
from bloom_seq.plugins.multiscale_bloom import MultiScaleBloomFilter
from bloom_seq.plugins.dnabert2.wrapper import DNABERTWrapper
from bloom_seq.pipeline import BloomGuidedPipeline

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
from bloom_seq.pipeline import HybridClassifierPipeline

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

## Project structure

```
BloomDNABert/
+-- app.py                         # Gradio web dashboard
+-- launch_dashboard.py            # Thin launcher (imports app.main)
+-- pyproject.toml                 # bloom-seq metadata, extras [dl]/[ui]/[dev], entry points
+-- requirements.txt               # Practical full stack (≈ pip install -e ".[dl,ui]")
+-- PLUGINS.md, MODELS.md          # Plugin how-to and model catalog (docs only)
+-- LICENSE                        # MIT
+-- README.md                      # This manual
+-- DATASETS.md                    # Data provenance and build commands
+-- CONTRIBUTING.md                # Dev setup, tests, smoke scripts
+-- CODE_OF_CONDUCT.md
+-- SECURITY.md
+-- .github/workflows/ci.yml       # minimal (core) + full (extras) pytest jobs
+-- bloom_seq/                     # Core framework + bundled reference plugins
|   +-- protocols.py, registry.py, errors.py, alphabets.py, splits.py
|   +-- bridge.py                  # BGPCA layers
|   +-- pipeline.py                # Baseline + BGPCA pipelines
|   +-- viz.py
|   +-- plugins/
|       +-- dnabert2/              # Backbone + Triton compat
|       +-- multiscale_bloom/      # Pattern index
|       +-- clinvar_hbb/           # Data source, seeds, clinvar_pan
|       +-- plausibility_dna_trinuc/
+-- bloom_dnabert/                 # Deprecation shim (re-exports bloom_seq)
|   +-- __init__.py
+-- scripts/
|   +-- build_hbb_clinvar_dataset.py
|   +-- build_clinvar_pan_dataset.py
+-- tests/                         # pytest collection (mocks heavy DL stack where needed)
+-- data/                          # Gitignored large files; see DATASETS.md
    +-- (optional) hbb_clinvar_refined.csv, hbb_variants.csv
    +-- (optional, large) clinvar_pan_grch38_snvs.csv, refs/hg38.fa, ...
```

Root-level `test_system.py`, `test_train.py`, and `test_end_to_end.py` are **manual smoke scripts**, not pytest tests ([CONTRIBUTING.md](CONTRIBUTING.md)).

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

- Training quality depends on real labels (ClinVar CSV/API or pan-genome table); synthetic data is **not** a default training source.
- Not a clinical-grade or regulatory-validated system; independent validation would be required for any diagnostic use.
- DNABERT-2 practical context length is limited by model max length (on the order of **~512–2048** tokens depending on tokenizer settings; see wrapper and model card).
- Bloom filters have false positives (error rate configurable; dashboard uses a fixed capacity/error tradeoff in code).
- BGPCA is heavier than the baseline (sequence-length-dependent attention cost).
- Windows/macOS/Linux CPU paths rely on PyTorch attention implementations; GPU Triton kernels are environment-dependent.

## Testing

Automated tests live under `tests/` and mock Hugging Face / Gradio where appropriate so CI does not download multi-gigabyte weights.

```bash
pip install -e ".[dl,ui,dev]"
pytest tests/ -q
```

For full-stack manual checks with real weights and optional live data, see [CONTRIBUTING.md](CONTRIBUTING.md) (root smoke scripts).

## Community

- [CONTRIBUTING.md](CONTRIBUTING.md) — pull requests, local setup, smoke scripts
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [SECURITY.md](SECURITY.md) — responsible disclosure

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

If you use this software, please cite the repository and the underlying DNABERT-2 model (see the [model card](https://huggingface.co/zhihan1996/DNABERT-2-117M)). Replace maintainer names in `author` when you publish a paper or Zenodo archive.

```bibtex
@software{bloom_seq,
  title        = {Bloom Seq: Plugin Framework for Biosequence Models with Bloom--Transformer {BGPCA}},
  author       = {BloomDNABert contributors},
  year         = {2026},
  note         = {PyPI distribution name bloom-seq; legacy package bloom\_dnabert is deprecated.}
}
```

## Disclaimer

**This is a research prototype for educational and research purposes only.**
This system should NOT be used for clinical diagnosis or treatment decisions.
Always consult with qualified healthcare professionals and genetic counselors for medical interpretation of genetic variants.

---

**Built with**: PyTorch, Transformers, Gradio, Plotly
