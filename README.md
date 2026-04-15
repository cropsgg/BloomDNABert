# BloomDNABert

BloomDNABert is a **research system** that studies how to classify **DNA sequence windows** around variants as **pathogenic-like vs benign-like**, using **HBB (hemoglobin beta)** as the primary worked example (sickle cell and related alleles). It is **not** a clinical diagnostic tool.

---

## What we are trying to do

Human genetics produces enormous numbers of DNA variants. For a given locus, the practical question is often: does this sequence pattern behave like known **pathogenic** changes or like **benign** ones? Classical approaches use expert rules, population data, and literature. Here we explore a **hybrid machine-learning approach** that combines:

1. **Prior biological knowledge encoded as k-mers** — pathogenic-associated short motifs are indexed in **Bloom filters** for fast, position-resolved signals (with the usual Bloom false-positive tradeoff).
2. **General sequence representation** — **DNABERT-2** embeds the local window so the model can pick up context beyond fixed k-mer lists.
3. **Structured fusion** — the **Bloom-Guided Positional Cross-Attention (BGPCA)** design ties Bloom hit locations to transformer tokens so positional information is not collapsed too early; a simpler **baseline** concatenates Bloom summaries with pooled DNABERT features for comparison.

**Problem we address in software:** build a **config-driven**, **gene-agnostic pipeline skeleton** (defaults target **HBB / NM_000518.5**) that can **pull ClinVar** summaries when the network is available, **validate** variants against a **reference FASTA**, **augment** with controlled **synthetic** examples, **train** two model styles (baseline vs BGPCA), and **inspect** behavior in a **Gradio** dashboard (predictions, attention-related views, Bloom context).

---

## What we are trying to achieve

**Scientific and engineering goals:**

| Goal | Meaning |
|------|--------|
| **Biologically grounded labels** | Training examples should use **reference-checked** alleles where possible: ClinVar rows are only kept if HGVS resolves to an **SNV**, the **reference base matches** the FASTA, and the window is actually **mutated** (no pathogenic label on wild-type sequence). Intronic HGVS that do not map cleanly to linear indices use explicit **`hgvs_linear_overrides`** in config so coordinates stay consistent with the bundled reference. |
| **Reproducible configuration** | One **YAML** profile (`config/default.yaml`) drives gene symbol, transcript id, reference path, ClinVar query, Bloom and DNABERT settings, training hyperparameters, and cache layout. Override with env **`BLOOM_CONFIG`** pointing at another YAML file. |
| **Interpretability hooks** | Bloom provides **where** k-mer hits land; DNABERT provides **context**; BGPCA and the dashboard visualize **attention-related** structure so researchers can sanity-check whether the model focuses on plausible regions. |
| **Honest scope** | The model predicts a **binary research label** on **synthetic + ClinVar-derived windows**, not clinical pathogenicity. **Uncertainty** (e.g. MC dropout where enabled) indicates model doubt, not patient risk. |

**Outcomes we want from this repository:**

- A **clear baseline** (concat + MLP) vs **BGPCA** comparison on the same data pipeline.
- A **documented data path** from ClinVar → parsed HGVS → sequence windows → train/val/test splits with **leakage reduction** (duplicate sequences trimmed across splits where implemented).
- A **runnable demo** (`app.py` / `run_app.sh`) for interactive exploration.

---

## How it works (high level)

1. **Reference** — Load the cDNA (or aligned) sequence from `reference.fasta_path` in config. Optional SHA-256 check. Ambiguous bases (e.g. `N`) can be rejected when `annotations.reject_ambiguous_bases_in_reference` is true.
2. **Variants** — `ClinVarDataLoader` searches ClinVar with the configured term, parses **simple cDNA SNV** HGVS, requires **RefSeq in the title** when enabled, matches **ref allele** to the FASTA, and builds a labeled window per variant. If ClinVar is unavailable, **synthetic** pathogenic/benign/VUS examples from YAML still run.
3. **Features** — Bloom: multi-scale k-mers from a **seed file**; DNABERT: tokenizer + encoder; pipelines optionally use **caching**, **DataLoader** batching, and **AMP** per training config.
4. **Models** — **HybridClassifierPipeline** (baseline) vs **BloomGuidedPipeline** (BGPCA); training knobs live under `training:` in YAML.

BGPCA idea (attention with Bloom-derived bias) is summarized in the architecture diagram in the section below; mathematically, Bloom-derived structure enters as **positional bias** into cross-attention so high Bloom signal positions can pull attention before pooling and classification.

---

## Architecture (BGPCA vs baseline)

**Baseline:** Bloom → fixed-size summary (e.g. per-scale statistics) + DNABERT → mean-pooled vector → **concatenation** → MLP.

**BGPCA:** Bloom → **per-position** encoding; DNABERT → **per-token** hidden states → **cross-attention** with Bloom-guided bias → mutation-aware pooling and **gated** fusion with Bloom summaries → classifier (+ optional uncertainty).

```
DNA window |
    +--> Multi-scale Bloom (k=6,8,10) --> positional signal + summaries
    |
    +--> DNABERT-2 --> per-token hidden states |
              +--> Bloom-guided cross-attention + pooling + gate --> logit / uncertainty
```

---

## Configuration

- **Default file:** `config/default.yaml`
- **Override:** set `BLOOM_CONFIG` to an absolute or relative path to another YAML file.
- **Important keys for biological consistency:**
  - `gene.refseq_transcript` — used to gate ClinVar titles (e.g. NM_000518).
  - `annotations.cds_end_exclusive_0` — end of first CDS segment in **0-based exclusive** indexing for intron linearization heuristics.
  - `clinvar.require_refseq_in_title` — drop ClinVar rows whose title does not contain the configured RefSeq stem.
  - `clinvar.hgvs_linear_overrides` — map specific HGVS strings to **0-based** indices when automatic linearization is insufficient (e.g. complex intronic numbering vs your FASTA).

Paths in YAML are resolved relative to the **project root** (parent of `bloom_dnabert/`).

---

## Quick start

**Option A — helper script (creates `.venv` with Python 3.12 if available):**

```bash
./run_app.sh
```

**Option B — manual:**

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```

**CLI training example:**

```bash
.venv/bin/python run_training.py
```

Gradio listens on `0.0.0.0` by default; port may be chosen from `gradio.port_scan_start`–`port_scan_end` if `server_port` is null.

---

## Python API (config-aware)

```python
from pathlib import Path
from bloom_dnabert import MultiScaleBloomFilter, DNABERTWrapper, load_settings
from bloom_dnabert.classifier import HybridClassifierPipeline, BloomGuidedPipeline
from bloom_dnabert.data_loader import ClinVarDataLoader

settings = load_settings(Path("config/default.yaml"))
data_loader = ClinVarDataLoader(settings)
train_df, val_df, test_df = data_loader.get_training_data(use_cache=True)

bc, dc = settings.bloom, settings.dnabert
bloom = MultiScaleBloomFilter(
    capacity=bc.capacity,
    error_rate=bc.error_rate,
    k_sizes=bc.k_sizes,
)
bloom.load_pathogenic_seeds(bc.seeds_path)

dnabert = DNABERTWrapper(
    model_name=dc.model_name,
    tokenizer_max_length=dc.tokenizer_max_length,
)

baseline = HybridClassifierPipeline(bloom, dnabert, settings=settings)
bgpca = BloomGuidedPipeline(bloom, dnabert, settings=settings)
```

---

## Project structure (current)

```
BloomDNABert/
├── app.py                 # Gradio dashboard
├── run_training.py        # Example training entrypoint
├── run_app.sh             # Venv + app launcher
├── requirements.txt
├── config/
│   └── default.yaml       # Primary configuration
├── bloom_dnabert/
│   ├── settings.py        # Pydantic models + load_settings()
│   ├── data_loader.py     # ClinVar + synthetic data
│   ├── reference.py       # FASTA loading
│   ├── bloom_filter.py
│   ├── dnabert_wrapper.py
│   ├── bloom_attention_bridge.py
│   ├── classifier.py
│   ├── feature_cache.py
│   ├── codon.py
│   ├── data/
│   │   ├── reference/     # e.g. HBB transcript FASTA
│   │   ├── codon_table.json
│   │   └── pathogenic_kmer_seeds.txt
│   └── ...
├── tests/
└── data/                  # Default cache dir for variant CSV (configurable)
```

---

## Data sources and limitations

- **ClinVar** — Public NCBI summaries; parsing is limited to **simple SNV** HGVS patterns and **transcript gating** as configured. Many real variants (indels, complex HGVS) are skipped.
- **Synthetic data** — Augments class balance and covers known HBB examples (e.g. HbS, HbC, HbE, selected thalassemia-style entries per YAML). **Benign intronic** rows use synthetic HGVS-like strings; they are for **training distribution**, not clinical assertions.
- **Bloom seeds** — K-mers are a **prior**, not a gold standard; false positives exist by design.

---

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
```

Use a Python version for which **PyTorch** and other dependencies in `requirements.txt` install cleanly (often **3.10–3.12**).

---

## Disclaimer

**Research and education only.** Do not use this system for clinical diagnosis, treatment, or genetic counseling decisions. Variant interpretation requires validated pipelines, human expertise, and appropriate regulatory oversight.

---

## Related concepts (literature pointers)

- DNABERT-2 and related DNA transformers  
- Bloom filters in bioinformatics (k-mer indexing)  
- ClinVar and HGVS nomenclature  
- Monte Carlo dropout for uncertainty (where implemented)
