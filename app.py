"""
Gradio Web Dashboard for Bloom-Enhanced DNABERT Variant Classifier

Interactive web interface for analyzing DNA sequences for pathogenic variants
with real-time visualization of attention patterns and Bloom filter hits.

Supports two architectures:
1. Baseline: Simple concatenation + MLP (HybridClassifier)
2. BGPCA: Bloom-Guided Positional Cross-Attention (novel)
"""
# Disable Gradio analytics to avoid outbound HTTPS calls (e.g. checkip.amazonaws.com)
# that can timeout when offline or behind a firewall
import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# Override huggingface cache directory to use D drive (C drive is full)
os.environ["HF_HOME"] = r"D:\BloomDNABert\.cache\huggingface"

import gradio as gr
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import socket

from bloom_dnabert import MultiScaleBloomFilter, DNABERTWrapper, HybridClassifier, AttentionVisualizer
from bloom_dnabert.classifier import HybridClassifierPipeline, BloomGuidedPipeline
from bloom_dnabert.data_loader import ClinVarDataLoader


# ─── Design System CSS ────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Design tokens ── */
:root {
    --bg-base:        #0b0f1e;
    --bg-surface:     rgba(14, 21, 42, 0.92);
    --bg-elevated:    rgba(20, 30, 58, 0.88);
    --bg-input:       rgba(15, 23, 46, 0.95);
    --border-subtle:  rgba(82, 158, 224, 0.14);
    --border-default: rgba(82, 158, 224, 0.26);
    --border-focus:   rgba(82, 158, 224, 0.60);
    --blue:           #529ee0;
    --cyan:           #67d4e8;
    --green:          #56d49b;
    --purple:         #a78bfa;
    --amber:          #f5c842;
    --red:            #f87171;
    --text-1:         #dce8f5;
    --text-2:         #7e97b4;
    --text-3:         #3d5470;
    --shadow-card:    0 4px 24px rgba(0,0,0,0.45);
    --shadow-glow:    0 0 28px rgba(82,158,224,0.22);
    --radius-xl:      16px;
    --radius-lg:      12px;
    --radius-md:      8px;
    --ease:           cubic-bezier(.4,0,.2,1);
}

/* ── Page shell ── */
body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: var(--bg-base) !important;
    background-image:
        radial-gradient(ellipse 90% 50% at 50% -5%,  rgba(82,158,224,0.09) 0%, transparent 65%),
        radial-gradient(ellipse 55% 35% at 95%  95%, rgba(103,212,232,0.05) 0%, transparent 55%) !important;
    color: var(--text-1) !important;
}
footer { display: none !important; }

/* ── Sidebar / topnav strip ── */
#navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.9rem 2rem;
    background: rgba(11,15,30,0.98);
    border-bottom: 1px solid var(--border-subtle);
    position: sticky; top: 0; z-index: 100;
    backdrop-filter: blur(14px);
}
#navbar .brand {
    display: flex; align-items: center; gap: 0.6rem;
    font-size: 1rem; font-weight: 700; letter-spacing: -0.3px;
    color: var(--text-1);
}
#navbar .brand .hex { color: var(--cyan); font-size: 1.2rem; }
#navbar .nav-pills { display: flex; gap: 0.5rem; }
.nav-pill {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.6px;
    text-transform: uppercase; padding: 0.28rem 0.75rem;
    border-radius: 999px; border: 1px solid;
}
.pill-blue   { color: var(--blue);   border-color: rgba(82,158,224,0.35);  background: rgba(82,158,224,0.09); }
.pill-cyan   { color: var(--cyan);   border-color: rgba(103,212,232,0.35); background: rgba(103,212,232,0.09); }
.pill-green  { color: var(--green);  border-color: rgba(86,212,155,0.35);  background: rgba(86,212,155,0.09); }
.pill-purple { color: var(--purple); border-color: rgba(167,139,250,0.35); background: rgba(167,139,250,0.09); }

/* ── Hero ── */
#hero {
    padding: 2.2rem 2.4rem 1.8rem;
    background: linear-gradient(160deg, rgba(12,18,38,0.0) 0%, rgba(82,158,224,0.04) 100%);
    border-bottom: 1px solid var(--border-subtle);
}
#hero h1 {
    font-size: 2rem; font-weight: 800; letter-spacing: -0.8px; line-height: 1.15;
    margin: 0 0 0.55rem;
    background: linear-gradient(130deg, var(--cyan) 0%, var(--blue) 55%, var(--purple) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
#hero p {
    font-size: 0.92rem; color: var(--text-2); line-height: 1.7;
    max-width: 780px; margin: 0;
}

/* ── Tabs ── */
.tabs > .tab-nav {
    background: rgba(11,15,30,0.95) !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    padding: 0 1.5rem !important; gap: 0 !important;
    backdrop-filter: blur(10px) !important;
}
.tabs > .tab-nav button {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.84rem !important; font-weight: 500 !important;
    color: var(--text-2) !important;
    padding: 0.85rem 1.3rem !important;
    border: none !important; border-bottom: 2px solid transparent !important;
    border-radius: 0 !important; background: transparent !important;
    transition: color 0.2s var(--ease), border-color 0.2s var(--ease) !important;
}
.tabs > .tab-nav button:hover { color: var(--blue) !important; }
.tabs > .tab-nav button.selected {
    color: var(--cyan) !important;
    border-bottom-color: var(--cyan) !important;
}

/* ── Section titles inside tabs ── */
.sec-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 1.8px;
    text-transform: uppercase; color: var(--text-3);
    margin: 0 0 0.3rem;
}
.sec-title {
    font-size: 0.98rem; font-weight: 600; color: var(--text-1);
    margin: 0 0 1.2rem;
}

/* ── Glass card ── */
.card {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-xl) !important;
    padding: 1.4rem 1.5rem !important;
    box-shadow: var(--shadow-card) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    transition: border-color 0.22s var(--ease), box-shadow 0.22s var(--ease) !important;
}
.card:hover {
    border-color: var(--border-default) !important;
    box-shadow: var(--shadow-card), var(--shadow-glow) !important;
}

/* ── Inputs ── */
label, .label-wrap span {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.75rem !important; font-weight: 600 !important;
    letter-spacing: 0.8px !important; text-transform: uppercase !important;
    color: var(--text-2) !important;
}
textarea, input[type="text"] {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-lg) !important;
    color: var(--text-1) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important; line-height: 1.6 !important;
    transition: border-color 0.2s var(--ease), box-shadow 0.2s var(--ease) !important;
    caret-color: var(--cyan) !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 3px rgba(82,158,224,0.14) !important;
    outline: none !important;
}
textarea::placeholder { color: var(--text-3) !important; font-style: italic; }

/* ── Buttons ── */
button[variant="primary"], .gr-button-primary {
    background: linear-gradient(135deg, #2d7ad4, var(--blue)) !important;
    border: none !important; color: #fff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important; font-weight: 600 !important;
    border-radius: var(--radius-lg) !important;
    padding: 0.65rem 1.6rem !important;
    box-shadow: 0 4px 16px rgba(45,122,212,0.38) !important;
    transition: transform 0.18s var(--ease), box-shadow 0.18s var(--ease) !important;
}
button[variant="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(45,122,212,0.52) !important;
}
button[variant="primary"]:active { transform: translateY(0) !important; }

button[variant="secondary"], .gr-button-secondary {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-default) !important;
    color: var(--text-2) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important; font-weight: 500 !important;
    border-radius: var(--radius-lg) !important;
    padding: 0.65rem 1.2rem !important;
    transition: border-color 0.18s var(--ease), color 0.18s var(--ease) !important;
}
button[variant="secondary"]:hover {
    border-color: var(--border-focus) !important;
    color: var(--text-1) !important;
}

/* ── Dropdown ── */
select, .wrap select {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-subtle) !important;
    color: var(--text-1) !important;
    border-radius: var(--radius-lg) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Slider ── */
input[type="range"] { accent-color: var(--blue) !important; }

/* ── Radio ── */
.radio-group label {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-lg) !important;
    padding: 0.6rem 1rem !important;
    transition: border-color 0.2s var(--ease), background 0.2s var(--ease) !important;
    cursor: pointer !important;
    font-size: 0.875rem !important;
    color: var(--text-2) !important;
}
.radio-group label:hover {
    border-color: var(--border-default) !important;
    color: var(--text-1) !important;
}
.radio-group label:has(input:checked) {
    border-color: var(--blue) !important;
    background: rgba(82,158,224,0.11) !important;
    color: var(--blue) !important;
    font-weight: 600 !important;
}

/* ── Markdown output panels ── */
.output-panel, .prose, .md {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-xl) !important;
    padding: 1.3rem 1.5rem !important;
    color: var(--text-1) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important; line-height: 1.75 !important;
    backdrop-filter: blur(12px) !important;
}
.output-panel h3, .prose h3, .md h3 {
    font-size: 0.95rem !important; font-weight: 700 !important;
    color: var(--cyan) !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    padding-bottom: 0.4rem !important; margin-bottom: 0.85rem !important;
}
.output-panel strong, .prose strong, .md strong {
    color: var(--blue) !important; font-weight: 600 !important;
}
.output-panel code, .prose code, .md code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important; padding: 0.1em 0.4em !important;
    border-radius: 4px !important;
    background: rgba(0,0,0,0.30) !important;
    color: var(--cyan) !important;
}
.output-panel hr, .prose hr, .md hr {
    border: none !important;
    border-top: 1px solid var(--border-subtle) !important;
    margin: 0.85rem 0 !important;
}
.output-panel table, .prose table, .md table {
    width: 100%; border-collapse: collapse;
    font-size: 0.84rem !important; margin: 0.5rem 0;
}
.output-panel th, .prose th, .md th {
    background: rgba(82,158,224,0.12) !important;
    color: var(--blue) !important; font-weight: 600 !important;
    padding: 0.5rem 0.85rem !important;
    border: 1px solid var(--border-subtle) !important;
    text-align: left;
}
.output-panel td, .prose td, .md td {
    padding: 0.45rem 0.85rem !important;
    border: 1px solid rgba(82,158,224,0.08) !important;
    color: var(--text-1) !important;
}
.output-panel tr:nth-child(even) td, .prose tr:nth-child(even) td, .md tr:nth-child(even) td {
    background: rgba(82,158,224,0.04) !important;
}
.output-panel blockquote, .prose blockquote, .md blockquote {
    border-left: 3px solid var(--blue) !important;
    padding: 0.4rem 0.9rem !important;
    margin: 0.5rem 0 !important;
    background: rgba(82,158,224,0.06) !important;
    border-radius: 0 var(--radius-md) var(--radius-md) 0 !important;
    color: var(--text-2) !important;
    font-size: 0.83rem !important;
}

/* ── Plot areas ── */
.gradio-plot, .plot-container, [data-testid="plot"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-xl) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-card) !important;
}

/* ── Status chip ── */
.chip {
    display: inline-flex; align-items: center; gap: 0.35rem;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.6px;
    text-transform: uppercase; padding: 0.28rem 0.78rem;
    border-radius: 999px; border: 1px solid;
}
.chip-ready  { color: var(--green);  border-color: rgba(86,212,155,0.40);  background: rgba(86,212,155,0.09); }
.chip-warn   { color: var(--amber);  border-color: rgba(245,200,66,0.40);  background: rgba(245,200,66,0.09); }

/* ── Divider ── */
.divider {
    border: none; border-top: 1px solid var(--border-subtle);
    margin: 1rem 0;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(82,158,224,0.22); border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(82,158,224,0.45); }
"""

# ─── Static HTML blocks ───────────────────────────────────────────────────────

NAVBAR_HTML = """
<div id="navbar">
  <div class="brand">
    <span class="hex">⬡</span>
    <span>BloomDNABERT</span>
  </div>
  <div class="nav-pills">
    <span class="nav-pill pill-blue">Bloom Filter</span>
    <span class="nav-pill pill-cyan">DNABERT-2</span>
    <span class="nav-pill pill-purple">BGPCA</span>
    <span class="nav-pill pill-green">MC Dropout</span>
  </div>
</div>
"""

HERO_HTML = """
<div id="hero">
  <h1>Sickle Cell Variant Classifier</h1>
  <p>
    A novel hybrid system combining <strong style="color:#67d4e8">Bloom filters</strong> for O(1) pathogenic k-mer detection
    with <strong style="color:#529ee0">DNABERT-2</strong> transformer embeddings,
    powered by the <strong style="color:#a78bfa">Bloom-Guided Positional Cross-Attention (BGPCA)</strong>
    architecture for position-aware cross-modal fusion.
  </p>
</div>
"""

ANALYZE_HEADER = """
<div style="padding:1.2rem 0 0.2rem">
  <div class="sec-label">Sequence Analysis</div>
  <div class="sec-title">Enter or load a known HBB variant to classify</div>
</div>
"""

TRAIN_HEADER = """
<div style="padding:1.2rem 0 0.2rem">
  <div class="sec-label">Model Training</div>
  <div class="sec-title">Configure architecture and start training</div>
</div>
"""

FOOTER_HTML = """
<div style="text-align:center;padding:1.4rem 0 0.6rem;border-top:1px solid rgba(82,158,224,0.12);margin-top:0.5rem">
  <span style="font-size:0.76rem;color:#3d5470">
    ⚠️ Research prototype — not for clinical use &nbsp;·&nbsp;
    BloomDNABERT &nbsp;·&nbsp; BGPCA Architecture &nbsp;·&nbsp; 2026
  </span>
</div>
"""


class VariantAnalysisDashboard:
    """Interactive web dashboard for variant analysis."""

    def __init__(self):
        """Initialize the dashboard with all components."""
        self.bloom_filter = None
        self.dnabert_wrapper = None
        self.baseline_pipeline = None
        self.bgpca_pipeline = None
        self.visualizer = None
        self.active_pipeline = None
        self.active_model_name = None
        self.trained = False

        print("Initializing Bloom-Enhanced DNABERT Dashboard...")
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all ML components with error handling."""
        try:
            print("Loading Bloom filter...")
            self.bloom_filter = MultiScaleBloomFilter(capacity=100000, error_rate=0.001)
            self.bloom_filter.load_hbb_pathogenic_variants()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Bloom filter: {e}") from e

        try:
            print("Loading DNABERT-2 model...")
            self.dnabert_wrapper = DNABERTWrapper()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load DNABERT-2 model: {e}. "
                "Ensure transformers and model weights are available."
            ) from e

        self.visualizer = AttentionVisualizer(self.dnabert_wrapper, self.bloom_filter)

        self.baseline_pipeline = HybridClassifierPipeline(
            bloom_filter=self.bloom_filter,
            dnabert_wrapper=self.dnabert_wrapper
        )

        self.bgpca_pipeline = BloomGuidedPipeline(
            bloom_filter=self.bloom_filter,
            dnabert_wrapper=self.dnabert_wrapper
        )

        print("Dashboard initialized successfully!")

    def train_model(self, model_choice: str, epochs: int = 30, progress=gr.Progress()):
        """Train the selected classifier model."""
        try:
            progress(0, desc="Loading data...")

            data_loader = ClinVarDataLoader()
            train_df, val_df, test_df = data_loader.get_training_data()

            progress(0.2, desc="Preparing datasets...")

            train_sequences = train_df['sequence'].tolist()
            train_labels = train_df['label'].tolist()
            val_sequences = val_df['sequence'].tolist()
            val_labels = val_df['label'].tolist()
            test_sequences = test_df['sequence'].tolist()
            test_labels = test_df['label'].tolist()

            use_bgpca = model_choice == "BGPCA (Novel Cross-Attention)"

            if use_bgpca:
                pipeline = self.bgpca_pipeline
                model_name = "BGPCA"
                progress(0.3, desc="Training BGPCA model (novel architecture)...")
            else:
                pipeline = self.baseline_pipeline
                model_name = "Baseline"
                progress(0.3, desc="Training Baseline model...")

            history = pipeline.train(
                train_sequences=train_sequences,
                train_labels=train_labels,
                val_sequences=val_sequences,
                val_labels=val_labels,
                epochs=epochs,
                batch_size=16
            )

            progress(0.9, desc="Evaluating on held-out test set...")
            metrics = pipeline.evaluate(test_sequences, test_labels)

            self.active_pipeline = pipeline
            self.active_model_name = model_name
            self.trained = True

            progress(1.0, desc="Training complete!")
        except Exception as e:
            return f"### ❌  Training Failed\n\n**Error:** `{str(e)}`\n\nCheck console output for details."

        arch_note = (
            "- Positional Bloom Encoder (multi-scale 1D CNN)\n"
            "- Bloom-Guided Cross-Attention (2 layers, 4 heads)\n"
            "- Mutation-Aware Pooling\n"
            "- Gated Cross-Modal Fusion\n"
            "- Monte Carlo Dropout Uncertainty"
        ) if use_bgpca else (
            "- Bloom features (18-dim) + DNABERT embedding (768-dim)\n"
            "- Concatenation → 2-layer MLP"
        )

        last_train_loss = history['train_loss'][-1] if history['train_loss'] else 0.0
        last_train_acc  = history['train_acc'][-1]  if history['train_acc']  else 0.0
        last_val_loss   = history['val_loss'][-1]   if history['val_loss']   else 0.0
        last_val_acc    = history['val_acc'][-1]    if history['val_acc']    else 0.0

        return f"""
### ✅  Training Complete — {model_name}

**Architecture**
{arch_note}

---

### 📊  Test-Set Performance

| Metric | Score |
|:-------|------:|
| Accuracy  | `{metrics['accuracy']:.3f}`  |
| Precision | `{metrics['precision']:.3f}` |
| Recall    | `{metrics['recall']:.3f}`    |
| F1 Score  | `{metrics['f1_score']:.3f}`  |
| AUC-ROC   | `{metrics['auc_roc']:.3f}`   |

> Data split: **60 % train / 20 % val / 20 % test** (stratified, no leakage)

---

### 📉  Final Training History

| Phase | Loss | Accuracy |
|:------|-----:|---------:|
| Train      | `{last_train_loss:.4f}` | `{last_train_acc:.4f}` |
| Validation | `{last_val_loss:.4f}`   | `{last_val_acc:.4f}`   |

> Switch to **Analyze Sequence** to classify any HBB variant.
"""

    def analyze_sequence(self, sequence: str):
        """Analyze a DNA sequence."""
        if not sequence or len(sequence) < 10:
            return "Please enter a valid DNA sequence (at least 10 nucleotides).", None, None, "*Sequence too short for pipeline trace.*"

        sequence = sequence.upper().strip()
        if not all(base in 'ATCGN' for base in sequence):
            return "❌  Invalid sequence: only A, T, C, G, N are allowed.", None, None, "*Invalid sequence — pipeline halted.*"

        interp = None
        try:
            if self.trained and self.active_pipeline is not None:
                is_bgpca = isinstance(self.active_pipeline, BloomGuidedPipeline)

                if is_bgpca:
                    result = self.active_pipeline.predict_with_uncertainty(sequence)
                    interp = self.active_pipeline.predict_with_interpretability(sequence)
                    icon = "🔴" if "pathogenic" in result['prediction'].lower() else "🟢"

                    prediction_text = f"""
### 🧬  Prediction — {self.active_model_name}

**Result: {icon} `{result['prediction'].upper()}`**

| Property | Value |
|:---------|------:|
| Probability | `{result['probability']:.3f}` |
| Confidence  | `{result['confidence']:.3f}` |

---

### 🎲  Uncertainty (MC Dropout)

| Property | Value |
|:---------|------:|
| Epistemic Uncertainty | `{result['uncertainty']:.4f}` |
| Uncertainty Level     | `{result['uncertainty_level']}` |

---

### 🔀  Cross-Modal Fusion Gate

| Property | Value |
|:---------|------:|
| Mean gate value | `{interp['gate_values'].mean():.3f}` |

> Gate > 0.5 → model trusts **DNABERT** more
> Gate < 0.5 → model trusts **Bloom filter** more
"""
                else:
                    result = self.active_pipeline.predict(sequence)
                    icon = "🔴" if "pathogenic" in result['prediction'].lower() else "🟢"
                    prediction_text = f"""
### 🧬  Prediction — {self.active_model_name}

**Result: {icon} `{result['prediction'].upper()}`**

| Property | Value |
|:---------|------:|
| Probability | `{result['probability']:.3f}` |
| Confidence  | `{result['confidence']:.3f}` |

---

> Probability > 0.5 → **pathogenic** variant.
> Model is trained specifically on HBB gene variants.
"""
            else:
                result = None
                prediction_text = """
### ⚠️  Model Not Yet Trained

Go to the **Train Model** tab to train first.

Bloom filter and attention visualisations are shown below regardless.
"""

            importance_plot = self.visualizer.create_nucleotide_importance_plot(
                sequence, show_bloom_hits=True
            )
            dashboard_plot = self.visualizer.create_dashboard(
                sequence, prediction_result=result
            )

            # Generate trace
            trace_text = self._generate_pipeline_trace(sequence, result, interp, is_bgpca if self.trained and self.active_pipeline else False)

            return prediction_text, importance_plot, dashboard_plot, trace_text

        except Exception as e:
            return f"**Error during analysis:** `{str(e)}`", None, None, f"**Trace Error:** `{str(e)}`"

    def _generate_pipeline_trace(self, sequence: str, result, interp, is_bgpca: bool) -> str:
        """Generate a detailed step-by-step trace of the pipeline processing."""
        trace = f"## 🧪 Pipeline Execution Trace\n\n"
        
        # 1. Input
        trace += f"### 1. Input Sequence\n"
        trace += f"- **Length**: `{len(sequence)}` base pairs\n"
        
        # 2. Bloom Filter
        trace += f"\n### 2. Multi-Scale Bloom Filter Analysis\n"
        features = self.bloom_filter.get_hit_features(sequence)
        for k in self.bloom_filter.k_sizes:
            count = features.get(f'hit_count_k{k}', 0)
            ratio = features.get(f'hit_ratio_k{k}', 0)
            trace += f"- **k={k}**: `{count}` hits found (Hit Ratio: `{ratio:.4f}`)\n"
        trace += f"- **Mean Hit Ratio**: `{features.get('mean_hit_ratio', 0):.4f}`\n"
        
        if not self.trained or self.active_pipeline is None:
            trace += "\n> ⚠️ *Pipeline stops here because the model is not trained yet.*"
            return trace

        # 3. DNABERT
        trace += f"\n### 3. DNABERT-2 Transformer\n"
        try:
            tokens = self.dnabert_wrapper.tokenizer.tokenize(sequence)
            trace += f"- **Tokenization**: Sequence converted into `{len(tokens)}` BPE tokens.\n"
        except Exception:
            pass
        if is_bgpca:
            trace += f"- **Feature Extraction**: Extracted per-token hidden states `[num_tokens, 768]`.\n"
        else:
            trace += f"- **Feature Extraction**: Extracted pooled embedding `[1, 768]`.\n"

        # 4. Fusion
        trace += f"\n### 4. Cross-Modal Fusion ({self.active_model_name})\n"
        if is_bgpca:
            trace += f"- **Positional Encoding**: Bloom activations projected via 1D CNN.\n"
            trace += f"- **Cross-Attention**: Applied Bloom-Guided Positional Cross-Attention.\n"
            if interp is not None and 'gate_values' in interp:
                gate_mean = interp['gate_values'].mean()
                trace += f"- **Gated Fusion**: Calculated gate weight: `{gate_mean:.3f}`.\n"
                trace += f"  - *(Note: {'>' if gate_mean > 0.5 else '<'} 0.5 indicates heavier reliance on {'DNABERT' if gate_mean > 0.5 else 'Bloom Filter'})*\n"
        else:
            trace += f"- **Concatenation**: Combined 18-dim Bloom features with 768-dim DNABERT embedding.\n"
            trace += f"- **MLP**: Passed 786-dim vector through 2-layer classifier.\n"

        # 5. Output
        trace += f"\n### 5. Final Classification\n"
        if result:
            trace += f"- **Raw Probability (Pathogenic)**: `{result['probability']:.5f}`\n"
            trace += f"- **Final Decision**: `{result['prediction'].upper()}`\n"
            if is_bgpca and 'uncertainty' in result:
                trace += f"- **MC Dropout Uncertainty**: `{result['uncertainty']:.5f}` ({result['uncertainty_level']})\n"
                
        return trace

    def analyze_example(self, example_name: str):
        """Return a pre-defined example sequence."""
        examples = {
            "Normal HBB (Wild-type)":
                "CACGTGGACTACCCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCC"
                "CTGGGCAGGTTGGTATCAAGGTTACAAGACAGGTTTAAGGAGACCAATAGAAACTGGGCATGTGGAGACAGAGAAGACTCTT"
                "GGGTTTCTGATAGGCACTGACTCTCTCTGCCTATTGGTCTATTTTCCCACCCTTAGG",
            "Sickle Cell (HbS E6V)":
                "CACGTGGTCTACCCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCC"
                "CTGGGCAGGTTGGTATCAAGGTTACAAGACAGGTTTAAGGAGACCAATAGAAACTGGGCATGTGGAGACAGAGAAGACTCTT"
                "GGGTTTCTGATAGGCACTGACTCTCTCTGCCTATTGGTCTATTTTCCCACCCTTAGG",
            "HbC Disease (E6K)":
                "CACGTGAAGTACCCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCC"
                "CTGGGCAGGTTGGTATCAAGGTTACAAGACAGGTTTAAGGAGACCAATAGAAACTGGGCATGTGGAGACAGAGAAGACTCTT"
                "GGGTTTCTGATAGGCACTGACTCTCTCTGCCTATTGGTCTATTTTCCCACCCTTAGG",
            "Random Benign Variant":
                "CACGTGGACTACCCCTGAGGAGAAGTCTGCCGTTACTACCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCC"
                "CTGGGCAGGTTGGTATCAAGGTTACAAGACAGGTTTAAGGAGACCAATAGAAACTGGGCATGTGGAGACAGAGAAGACTCTT"
                "GGGTTTCTGATAGGCACTGACTCTCTCTGCCTATTGGTCTATTTTCCCACCCTTAGG",
        }
        return examples.get(example_name, "")

    def create_interface(self):
        """Create the Gradio interface."""

        # Programmatic dark theme via gr.themes.Base
        theme = gr.themes.Base(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.sky,
            neutral_hue=gr.themes.colors.slate,
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
            font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace"],
        ).set(
            body_background_fill              = "#0b0f1e",
            body_background_fill_dark         = "#0b0f1e",
            body_text_color                   = "#dce8f5",
            body_text_color_dark              = "#dce8f5",
            border_color_primary              = "rgba(82,158,224,0.22)",
            border_color_primary_dark         = "rgba(82,158,224,0.22)",
            block_background_fill             = "rgba(14,21,42,0.92)",
            block_background_fill_dark        = "rgba(14,21,42,0.92)",
            input_background_fill             = "rgba(15,23,46,0.95)",
            input_background_fill_dark        = "rgba(15,23,46,0.95)",
            button_primary_background_fill    = "linear-gradient(135deg,#2d7ad4,#529ee0)",
            button_primary_background_fill_dark = "linear-gradient(135deg,#2d7ad4,#529ee0)",
            button_primary_text_color         = "#ffffff",
            button_secondary_background_fill  = "rgba(20,30,58,0.88)",
            button_secondary_text_color       = "#7e97b4",
        )

        with gr.Blocks(
            title="BloomDNABERT — Sickle Cell Variant Classifier",
        ) as interface:
            # Store theme/css so launch() can inject them (Gradio 6+)
            interface._theme = theme
            interface._css = CUSTOM_CSS

            gr.HTML(NAVBAR_HTML)
            gr.HTML(HERO_HTML)

            with gr.Tabs():

                # ── Tab 1 · Analyze ──────────────────────────────────────
                with gr.Tab("🔬  Analyze Sequence"):
                    gr.HTML(ANALYZE_HEADER)

                    with gr.Row(equal_height=False):

                        # Left pane — input
                        with gr.Column(scale=3, min_width=360):
                            sequence_input = gr.Textbox(
                                label="DNA Sequence",
                                placeholder="Paste or type a sequence (A, T, C, G, N)…",
                                lines=6,
                                max_lines=14,
                                elem_id="seq_input",
                            )

                            with gr.Row():
                                analyze_btn = gr.Button(
                                    "⬡  Analyze",
                                    variant="primary",
                                    size="lg",
                                    elem_id="analyze_btn",
                                )
                                clear_btn = gr.ClearButton(
                                    [sequence_input],
                                    value="✕  Clear",
                                    variant="secondary",
                                    elem_id="clear_btn",
                                )

                            gr.Markdown(
                                "**Examples** — load a known HBB variant:",
                                elem_classes=["sec-label"],
                            )
                            with gr.Row():
                                example_dropdown = gr.Dropdown(
                                    choices=[
                                        "Normal HBB (Wild-type)",
                                        "Sickle Cell (HbS E6V)",
                                        "HbC Disease (E6K)",
                                        "Random Benign Variant",
                                    ],
                                    label="Select Example",
                                    elem_id="example_dropdown",
                                )
                                load_btn = gr.Button(
                                    "Load",
                                    variant="secondary",
                                    elem_id="load_btn",
                                )

                        # Right pane — output
                        with gr.Column(scale=2, min_width=280):
                            prediction_output = gr.Markdown(
                                value="_Enter a sequence above and click **Analyze** to see results._",
                                label="Prediction Results",
                                elem_id="prediction_output",
                            )

                    # Visualisation rows
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown(
                                "### 🧩  Nucleotide Importance & Bloom Filter Hits"
                            )
                            importance_plot = gr.Plot(
                                label="Importance Analysis",
                                elem_id="importance_plot",
                            )

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown(
                                "### 📊  Comprehensive Analysis Dashboard"
                            )
                            dashboard_plot = gr.Plot(
                                label="Full Dashboard",
                                elem_id="dashboard_plot",
                            )

                    # Note: Event wiring moved below the Pipeline Trace tab so trace_output is defined

                # ── Tab 1.5 · Pipeline Trace ──────────────────────────────────────
                with gr.Tab("🧪  Pipeline Trace"):
                    gr.Markdown("### Step-by-Step Processing Log")
                    trace_output = gr.Markdown(
                        value="_Run an analysis in the **Analyze Sequence** tab to see the pipeline trace._",
                        elem_id="trace_output",
                        elem_classes=["output-panel"]
                    )
                    
                # ── Event Wiring ──────────────────────────────────────────
                analyze_btn.click(
                    fn=self.analyze_sequence,
                    inputs=[sequence_input],
                    outputs=[prediction_output, importance_plot, dashboard_plot, trace_output],
                )
                load_btn.click(
                    fn=self.analyze_example,
                    inputs=[example_dropdown],
                    outputs=[sequence_input],
                )

                # ── Tab 2 · Train ────────────────────────────────────────
                with gr.Tab("⚙️  Train Model"):
                    gr.HTML(TRAIN_HEADER)

                    gr.Markdown("""
Choose an architecture to train. See the comparison below:

| | Baseline | BGPCA (Novel) |
|:--|:--|:--|
| Bloom features | 18-dim summary | Per-position signal |
| DNABERT features | Pooled 768-dim | Per-token hidden states |
| Fusion | Concatenation | Cross-attention + gating |
| Position info | Lost | Preserved |
| Uncertainty | — | MC Dropout |
| Interpretability | Basic | Position importance + gate |
""")

                    with gr.Row():
                        model_choice = gr.Radio(
                            choices=[
                                "Baseline (Concatenation + MLP)",
                                "BGPCA (Novel Cross-Attention)",
                            ],
                            value="BGPCA (Novel Cross-Attention)",
                            label="Architecture",
                            elem_id="model_choice",
                            elem_classes=["radio-group"],
                        )

                    with gr.Row():
                        epochs_slider = gr.Slider(
                            minimum=10, maximum=100, value=30, step=5,
                            label="Training Epochs",
                            elem_id="epochs_slider",
                        )

                    train_btn = gr.Button(
                        "🚀  Start Training",
                        variant="primary",
                        size="lg",
                        elem_id="train_btn",
                    )

                    training_output = gr.Markdown(
                        value="_Configure settings above and click **Start Training**._",
                        label="Training Results",
                        elem_id="training_output",
                    )

                    train_btn.click(
                        fn=self.train_model,
                        inputs=[model_choice, epochs_slider],
                        outputs=[training_output],
                    )

                # ── Tab 3 · About ────────────────────────────────────────
                with gr.Tab("📖  About"):
                    gr.Markdown("""
## About BloomDNABERT

### Novel Contribution: BGPCA Architecture

**Bloom-Guided Positional Cross-Attention (BGPCA)** bridges probabilistic data structures (Bloom filters)
with neural attention mechanisms (transformers) through position-aware cross-modal attention.

---

### The Problem

Existing hybrid approaches simply concatenate features from different sources,
destroying spatial correspondence. A Bloom filter knows **exactly** where pathogenic k-mer hits
occur — but this positional information is lost when compressed to summary statistics.

### The Solution

BGPCA preserves per-position Bloom filter signals and uses them as additive attention biases
in a cross-attention mechanism with DNABERT's per-token hidden states.

---

### Architecture Components

1. **Positional Bloom Encoder** — multi-scale 1D CNN over raw Bloom activation signals.
2. **Bloom-Guided Cross-Attention**
   - Q = DNABERT tokens · K = Bloom positional encodings · V = DNABERT tokens
   - Bias = Bloom activation magnitude (structural prior)
   - `Attn(Q,K,V;B) = softmax(QKᵀ/√d + φ(B)) V`
3. **Mutation-Aware Pooling** — position-wise importance guided by Bloom activations.
4. **Gated Cross-Modal Fusion** — dynamically balances Bloom pattern-matching vs. DNABERT.
5. **Monte Carlo Dropout** — epistemic uncertainty for clinical safety.

---

### Scientific Background

| Property | Value |
|:---------|:------|
| Gene | HBB |
| Mutation | Codon 6, GAG → GTG |
| Effect | Glutamic acid → Valine (E6V) |
| Result | HbS — sickle cell disease |

---

### References

- DNABERT-2: Zhou et al. (2023)
- Bloom Filters in Bioinformatics: Solomon & Kingsford (2016)
- ALiBi (learned attention biases): Press et al. (2022)
- Perceiver (cross-attention): Jaegle et al. (2021)
- MC Dropout Uncertainty: Gal & Ghahramani (2016)
- ClinVar: NCBI database of genetic variants
""")

            gr.HTML(FOOTER_HTML)

        return interface

    def launch(self, **kwargs):
        """Launch the dashboard."""
        interface = self.create_interface()
        # Gradio 6+: theme and css are passed to launch(), not Blocks()
        interface.launch(
            theme=getattr(interface, '_theme', None),
            css=getattr(interface, '_css', None),
            **kwargs
        )


def _find_free_port(start: int = 7860, end: int = 7870) -> int:
    """Find first available port in [start, end)."""
    for port in range(start, end):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    return start


def main():
    """Main entry point for the dashboard."""
    print("\n" + "=" * 60)
    print("BloomDNABERT Variant Classifier")
    print("Bloom-Guided Positional Cross-Attention (BGPCA)")
    print("=" * 60 + "\n")

    port = _find_free_port()
    if port != 7860:
        print(f"Port 7860 in use; using port {port} instead.\n")

    dashboard = VariantAnalysisDashboard()
    dashboard.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=port,
    )


if __name__ == "__main__":
    main()
