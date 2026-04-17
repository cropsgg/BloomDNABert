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

import gradio as gr
import pandas as pd
import numpy as np
from pathlib import Path
import socket

from bloom_seq import list_all_plugins
from bloom_seq.alphabets import DNA_ALPHABET
from bloom_seq.errors import DataSourceError
from bloom_seq.registry import backbones as backbone_plugins
from bloom_seq.registry import data_sources as data_source_plugins
from bloom_seq.registry import pattern_indexes as pattern_index_plugins
from bloom_seq.pipeline import BloomGuidedPipeline, HybridClassifierPipeline
from bloom_seq.viz import AttentionVisualizer


def _default_plugin_id(group: list[str], preferred: str) -> str:
    if preferred in group:
        return preferred
    return group[0] if group else preferred


def _plugin_registry_markdown() -> str:
    lines = ["**Entry-point plugins** (extend via `pyproject.toml` — see PLUGINS.md):", ""]
    for group, names in sorted(list_all_plugins().items()):
        lines.append(f"- **{group}**: {', '.join(names) if names else '—'}")
    return "\n".join(lines)


# ─── Design System CSS ────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Design tokens ── */
:root {
    --bg-base:        #0B1121;
    --bg-surface:     #1E293B;
    --bg-elevated:    #334155;
    --bg-input:       #0F172A;
    --border-subtle:  #334155;
    --border-default: #475569;
    --border-focus:   #60A5FA;
    --blue:           #3B82F6;
    --cyan:           #38BDF8;
    --green:          #34D399;
    --purple:         #8B5CF6;
    --amber:          #FBBF24;
    --red:            #F87171;
    --text-1:         #F8FAFC;
    --text-2:         #94A3B8;
    --text-3:         #64748B;
    --shadow-card:    0 10px 15px -3px rgba(0, 0, 0, 0.4), 0 4px 6px -2px rgba(0, 0, 0, 0.2);
    --shadow-hover:   0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 10px 10px -5px rgba(0, 0, 0, 0.3);
    --radius-xl:      12px;
    --radius-lg:      8px;
    --radius-md:      6px;
    --ease:           cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Page shell ── */
body, .gradio-container {
    font-family: 'Inter', system-ui, sans-serif !important;
    background: var(--bg-base) !important;
    background-image: none !important;
    color: var(--text-1) !important;
    font-weight: 400 !important;
}
footer { display: none !important; }

/* ── Sidebar / topnav strip ── */
#navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 2rem;
    background: #0F172A;
    border-bottom: 1px solid var(--border-subtle);
    position: sticky; top: 0; z-index: 100;
}
#navbar .brand {
    display: flex; align-items: center; gap: 0.6rem;
    font-size: 1.1rem; font-weight: 700; letter-spacing: -0.3px;
    color: var(--text-1);
}
#navbar .brand .hex { 
    color: var(--blue); 
    font-size: 1.3rem; 
}
#navbar .nav-pills { display: flex; gap: 0.5rem; }
.nav-pill {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.5px;
    text-transform: uppercase; padding: 0.3rem 0.8rem;
    border-radius: 999px;
    background: var(--bg-surface);
    color: var(--text-2);
    border: 1px solid var(--border-subtle);
}

/* ── Hero ── */
#hero {
    padding: 2.5rem 2rem 2rem;
    background: #0F172A;
    border-bottom: 1px solid var(--border-subtle);
    text-align: center;
}
#hero h1 {
    font-size: 2.2rem; font-weight: 700; letter-spacing: -0.8px; line-height: 1.2;
    margin: 0 auto 0.8rem;
    color: var(--text-1);
}
#hero p {
    font-size: 1.05rem; color: var(--text-2); line-height: 1.6;
    max-width: 800px; margin: 0 auto;
}
#hero strong { color: var(--text-1); font-weight: 600; }

/* ── Tabs ── */
.tabs > .tab-nav {
    background: #0F172A !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    padding: 0 1.5rem !important; gap: 0.5rem !important;
    margin-top: 0 !important;
}
.tabs > .tab-nav button {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important; font-weight: 600 !important;
    color: var(--text-2) !important;
    padding: 0.8rem 1rem !important;
    border: none !important; border-bottom: 2px solid transparent !important;
    border-radius: 0 !important; background: transparent !important;
    transition: all 0.2s var(--ease) !important;
}
.tabs > .tab-nav button:hover { color: var(--text-1) !important; }
.tabs > .tab-nav button.selected {
    color: var(--blue) !important;
    border-bottom-color: var(--blue) !important;
}

/* ── Section titles inside tabs ── */
.sec-label {
    font-size: 0.75rem; font-weight: 600; letter-spacing: 1px;
    text-transform: uppercase; color: var(--text-3);
    margin: 0 0 0.4rem;
    display: inline-block;
}
.sec-title {
    font-size: 1.2rem; font-weight: 600; color: var(--text-1);
    margin: 0 0 1.5rem;
    letter-spacing: -0.3px;
}

/* ── Glass card ── */
.card, .output-panel, .gradio-plot, [data-testid="plot"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-xl) !important;
    box-shadow: var(--shadow-card) !important;
    transition: all 0.2s var(--ease) !important;
}
.card:hover, .output-panel:hover {
    box-shadow: var(--shadow-hover) !important;
}
.card { padding: 1.5rem !important; }

/* ── Markdown & Prose Panels ── */
.output-panel, .prose, .md {
    padding: 1.5rem !important;
    color: var(--text-1) !important;
    font-size: 0.9rem !important; line-height: 1.6 !important;
}
.output-panel h3, .prose h3, .md h3 {
    font-size: 1rem !important; font-weight: 600 !important;
    color: var(--cyan) !important; /* cyan for headers */
    border-bottom: 1px solid var(--border-subtle) !important;
    padding-bottom: 0.5rem !important; margin-bottom: 1rem !important;
}
.output-panel strong, .prose strong, .md strong {
    color: var(--text-1) !important; font-weight: 600 !important;
}
.output-panel code, .prose code, .md code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important; padding: 0.15em 0.4em !important;
    border-radius: 4px !important;
    background: var(--bg-input) !important;
    border: 1px solid var(--border-subtle) !important;
    color: var(--cyan) !important;
}
.output-panel hr, .prose hr, .md hr {
    border-top: 1px solid var(--border-subtle) !important;
    margin: 1.5rem 0 !important;
}
.output-panel table, .prose table, .md table {
    width: 100%; border-collapse: separate !important; border-spacing: 0 !important;
    font-size: 0.85rem !important; margin: 1rem 0;
    border-radius: var(--radius-lg); overflow: hidden;
    border: 1px solid var(--border-subtle);
}
.output-panel th, .prose th, .md th {
    background: var(--bg-elevated) !important;
    color: var(--text-2) !important; font-weight: 600 !important;
    padding: 0.6rem 1rem !important;
    text-align: left;
    border-bottom: 1px solid var(--border-subtle) !important;
}
.output-panel td, .prose td, .md td {
    padding: 0.6rem 1rem !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    color: var(--text-1) !important;
}

/* ── Inputs ── */
label, .label-wrap span {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.75rem !important; font-weight: 600 !important;
    color: var(--text-2) !important;
}
textarea, input[type="text"] {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-lg) !important;
    color: var(--text-1) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important; line-height: 1.6 !important;
    padding: 0.8rem !important;
    transition: all 0.2s var(--ease) !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
    outline: none !important;
}
textarea::placeholder { color: var(--text-3) !important; font-family: 'Inter', sans-serif !important; }

/* ── Buttons ── */
button[variant="primary"], .gr-button-primary {
    background: var(--blue) !important;
    border: none !important; color: #fff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important; font-weight: 500 !important;
    border-radius: var(--radius-lg) !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s var(--ease) !important;
}
button[variant="primary"]:hover {
    background: #1D4ED8 !important;
}

button[variant="secondary"], .gr-button-secondary {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-default) !important;
    color: var(--text-2) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important; font-weight: 500 !important;
    border-radius: var(--radius-lg) !important;
    transition: all 0.2s var(--ease) !important;
}
button[variant="secondary"]:hover {
    background: var(--bg-base) !important;
    color: var(--text-1) !important;
}

/* ── Dropdown ── */
select, .wrap select, .wrap .gr-box {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-default) !important;
    color: var(--text-1) !important;
    border-radius: var(--radius-lg) !important;
}

/* ── Slider ── */
input[type="range"] { accent-color: var(--blue) !important; }

/* ── Radio ── */
.radio-group label {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-lg) !important;
    padding: 0.6rem 1rem !important;
    cursor: pointer !important;
    font-size: 0.9rem !important; font-weight: 500 !important;
    color: var(--text-2) !important;
}
.radio-group label:hover {
    background: var(--bg-base) !important;
}
.radio-group label:has(input:checked) {
    border-color: var(--blue) !important;
    background: rgba(59, 130, 246, 0.05) !important;
    color: var(--blue) !important;
}
"""

# ─── Static HTML blocks ───────────────────────────────────────────────────────

NAVBAR_HTML = """
<div id="navbar">
  <div class="brand">
    <span class="hex">⬡</span>
    <span>BloomDNABERT</span>
  </div>
  <div class="nav-pills">
    <span class="nav-pill">Pattern index</span>
    <span class="nav-pill">Sequence backbone</span>
    <span class="nav-pill">BGPCA</span>
  </div>
</div>
"""

HERO_HTML = """
<div id="hero">
  <h1>Variant Classification Dashboard</h1>
  <p>
    <strong>bloom_seq</strong> dashboard: fast approximate <strong>k-mer / pattern indexes</strong> plus a
    <strong>sequence backbone</strong> (default: DNABERT-2), with optional
    <strong>Bloom-Guided Positional Cross-Attention (BGPCA)</strong>. Register alternate DNA/RNA/protein
    models via Python entry points (see PLUGINS.md).
  </p>
</div>
"""

ANALYZE_HEADER = """
<div style="padding:0.5rem 0 1rem;">
  <div class="sec-label">Inference Target</div>
  <h2 class="sec-title">Supply a DNA sequence for real-time analysis (reference task: HBB / ClinVar)</h2>
</div>
"""

TRAIN_HEADER = """
<div style="padding:0.5rem 0 1rem;">
  <div class="sec-label">Model Configuration</div>
  <h2 class="sec-title">Architect and initialize your neural network training parameters</h2>
</div>
"""

FOOTER_HTML = """
<div style="text-align:center;padding:1.5rem 0;border-top:1px solid #334155;margin-top:2rem">
  <span style="font-size:0.8rem;color:#64748B;">
    BloomDNABert &nbsp;·&nbsp; open-source research dashboard (BGPCA + baseline)
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
        reg = list_all_plugins()
        self._plugin_backbone_id = _default_plugin_id(reg.get("backbones", []), "dnabert2")
        self._plugin_index_id = _default_plugin_id(reg.get("pattern_indexes", []), "multiscale_bloom")
        self._data_source_id = _default_plugin_id(reg.get("data_sources", []), "clinvar_hbb")

        print("Initializing Bloom Seq dashboard (plugin registry)...")
        self._rebuild_ml_stack(self._plugin_backbone_id, self._plugin_index_id)

    def _rebuild_ml_stack(self, backbone_id: str, index_id: str) -> None:
        """Load pattern index + backbone from registry entry points."""
        # Pipelines hold references to the previous backbone/index; a reload must
        # invalidate any trained weights or inference becomes silently inconsistent.
        self.trained = False
        self.active_pipeline = None
        self.active_model_name = None

        try:
            print(f"Loading pattern index plugin {index_id!r}...")
            IndexCls = pattern_index_plugins.get_class(index_id)
            self.bloom_filter = IndexCls(capacity=100000, error_rate=0.001)
            if hasattr(self.bloom_filter, "load_hbb_pathogenic_variants"):
                self.bloom_filter.load_hbb_pathogenic_variants()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize pattern index {index_id!r}: {e}") from e

        try:
            print(f"Loading backbone plugin {backbone_id!r}...")
            BackboneCls = backbone_plugins.get_class(backbone_id)
            self.dnabert_wrapper = BackboneCls()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load backbone {backbone_id!r}: {e}. "
                "Ensure `[dl]` extras and model weights are available."
            ) from e

        self.visualizer = AttentionVisualizer(self.dnabert_wrapper, self.bloom_filter)

        self.baseline_pipeline = HybridClassifierPipeline(
            bloom_filter=self.bloom_filter,
            dnabert_wrapper=self.dnabert_wrapper,
        )

        self.bgpca_pipeline = BloomGuidedPipeline(
            bloom_filter=self.bloom_filter,
            dnabert_wrapper=self.dnabert_wrapper,
        )

        self._plugin_backbone_id = backbone_id
        self._plugin_index_id = index_id
        print("Dashboard ML stack ready.")

    def train_model(
        self,
        model_choice: str,
        epochs: int,
        backbone_id: str,
        index_id: str,
        data_source_id: str,
        progress=gr.Progress(),
    ):
        """Train the selected classifier model."""
        try:
            progress(0, desc="Loading data...")

            if backbone_id != self._plugin_backbone_id or index_id != self._plugin_index_id:
                self._rebuild_ml_stack(backbone_id, index_id)

            SourceCls = data_source_plugins.get_class(data_source_id)
            data_source = SourceCls()
            train_df, val_df, test_df = data_source.get_training_splits()
            self._data_source_id = data_source_id

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
        except DataSourceError as e:
            return (
                "### Training data required\n\n"
                f"{e}\n\n"
                "See **DATASETS.md** in the repository for how to build or download CSVs.",
                gr.update(),
            )
        except Exception as e:
            return (
                f"### ❌  Training Failed\n\n**Error:** `{str(e)}`\n\nCheck console output for details.",
                gr.update(),
            )

        bf_dim = self.bloom_filter.feature_dim
        h_dim = self.dnabert_wrapper.hidden_size
        arch_note = (
            "- Positional Bloom Encoder (multi-scale 1D CNN)\n"
            "- Bloom-Guided Cross-Attention (2 layers, 4 heads)\n"
            "- Mutation-Aware Pooling\n"
            "- Gated Cross-Modal Fusion\n"
            "- Monte Carlo Dropout Uncertainty"
        ) if use_bgpca else (
            f"- Bloom features ({bf_dim}-dim) + backbone embedding ({h_dim}-dim)\n"
            "- Concatenation → 2-layer MLP"
        )

        last_train_loss = history['train_loss'][-1] if history['train_loss'] else 0.0
        last_train_acc  = history['train_acc'][-1]  if history['train_acc']  else 0.0
        last_val_loss   = history['val_loss'][-1]   if history['val_loss']   else 0.0
        last_val_acc    = history['val_acc'][-1]    if history['val_acc']    else 0.0

        return (
            f"""
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

> Switch to **Analyze Sequence** to classify variants from the trained data domain.
""",
            gr.update(choices=self._example_choice_labels()),
        )

    def analyze_sequence(self, sequence: str):
        """Analyze a DNA sequence."""
        if not sequence or len(sequence) < 10:
            return "Please enter a valid DNA sequence (at least 10 nucleotides).", None, None, "*Sequence too short for pipeline trace.*"

        sequence = sequence.upper().strip()
        _allowed_dna = set(
            (DNA_ALPHABET.symbols + DNA_ALPHABET.ambiguity).upper()
        )
        if not all(base in _allowed_dna for base in sequence):
            amb = DNA_ALPHABET.ambiguity or ""
            return (
                f"❌  Invalid sequence: allowed {DNA_ALPHABET.symbols!r}"
                f"{f' (+ ambiguity {amb!r})' if amb else ''}.",
                None,
                None,
                "*Invalid sequence — pipeline halted.*",
            )

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
                    sq = result.get("sequence_plausibility")
                    if sq:
                        prediction_text += f"""

### 📏  Sequence plausibility (statistical sanity check)

| Property | Value |
|:---------|------:|
| Genomic plausibility (0–1) | `{sq['genomic_plausibility_score']:.3f}` |
| P(statistically unlike human DNA) | `{sq['probability_statistically_spurious']:.1%}` |
| Base diversity (entropy) | `{sq['base_diversity_score']:.3f}` |

> This does **not** verify coordinates on GRCh38. It flags pasted or random strings whose k-mer usage or composition is unlike typical human genomic windows.
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
                    sq = result.get("sequence_plausibility")
                    if sq:
                        prediction_text += f"""

### 📏  Sequence plausibility (statistical sanity check)

| Property | Value |
|:---------|------:|
| Genomic plausibility (0–1) | `{sq['genomic_plausibility_score']:.3f}` |
| P(statistically unlike human DNA) | `{sq['probability_statistically_spurious']:.1%}` |
| Base diversity (entropy) | `{sq['base_diversity_score']:.3f}` |

> This does **not** verify coordinates on GRCh38. It flags pasted or random strings whose k-mer usage or composition is unlike typical human genomic windows.
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
        trace += f"### 1. Sequence Ingestion & Preprocessing\n"
        trace += f"**Action:** The raw DNA sequence is ingested into the system. It is converted to uppercase and validated.\n"
        trace += f"- **Length Analyzed**: `{len(sequence)}` base pairs\n"
        
        # 2. Bloom Filter
        trace += f"\n### 2. Multi-Scale Classical Bloom Filter Analysis\n"
        trace += f"**Action:** The sequence is concurrently scanned by multiple probabilistic exact-match filters tuned to detect known pathogenic sickle-cell and thalassemia k-mers in native O(1) time.\n"
        features = self.bloom_filter.get_hit_features(sequence)
        for k in self.bloom_filter.k_sizes:
            count = features.get(f'hit_count_k{k}', 0)
            ratio = features.get(f'hit_ratio_k{k}', 0)
            trace += f"- **K-mer Window (k={k})**: Exact match probabilistic scanner found `{count}` pathogenic patterns. This creates a spatial density feature map marking exact mutated coordinates (Density Ratio: `{ratio:.4f}`).\n"
        trace += f"- **Aggregated Spatial Density**: The mathematical mean of the feature maps across all window sizes is `{features.get('mean_hit_ratio', 0):.4f}`.\n"
        
        if not self.trained or self.active_pipeline is None:
            trace += "\n> ⚠️ *Pipeline Execution Halted: The neural architecture has not been initialized with trained weights.*"
            return trace

        # 3. DNABERT
        trace += f"\n### 3. Deep Contextual Feature Extraction (DNABERT-2)\n"
        trace += f"**Action:** The string sequence is pipelined into the large DNABERT-2 Transformer encoder, leveraging multi-layer contextual Self-Attention to translate simple nucleotides into a high-dimensional vector space capturing deep biological semantics.\n"
        try:
            tokens = self.dnabert_wrapper.tokenizer.tokenize(sequence)
            trace += f"- **Subword BPE Tokenization**: The contiguous sequence is algorithmically sectioned into `{len(tokens)}` specific genomic vocabulary tokens optimized by the BPE methodology.\n"
        except Exception:
            pass
        h = self.dnabert_wrapper.hidden_size
        if is_bgpca:
            trace += f"- **Dense Token Representation**: The DL engine maps the representation space, outputting full deep hidden states for every token `[num_tokens, {h}]`. Because this is unpooled, exact spatial layout of the genetic code is retained for the next block.\n"
        else:
            trace += f"- **Pooled Representation**: The engine compresses the entire sequence into a single 1D scalar `[1, {h}]`. Crucially, exact token spatial location geometry is destroyed in this process.\n"

        # 4. Fusion
        trace += f"\n### 4. Cross-Modal Cognitive Fusion ({self.active_model_name})\n"
        if is_bgpca:
            trace += f"**Action:** The BGPCA architecture reconciles the classical exact-match statistical data (Multi-scale Bloom maps) with the deep semantic representations (DNABERT) using a sophisticated trainable Cross-Attention block.\n"
            trace += f"- **Coordination Projection**: The O(1) positional coordinate maps from the Bloom Filters are smoothed and mathematically projected into sequence dimensionality via a 1D Convolutional filter.\n"
            trace += f"- **Guided Cross-Attention Topology**: The spatial Bloom signals act as Attention `Queries`, deliberately scanning and multiplying the DNABERT `Keys/Values` to radically prioritize contextual biological vectors that physically align with pathogenic algorithmic hits.\n"
            if interp is not None and 'gate_values' in interp:
                gate_mean = interp['gate_values'].mean()
                trace += f"- **Dynamic Arbitration Gate**: The network autonomously computes a scalar interpolation variable to govern trust between the classical and deep learning subsystems: `{gate_mean:.3f}`.\n"
                trace += f"  - *Decision Result: The network intrinsically decides to rely {'>' if gate_mean > 0.5 else '<'} 50% on deep contextual features (DNABERT) compared to the classical exact matches (Bloom).* \n"
        else:
            bf = self.bloom_filter.feature_dim
            trace += f"**Action:** The baseline system executes a simplified unweighted vector append operation.\n"
            trace += (
                f"- **Fixed Vector Concatenation**: The {bf}-dimensional numeric hit summary from "
                f"the Bloom filter is blindly appended end-to-end onto the {h}-dimensional "
                f"scalar output of the backbone encoder.\n"
            )
            trace += f"- **Information Bottleneck (MLP Layer)**: The resulting {bf + h}-dimensional flattened vector is passed through a basic 2-layer classifier. The model attempts to map the vector manually.\n"

        # 5. Output
        trace += f"\n### 5. Probabilistic Variant Classification\n"
        trace += f"**Action:** The fully synthesized representations are collapsed into a dual-class output node predicting the likelihood of the phenotype being pathogenic.\n"
        if result:
            trace += f"- **Raw Sigmoid Density Probability**: `{result['probability']:.5f}` (Scale 0.0 - 1.0)\n"
            trace += f"- **Binary Heuristic Threshold**: Evaluated against established boundaries to deduce final variant status: **`{result['prediction'].upper()}`**\n"
            if is_bgpca and 'uncertainty' in result:
                trace += f"- **Stochastic Calibration (MC Dropout)**: The network ran 10 simultaneous stochastic forward passes with probabilistic neural dropout active to quantify Epistemic structural variance.\n"
                trace += f"- **Uncertainty Rating**: Calculated standard deviation metric: `{result['uncertainty']:.5f}` | Confidence Status: **{result['uncertainty_level']}**\n"
                
        return trace

    def _example_choice_labels(self) -> list[str]:
        """Human-readable example names for the current data-source plugin."""
        try:
            Cls = data_source_plugins.get_class(self._data_source_id)
            ex_fn = getattr(Cls, "example_sequences", None)
            raw = list(ex_fn()) if callable(ex_fn) else []
        except Exception:
            return []
        return [e["label"] for e in raw if "label" in e]

    def analyze_example(self, example_name: str):
        """Return a pre-defined example sequence from the active data-source plugin."""
        try:
            Cls = data_source_plugins.get_class(self._data_source_id)
            ex_fn = getattr(Cls, "example_sequences", None)
            raw = list(ex_fn()) if callable(ex_fn) else []
        except Exception:
            raw = []
        examples = {e["label"]: e["sequence"] for e in raw if "label" in e and "sequence" in e}
        return examples.get(example_name, "")

    def create_interface(self):
        """Create the Gradio interface."""

        # Programmatic light theme via gr.themes.Base
        theme = gr.themes.Base(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.slate,
            neutral_hue=gr.themes.colors.gray,
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
            font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace"],
        ).set(
            body_background_fill              = "#0B1121",
            body_background_fill_dark         = "#0B1121",
            body_text_color                   = "#F8FAFC",
            body_text_color_dark              = "#F8FAFC",
            border_color_primary              = "#334155",
            border_color_primary_dark         = "#334155",
            block_background_fill             = "#1E293B",
            block_background_fill_dark        = "#1E293B",
            input_background_fill             = "#0F172A",
            input_background_fill_dark        = "#0F172A",
            button_primary_background_fill    = "#3B82F6",
            button_primary_background_fill_dark = "#3B82F6",
            button_primary_text_color         = "#ffffff",
            button_secondary_background_fill  = "#1E293B",
            button_secondary_text_color       = "#94A3B8",
        )

        with gr.Blocks(
            title="BloomDNABERT — Sickle Cell Variant Classifier",
            theme=theme,
            css=CUSTOM_CSS,
        ) as interface:

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
                                "**Examples** — from the active data-source plugin (updates after training):",
                                elem_classes=["sec-label"],
                            )
                            with gr.Row():
                                _ex_labels = self._example_choice_labels()
                                example_dropdown = gr.Dropdown(
                                    choices=_ex_labels,
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

                    gr.Markdown(_plugin_registry_markdown())

                    gr.Markdown(f"""
Choose an architecture to train. See the comparison below:

| | Baseline | BGPCA (Novel) |
|:--|:--|:--|
| Bloom features | {self.bloom_filter.feature_dim}-dim summary | Per-position signal |
| Backbone features | Pooled {self.dnabert_wrapper.hidden_size}-dim | Per-token hidden states |
| Fusion | Concatenation | Cross-attention + gating |
| Position info | Lost | Preserved |
| Uncertainty | — | MC Dropout |
| Interpretability | Basic | Position importance + gate |
""")

                    _reg = list_all_plugins()
                    with gr.Row():
                        backbone_dd = gr.Dropdown(
                            choices=_reg.get("backbones", []),
                            value=self._plugin_backbone_id,
                            label="Backbone (registry)",
                            elem_id="backbone_dd",
                        )
                        index_dd = gr.Dropdown(
                            choices=_reg.get("pattern_indexes", []),
                            value=self._plugin_index_id,
                            label="Pattern index",
                            elem_id="index_dd",
                        )
                        data_dd = gr.Dropdown(
                            choices=_reg.get("data_sources", []),
                            value=self._data_source_id,
                            label="Data source",
                            elem_id="data_dd",
                        )

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
                        inputs=[model_choice, epochs_slider, backbone_dd, index_dd, data_dd],
                        outputs=[training_output, example_dropdown],
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
        interface.launch(**kwargs)


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
        server_name="127.0.0.1",
        server_port=port,
        # Skip Gradio's HTTP probe of localhost (fails in some sandboxes / proxy setups).
        _frontend=False,
    )


if __name__ == "__main__":
    main()
