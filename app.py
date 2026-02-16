"""
Gradio Web Dashboard for Bloom-Enhanced DNABERT Variant Classifier

Interactive web interface for analyzing DNA sequences for pathogenic variants
with real-time visualization of attention patterns and Bloom filter hits.

Supports two architectures:
1. Baseline: Simple concatenation + MLP (HybridClassifier)
2. BGPCA: Bloom-Guided Positional Cross-Attention (novel)
"""

import gradio as gr
import pandas as pd
import numpy as np
from pathlib import Path
import sys

from bloom_dnabert import MultiScaleBloomFilter, DNABERTWrapper, HybridClassifier, AttentionVisualizer
from bloom_dnabert.classifier import HybridClassifierPipeline, BloomGuidedPipeline
from bloom_dnabert.data_loader import ClinVarDataLoader


class VariantAnalysisDashboard:
    """
    Interactive web dashboard for variant analysis.
    """

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
            return f"### Training Failed\n\nError: {str(e)}\n\nCheck console output for details."

        arch_info = ""
        if use_bgpca:
            arch_info = """
        **Architecture: BGPCA (Novel)**
        - Positional Bloom Encoder (multi-scale 1D CNN)
        - Bloom-Guided Cross-Attention (2 layers, 4 heads)
        - Mutation-Aware Pooling
        - Gated Cross-Modal Fusion
        - Monte Carlo Dropout Uncertainty
        """
        else:
            arch_info = """
        **Architecture: Baseline**
        - Bloom features (18-dim) + DNABERT embedding (768-dim)
        - Simple concatenation + 2-layer MLP
        """

        results = f"""
        ### Training Complete! ({model_name})

        {arch_info}

        **Held-Out Test Set Performance (no leakage):**
        - Accuracy: {metrics['accuracy']:.3f}
        - Precision: {metrics['precision']:.3f}
        - Recall: {metrics['recall']:.3f}
        - F1 Score: {metrics['f1_score']:.3f}
        - AUC-ROC: {metrics['auc_roc']:.3f}

        **Data Split:** 60% train / 20% val / 20% test (stratified)

        **Training History:**
        - Final Training Loss: {history['train_loss'][-1]:.4f}
        - Final Training Accuracy: {history['train_acc'][-1]:.3f}
        - Final Validation Loss: {history['val_loss'][-1]:.4f}
        - Final Validation Accuracy: {history['val_acc'][-1]:.3f}
        """

        return results

    def analyze_sequence(self, sequence: str):
        """Analyze a DNA sequence."""
        if not sequence or len(sequence) < 10:
            return "Please enter a valid DNA sequence (at least 10 nucleotides)", None, None

        sequence = sequence.upper().strip()
        if not all(base in 'ATCGN' for base in sequence):
            return "Invalid sequence: Only A, T, C, G, N are allowed", None, None

        try:
            if self.trained and self.active_pipeline is not None:
                is_bgpca = isinstance(self.active_pipeline, BloomGuidedPipeline)

                if is_bgpca:
                    result = self.active_pipeline.predict_with_uncertainty(sequence)
                    interp = self.active_pipeline.predict_with_interpretability(sequence)

                    prediction_text = f"""
                    ### Prediction Results ({self.active_model_name})

                    **Classification:** {result['prediction']}
                    **Probability:** {result['probability']:.3f}
                    **Confidence:** {result['confidence']:.3f}

                    ---

                    **Uncertainty Estimation (MC Dropout):**
                    **Epistemic Uncertainty:** {result['uncertainty']:.4f}
                    **Uncertainty Level:** {result['uncertainty_level']}

                    ---

                    **Cross-Modal Fusion Gate:**
                    Mean gate value: {interp['gate_values'].mean():.3f}
                    (>0.5 = trusts DNABERT more, <0.5 = trusts Bloom more)

                    ---

                    **Interpretation:**
                    - BGPCA uses position-aware cross-attention between
                      Bloom filter signals and DNABERT hidden states
                    - Positions with high importance are shown in the plots below
                    - Uncertainty helps assess prediction reliability
                    """
                else:
                    result = self.active_pipeline.predict(sequence)
                    interp = None
                    prediction_text = f"""
                    ### Prediction Results ({self.active_model_name})

                    **Classification:** {result['prediction']}
                    **Probability:** {result['probability']:.3f}
                    **Confidence:** {result['confidence']:.3f}

                    ---

                    **Interpretation:**
                    - A probability > 0.5 indicates a pathogenic variant
                    - This model is trained specifically for HBB gene variants
                    """
            else:
                result = None
                interp = None
                prediction_text = """
                ### Model Not Trained

                Please train the model first using the "Train Model" tab.
                For now, showing Bloom filter and attention analysis only.
                """

            importance_plot = self.visualizer.create_nucleotide_importance_plot(
                sequence, show_bloom_hits=True
            )

            dashboard_plot = self.visualizer.create_dashboard(
                sequence, prediction_result=result
            )

            return prediction_text, importance_plot, dashboard_plot

        except Exception as e:
            return f"Error during analysis: {str(e)}", None, None

    def analyze_example(self, example_name: str):
        """Analyze a pre-defined example sequence."""
        examples = {
            "Normal HBB (Wild-type)": "CACGTGGACTACCCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGTTGGTATCAAGGTTACAAGACAGGTTTAAGGAGACCAATAGAAACTGGGCATGTGGAGACAGAGAAGACTCTTGGGTTTCTGATAGGCACTGACTCTCTCTGCCTATTGGTCTATTTTCCCACCCTTAGG",
            "Sickle Cell (HbS E6V)": "CACGTGGTCTACCCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGTTGGTATCAAGGTTACAAGACAGGTTTAAGGAGACCAATAGAAACTGGGCATGTGGAGACAGAGAAGACTCTTGGGTTTCTGATAGGCACTGACTCTCTCTGCCTATTGGTCTATTTTCCCACCCTTAGG",
            "HbC Disease (E6K)": "CACGTGAAGTACCCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGTTGGTATCAAGGTTACAAGACAGGTTTAAGGAGACCAATAGAAACTGGGCATGTGGAGACAGAGAAGACTCTTGGGTTTCTGATAGGCACTGACTCTCTCTGCCTATTGGTCTATTTTCCCACCCTTAGG",
            "Random Benign Variant": "CACGTGGACTACCCCTGAGGAGAAGTCTGCCGTTACTACCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGTTGGTATCAAGGTTACAAGACAGGTTTAAGGAGACCAATAGAAACTGGGCATGTGGAGACAGAGAAGACTCTTGGGTTTCTGATAGGCACTGACTCTCTCTGCCTATTGGTCTATTTTCCCACCCTTAGG"
        }

        return examples.get(example_name, "")

    def create_interface(self):
        """Create the Gradio interface."""
        with gr.Blocks(title="Bloom-Enhanced DNABERT Variant Classifier", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # Bloom-Enhanced DNABERT for Sickle Cell Variant Classification

            A novel hybrid system combining **Bloom filters** for fast pathogenic k-mer detection
            with **DNABERT-2** embeddings, featuring the **Bloom-Guided Positional Cross-Attention
            (BGPCA)** architecture for position-aware cross-modal fusion.

            ### Key Innovations:
            - **BGPCA Architecture**: Cross-attention where Bloom filter signals serve as attention biases
            - **Mutation-Aware Pooling**: Bloom-guided position importance weighting
            - **Gated Fusion**: Dynamically balances pattern matching vs contextual understanding
            - **Uncertainty Estimation**: Monte Carlo dropout for epistemic uncertainty
            """)

            with gr.Tabs():
                with gr.Tab("Analyze Sequence"):
                    gr.Markdown("### Enter a DNA sequence to analyze for pathogenic variants")

                    with gr.Row():
                        with gr.Column(scale=2):
                            sequence_input = gr.Textbox(
                                label="DNA Sequence",
                                placeholder="Enter DNA sequence (A, T, C, G)...",
                                lines=4,
                                max_lines=10
                            )

                            with gr.Row():
                                analyze_btn = gr.Button("Analyze Sequence", variant="primary", size="lg")
                                clear_btn = gr.ClearButton([sequence_input], value="Clear")

                            gr.Markdown("### Example Sequences")
                            example_dropdown = gr.Dropdown(
                                choices=[
                                    "Normal HBB (Wild-type)",
                                    "Sickle Cell (HbS E6V)",
                                    "HbC Disease (E6K)",
                                    "Random Benign Variant"
                                ],
                                label="Load Example"
                            )
                            load_example_btn = gr.Button("Load Example")

                        with gr.Column(scale=1):
                            prediction_output = gr.Markdown(label="Prediction Results")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Nucleotide Importance & Bloom Filter Hits")
                            importance_plot = gr.Plot(label="Importance Analysis")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Comprehensive Analysis Dashboard")
                            dashboard_plot = gr.Plot(label="Dashboard")

                    analyze_btn.click(
                        fn=self.analyze_sequence,
                        inputs=[sequence_input],
                        outputs=[prediction_output, importance_plot, dashboard_plot]
                    )

                    load_example_btn.click(
                        fn=self.analyze_example,
                        inputs=[example_dropdown],
                        outputs=[sequence_input]
                    )

                with gr.Tab("Train Model"):
                    gr.Markdown("""
                    ### Train the Variant Classifier

                    Choose between two architectures:

                    **Baseline**: Simple concatenation of Bloom features + DNABERT embeddings, fed to MLP.

                    **BGPCA (Novel)**: Bloom-Guided Positional Cross-Attention -- preserves
                    positional correspondence between Bloom hits and DNABERT token representations,
                    using cross-attention with Bloom-derived attention biases. Includes uncertainty
                    estimation via Monte Carlo dropout.
                    """)

                    with gr.Row():
                        model_choice = gr.Radio(
                            choices=[
                                "Baseline (Concatenation + MLP)",
                                "BGPCA (Novel Cross-Attention)"
                            ],
                            value="BGPCA (Novel Cross-Attention)",
                            label="Architecture"
                        )

                    with gr.Row():
                        epochs_slider = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=30,
                            step=5,
                            label="Training Epochs"
                        )

                    train_btn = gr.Button("Train Model", variant="primary", size="lg")
                    training_output = gr.Markdown(label="Training Results")

                    train_btn.click(
                        fn=self.train_model,
                        inputs=[model_choice, epochs_slider],
                        outputs=[training_output]
                    )

                with gr.Tab("About"):
                    gr.Markdown("""
                    ## About This System

                    ### Novel Contribution: BGPCA Architecture

                    **Bloom-Guided Positional Cross-Attention (BGPCA)** is a novel architecture
                    that bridges probabilistic data structures (Bloom filters) with neural
                    attention mechanisms (transformers) through position-aware cross-modal attention.

                    **The Problem**: Existing hybrid approaches simply concatenate features from
                    different sources, destroying spatial correspondence and treating each modality
                    independently. A Bloom filter knows EXACTLY where pathogenic k-mer hits occur,
                    but this positional information is lost when compressed to summary statistics.

                    **The Solution**: BGPCA preserves per-position Bloom filter signals and uses them
                    as additive attention biases in a cross-attention mechanism with DNABERT's
                    per-token hidden states.

                    ### Architecture Components

                    1. **Positional Bloom Encoder**
                       Multi-scale 1D convolutions encode raw per-position Bloom filter
                       activation signals into dense embeddings, capturing local hit patterns.

                    2. **Bloom-Guided Cross-Attention**
                       Novel cross-attention where:
                       - Q (queries) = DNABERT token representations (what to ask)
                       - K (keys) = Bloom positional encodings (where to look)
                       - V (values) = DNABERT token representations (what to extract)
                       - Bias = Bloom activation magnitude (structural prior)

                       `Attn(Q, K, V; B) = softmax(QK^T/sqrt(d) + phi(B)) V`

                    3. **Mutation-Aware Pooling**
                       Learns position-wise importance weights guided by Bloom activation,
                       naturally focusing on mutation-relevant regions.

                    4. **Gated Cross-Modal Fusion**
                       Dynamically balances trust between Bloom pattern matching
                       (known variants) and DNABERT understanding (novel variants).

                    5. **Monte Carlo Dropout Uncertainty**
                       Multiple stochastic forward passes estimate epistemic uncertainty,
                       critical for clinical reliability.

                    ### Why This Is Novel

                    - No prior work uses Bloom filter outputs as attention biases in transformers
                    - Bridges O(1) probabilistic pattern matching with O(n^2) neural attention
                    - Preserves spatial correspondence between modalities
                    - Provides interpretable attention maps showing Bloom-guided focus
                    - Quantifies prediction uncertainty for clinical safety

                    ### Comparison: Baseline vs BGPCA

                    | Feature | Baseline | BGPCA |
                    |---------|----------|-------|
                    | Bloom features | 18-dim summary | Per-position signal |
                    | DNABERT features | Pooled 768-dim | Per-token hidden states |
                    | Fusion | Concatenation | Cross-attention + gating |
                    | Position info | Lost | Preserved |
                    | Uncertainty | No | MC Dropout |
                    | Interpretability | Basic | Position importance + gate values |

                    ### Scientific Background

                    The sickle cell mutation is a point mutation in the HBB gene:
                    - Position: Codon 6
                    - Change: GAG -> GTG (E6V)
                    - Effect: Glutamic acid -> Valine
                    - Result: Abnormal hemoglobin (HbS) causing red blood cells to sickle

                    ### References

                    - DNABERT-2: Zhou et al. (2023)
                    - Bloom Filters in Bioinformatics: Solomon & Kingsford (2016)
                    - ALiBi (learned attention biases): Press et al. (2022)
                    - Perceiver (cross-attention): Jaegle et al. (2021)
                    - MC Dropout Uncertainty: Gal & Ghahramani (2016)
                    - ClinVar: NCBI database of genetic variants

                    ### Citation

                    ```
                    Bloom-Guided Positional Cross-Attention for DNA Variant Classification (2026)
                    A novel architecture bridging probabilistic data structures with neural
                    attention mechanisms for position-aware cross-modal fusion.
                    ```
                    """)

            gr.Markdown("""
            ---
            **Note**: This is a research prototype. Consult with healthcare professionals
            for clinical genetic interpretation.
            """)

        return interface

    def launch(self, **kwargs):
        """Launch the dashboard."""
        interface = self.create_interface()
        interface.launch(**kwargs)


def main():
    """Main entry point for the dashboard."""
    print("\n" + "=" * 60)
    print("Bloom-Enhanced DNABERT Variant Classifier")
    print("with Bloom-Guided Positional Cross-Attention (BGPCA)")
    print("=" * 60 + "\n")

    dashboard = VariantAnalysisDashboard()
    dashboard.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )


if __name__ == "__main__":
    main()
