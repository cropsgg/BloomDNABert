"""
Gradio Web Dashboard for Bloom-Enhanced DNABERT Variant Classifier

Interactive web interface for analyzing DNA sequences for pathogenic variants
with real-time visualization of attention patterns and Bloom filter hits.
"""

import gradio as gr
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import our modules
from bloom_dnabert import MultiScaleBloomFilter, DNABERTWrapper, HybridClassifier, AttentionVisualizer
from bloom_dnabert.classifier import HybridClassifierPipeline
from bloom_dnabert.data_loader import ClinVarDataLoader


class VariantAnalysisDashboard:
    """
    Interactive web dashboard for variant analysis.
    """
    
    def __init__(self):
        """Initialize the dashboard with all components."""
        self.bloom_filter = None
        self.dnabert_wrapper = None
        self.classifier_pipeline = None
        self.visualizer = None
        self.trained = False
        
        print("Initializing Bloom-Enhanced DNABERT Dashboard...")
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all ML components."""
        # Initialize Bloom filter
        print("Loading Bloom filter...")
        self.bloom_filter = MultiScaleBloomFilter(capacity=100000, error_rate=0.001)
        self.bloom_filter.load_hbb_pathogenic_variants()
        
        # Initialize DNABERT
        print("Loading DNABERT-2 model...")
        self.dnabert_wrapper = DNABERTWrapper()
        
        # Initialize visualizer
        self.visualizer = AttentionVisualizer(self.dnabert_wrapper, self.bloom_filter)
        
        # Initialize classifier pipeline
        self.classifier_pipeline = HybridClassifierPipeline(
            bloom_filter=self.bloom_filter,
            dnabert_wrapper=self.dnabert_wrapper
        )
        
        print("✓ Dashboard initialized successfully!")
    
    def train_model(self, epochs: int = 30, progress=gr.Progress()):
        """Train the classifier model."""
        progress(0, desc="Loading data...")
        
        # Load data
        data_loader = ClinVarDataLoader()
        train_df, test_df = data_loader.get_training_data()
        
        progress(0.2, desc="Preparing datasets...")
        
        # Prepare data
        train_sequences = train_df['sequence'].tolist()
        train_labels = train_df['label'].tolist()
        test_sequences = test_df['sequence'].tolist()
        test_labels = test_df['label'].tolist()
        
        progress(0.3, desc="Training model...")
        
        # Train
        history = self.classifier_pipeline.train(
            train_sequences=train_sequences,
            train_labels=train_labels,
            val_sequences=test_sequences,
            val_labels=test_labels,
            epochs=epochs,
            batch_size=16
        )
        
        progress(0.9, desc="Evaluating model...")
        
        # Evaluate
        metrics = self.classifier_pipeline.evaluate(test_sequences, test_labels)
        
        self.trained = True
        
        progress(1.0, desc="Training complete!")
        
        # Format results
        results = f"""
        ### Training Complete! ✓
        
        **Test Set Performance:**
        - Accuracy: {metrics['accuracy']:.3f}
        - Precision: {metrics['precision']:.3f}
        - Recall: {metrics['recall']:.3f}
        - F1 Score: {metrics['f1_score']:.3f}
        - AUC-ROC: {metrics['auc_roc']:.3f}
        
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
        
        # Validate sequence
        sequence = sequence.upper().strip()
        if not all(base in 'ATCGN' for base in sequence):
            return "Invalid sequence: Only A, T, C, G, N are allowed", None, None
        
        try:
            # Get prediction
            if self.trained:
                result = self.classifier_pipeline.predict(sequence)
                prediction_text = f"""
                ### Prediction Results
                
                **Classification:** {result['prediction']}  
                **Probability:** {result['probability']:.3f}  
                **Confidence:** {result['confidence']:.3f}
                
                ---
                
                **Interpretation:**
                - A probability > 0.5 indicates a pathogenic variant
                - This model is trained specifically for HBB gene variants
                - The sickle cell mutation (E6V) is a known pathogenic variant
                """
            else:
                result = None
                prediction_text = """
                ### Model Not Trained
                
                Please train the model first using the "Train Model" tab.
                For now, showing Bloom filter and attention analysis only.
                """
            
            # Create visualizations
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
            # 🧬 Bloom-Enhanced DNABERT for Sickle Cell Variant Classification
            
            A novel hybrid system combining **Bloom filters** for fast pathogenic k-mer detection 
            with **DNABERT-2** embeddings for variant classification.
            
            ### Key Features:
            - **Multi-scale Bloom Filters**: Fast O(1) lookup of known pathogenic k-mers
            - **DNABERT-2 Attention**: Deep learning for sequence understanding
            - **Interpretable Heatmaps**: Visual explanation of model predictions
            - **Sickle Cell Focus**: Specifically trained for HBB gene variants
            """)
            
            with gr.Tabs():
                # Analysis Tab
                with gr.Tab("🔬 Analyze Sequence"):
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
                    
                    # Connect buttons
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
                
                # Training Tab
                with gr.Tab("🎓 Train Model"):
                    gr.Markdown("""
                    ### Train the Hybrid Classifier
                    
                    The model combines Bloom filter features with DNABERT embeddings.
                    Training on the synthetic HBB variant dataset (with sickle cell, HbC, and benign variants).
                    
                    **Note:** Training may take 5-10 minutes depending on your hardware.
                    """)
                    
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
                        inputs=[epochs_slider],
                        outputs=[training_output]
                    )
                
                # Information Tab
                with gr.Tab("ℹ️ About"):
                    gr.Markdown("""
                    ## About This System
                    
                    ### Novel Contributions
                    
                    1. **Multi-scale Bloom Filter Pre-screening**  
                       Uses Bloom filters at k=6, 8, 10 populated with known pathogenic k-mers 
                       from the HBB gene region for fast O(1) lookup.
                    
                    2. **Hybrid Feature Fusion**  
                       Combines probabilistic Bloom filter features with DNABERT-2 embeddings 
                       for robust classification.
                    
                    3. **Interpretable Attention Heatmaps**  
                       Extracts and visualizes attention weights to show which nucleotides 
                       the model focuses on, overlaid with Bloom filter hit positions.
                    
                    4. **Sickle Cell Mutation Detection**  
                       Specifically trained to detect the E6V mutation (GAG to GTG at codon 6) 
                       and other HBB pathogenic variants.
                    
                    ### Architecture
                    
                    - **Input**: DNA sequence
                    - **Bloom Filter Module**: Multi-scale k-mer matching (6, 8, 10)
                    - **DNABERT-2**: 117M parameter transformer encoder
                    - **Hybrid Classifier**: 2-layer MLP combining both feature types
                    - **Output**: Pathogenic/Benign classification + attention visualization
                    
                    ### Sickle Cell Disease
                    
                    The sickle cell mutation is a point mutation in the HBB gene:
                    - Position: Codon 6
                    - Change: GAG → GTG (E6V)
                    - Effect: Glutamic acid → Valine
                    - Result: Abnormal hemoglobin (HbS) causing red blood cells to sickle
                    
                    ### Performance
                    
                    On the synthetic test set:
                    - Fast pre-screening with Bloom filters (microseconds)
                    - High accuracy combining pattern matching + deep learning
                    - Interpretable results showing model reasoning
                    
                    ### References
                    
                    - DNABERT-2: Zhou et al. (2023) - [Hugging Face](https://huggingface.co/zhihan1996/DNABERT-2-117M)
                    - Bloom Filters in Bioinformatics: Solomon & Kingsford (2016)
                    - ClinVar: NCBI database of genetic variants
                    
                    ### Citation
                    
                    If you use this system in your research, please cite:
                    ```
                    Bloom-Enhanced DNABERT for Sickle Cell Variant Classification (2026)
                    A novel hybrid system combining Bloom filters and transformer models
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
    print("\n" + "="*60)
    print("Bloom-Enhanced DNABERT Variant Classifier")
    print("="*60 + "\n")
    
    # Create and launch dashboard
    dashboard = VariantAnalysisDashboard()
    dashboard.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )


if __name__ == "__main__":
    main()
