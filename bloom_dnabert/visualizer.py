"""
Attention Heatmap Visualizer with Bloom Filter Hit Overlay

This module creates interactive visualizations showing:
1. DNABERT attention weights across the sequence
2. Bloom filter hit positions overlaid
3. Sequence annotation with nucleotide importance
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple, Dict, Optional
import pandas as pd


class AttentionVisualizer:
    """
    Creates interactive heatmap visualizations of DNABERT attention
    with Bloom filter hit overlays.
    """
    
    def __init__(self, dnabert_wrapper, bloom_filter):
        """
        Initialize visualizer.
        
        Args:
            dnabert_wrapper: DNABERTWrapper instance
            bloom_filter: MultiScaleBloomFilter instance
        """
        self.dnabert_wrapper = dnabert_wrapper
        self.bloom_filter = bloom_filter
    
    def create_attention_heatmap(
        self,
        sequence: str,
        layer: int = -1,
        show_bloom_hits: bool = True,
        k_size: int = 8,
        title: str = "DNABERT Attention with Bloom Filter Hits"
    ) -> go.Figure:
        """
        Create an attention heatmap with Bloom filter hits overlaid.
        
        Args:
            sequence: DNA sequence
            layer: Which attention layer to visualize
            show_bloom_hits: Whether to show Bloom filter hits
            k_size: K-mer size for Bloom filter hits
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        # Get attention weights
        attention_matrix, tokens = self.dnabert_wrapper.get_attention_weights(
            sequence, layer=layer
        )
        
        # Get Bloom filter hits
        bloom_hits = None
        if show_bloom_hits:
            bloom_hits = self.bloom_filter.get_hit_positions(sequence, k=k_size)
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.85, 0.15],
            vertical_spacing=0.05,
            subplot_titles=(title, "Bloom Filter Hits")
        )
        
        # Main attention heatmap
        fig.add_trace(
            go.Heatmap(
                z=attention_matrix,
                x=tokens,
                y=tokens,
                colorscale='Viridis',
                colorbar=dict(title="Attention", x=1.02),
                hovertemplate='From: %{y}<br>To: %{x}<br>Attention: %{z:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Bloom filter hits as a bar plot
        if show_bloom_hits and bloom_hits:
            hit_positions = [pos for pos, _ in bloom_hits]
            hit_counts = np.zeros(len(sequence))
            for pos in hit_positions:
                hit_counts[pos:pos+k_size] += 1
            
            # Normalize to 0-1
            if hit_counts.max() > 0:
                hit_counts = hit_counts / hit_counts.max()
            
            fig.add_trace(
                go.Bar(
                    x=list(range(len(sequence))),
                    y=hit_counts,
                    marker_color='red',
                    name='Pathogenic K-mer Hits',
                    hovertemplate='Position: %{x}<br>Hit Density: %{y:.2f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_xaxes(title_text="Target Token", row=1, col=1)
        fig.update_yaxes(title_text="Source Token", row=1, col=1)
        fig.update_xaxes(title_text="Sequence Position", row=2, col=1)
        fig.update_yaxes(title_text="Hit Density", row=2, col=1)
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_x=0.5
        )
        
        return fig
    
    def create_nucleotide_importance_plot(
        self,
        sequence: str,
        layer: int = -1,
        show_bloom_hits: bool = True,
        k_size: int = 8
    ) -> go.Figure:
        """
        Create a plot showing importance of each nucleotide.
        
        Args:
            sequence: DNA sequence
            layer: Which attention layer to use
            show_bloom_hits: Whether to show Bloom hits
            k_size: K-mer size for Bloom filter
            
        Returns:
            Plotly figure object
        """
        # Get nucleotide importance from attention
        importance, _ = self.dnabert_wrapper.map_attention_to_sequence(sequence, layer=layer)
        
        # Get Bloom hits
        bloom_hits = self.bloom_filter.check_sequence(sequence)[k_size]
        
        # Create colors for nucleotides
        colors = []
        for base in sequence.upper():
            if base == 'A':
                colors.append('#1f77b4')  # Blue
            elif base == 'T':
                colors.append('#ff7f0e')  # Orange
            elif base == 'C':
                colors.append('#2ca02c')  # Green
            elif base == 'G':
                colors.append('#d62728')  # Red
            else:
                colors.append('#7f7f7f')  # Gray
        
        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.5, 0.25, 0.25],
            vertical_spacing=0.08,
            subplot_titles=(
                "Nucleotide Importance (from Attention)",
                "Bloom Filter Pathogenic K-mer Hits",
                "DNA Sequence"
            )
        )
        
        # Importance plot
        fig.add_trace(
            go.Bar(
                x=list(range(len(sequence))),
                y=importance,
                marker_color=colors,
                name='Attention Importance',
                hovertemplate='Position: %{x}<br>Base: ' + 
                             '<br>'.join([f'{sequence[i]}' for i in range(len(sequence))]) +
                             '<br>Importance: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Bloom hits
        if show_bloom_hits:
            hit_indicator = [1 if h else 0 for h in bloom_hits]
            fig.add_trace(
                go.Bar(
                    x=list(range(len(bloom_hits))),
                    y=hit_indicator,
                    marker_color='red',
                    name='Pathogenic K-mer',
                    hovertemplate='Position: %{x}<br>Pathogenic: %{y}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Sequence display
        sequence_display = list(sequence.upper())
        fig.add_trace(
            go.Scatter(
                x=list(range(len(sequence))),
                y=[0.5] * len(sequence),
                mode='text',
                text=sequence_display,
                textfont=dict(size=10, family='monospace', color=colors),
                hovertemplate='Position: %{x}<br>Base: %{text}<extra></extra>',
                showlegend=False
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_xaxes(title_text="Sequence Position", row=1, col=1)
        fig.update_yaxes(title_text="Importance", row=1, col=1)
        
        fig.update_xaxes(title_text="Sequence Position", row=2, col=1)
        fig.update_yaxes(title_text="Hit", row=2, col=1, range=[-0.1, 1.1])
        
        fig.update_xaxes(title_text="Sequence Position", row=3, col=1)
        fig.update_yaxes(showticklabels=False, row=3, col=1)
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Nucleotide-Level Analysis with Pathogenic K-mer Detection",
            title_x=0.5
        )
        
        return fig
    
    def create_comparison_plot(
        self,
        sequences: List[str],
        labels: List[str],
        layer: int = -1
    ) -> go.Figure:
        """
        Create a comparison plot showing importance across multiple sequences.
        
        Args:
            sequences: List of DNA sequences
            labels: List of labels for each sequence
            layer: Attention layer to use
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        for seq, label in zip(sequences, labels):
            importance, _ = self.dnabert_wrapper.map_attention_to_sequence(seq, layer=layer)
            
            # Pad/truncate to consistent length
            max_len = max(len(s) for s in sequences)
            if len(importance) < max_len:
                importance = np.pad(importance, (0, max_len - len(importance)))
            else:
                importance = importance[:max_len]
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(importance))),
                    y=importance,
                    mode='lines',
                    name=label,
                    hovertemplate=f'{label}<br>Position: %{{x}}<br>Importance: %{{y:.3f}}<extra></extra>'
                )
            )
        
        fig.update_layout(
            title="Attention Importance Comparison Across Sequences",
            xaxis_title="Sequence Position",
            yaxis_title="Attention Importance",
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def create_layer_comparison(
        self,
        sequence: str,
        layers: List[int] = None
    ) -> go.Figure:
        """
        Compare attention patterns across different layers.
        
        Args:
            sequence: DNA sequence
            layers: List of layers to compare (if None, use [0, 5, 11])
            
        Returns:
            Plotly figure object
        """
        if layers is None:
            layers = [0, 5, 11]  # First, middle, last
        
        # Get attention from all layers
        all_attentions = self.dnabert_wrapper.get_all_layer_attentions(sequence)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=len(layers),
            subplot_titles=[f"Layer {l}" for l in layers],
            horizontal_spacing=0.05
        )
        
        tokens = self.dnabert_wrapper.tokenizer.convert_ids_to_tokens(
            self.dnabert_wrapper.tokenizer(sequence, return_tensors="pt")["input_ids"][0]
        )
        
        for idx, layer in enumerate(layers):
            attention = all_attentions[layer]
            
            fig.add_trace(
                go.Heatmap(
                    z=attention,
                    x=tokens,
                    y=tokens,
                    colorscale='Viridis',
                    showscale=(idx == len(layers) - 1),
                    hovertemplate=f'Layer {layer}<br>From: %{{y}}<br>To: %{{x}}<br>Attention: %{{z:.3f}}<extra></extra>'
                ),
                row=1, col=idx+1
            )
        
        fig.update_layout(
            title_text="Attention Patterns Across Layers",
            height=500
        )
        
        return fig
    
    def create_dashboard(
        self,
        sequence: str,
        prediction_result: Dict = None,
        bloom_k: int = 8
    ) -> go.Figure:
        """
        Create a comprehensive dashboard with all visualizations.
        
        Args:
            sequence: DNA sequence
            prediction_result: Prediction results from classifier
            bloom_k: K-mer size for Bloom filter
            
        Returns:
            Plotly figure object
        """
        # Get data
        importance, _ = self.dnabert_wrapper.map_attention_to_sequence(sequence)
        bloom_hits = self.bloom_filter.check_sequence(sequence)[bloom_k]
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            row_heights=[0.4, 0.3, 0.3],
            column_widths=[0.7, 0.3],
            specs=[
                [{"type": "bar"}, {"type": "indicator"}],
                [{"type": "bar"}, {"type": "table"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ],
            subplot_titles=(
                "Nucleotide Importance",
                "Prediction",
                "Bloom Filter Hits",
                "Statistics",
                "Sequence",
                "Feature Distribution"
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Nucleotide importance
        colors = ['#1f77b4' if b=='A' else '#ff7f0e' if b=='T' else '#2ca02c' if b=='C' else '#d62728' 
                  for b in sequence.upper()]
        
        fig.add_trace(
            go.Bar(x=list(range(len(sequence))), y=importance, marker_color=colors),
            row=1, col=1
        )
        
        # 2. Prediction indicator
        if prediction_result:
            prob = prediction_result['probability']
            pred_label = prediction_result['prediction']
            color = 'red' if pred_label == 'Pathogenic' else 'green'
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=prob * 100,
                    title={'text': pred_label},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': color},
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ),
                row=1, col=2
            )
        
        # 3. Bloom filter hits
        hit_indicator = [1 if h else 0 for h in bloom_hits]
        fig.add_trace(
            go.Bar(x=list(range(len(bloom_hits))), y=hit_indicator, marker_color='red'),
            row=2, col=1
        )
        
        # 4. Statistics table
        bloom_features = self.bloom_filter.get_hit_features(sequence)
        stats_data = [
            ["Sequence Length", str(len(sequence))],
            ["Bloom Hits (k=6)", str(int(bloom_features['hit_count_k6']))],
            ["Bloom Hits (k=8)", str(int(bloom_features['hit_count_k8']))],
            ["Bloom Hits (k=10)", str(int(bloom_features['hit_count_k10']))],
            ["Hit Ratio", f"{bloom_features['mean_hit_ratio']:.3f}"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=["Metric", "Value"]),
                cells=dict(values=[[row[0] for row in stats_data], 
                                   [row[1] for row in stats_data]])
            ),
            row=2, col=2
        )
        
        # 5. Sequence display
        fig.add_trace(
            go.Scatter(
                x=list(range(len(sequence))),
                y=[0.5] * len(sequence),
                mode='text',
                text=list(sequence.upper()),
                textfont=dict(size=8, family='monospace'),
                showlegend=False
            ),
            row=3, col=1
        )
        
        # 6. Feature distribution
        feature_names = ['k6', 'k8', 'k10']
        feature_values = [
            bloom_features['hit_ratio_k6'],
            bloom_features['hit_ratio_k8'],
            bloom_features['hit_ratio_k10']
        ]
        
        fig.add_trace(
            go.Bar(x=feature_names, y=feature_values, marker_color='purple'),
            row=3, col=2
        )
        
        # Update layout
        fig.update_xaxes(title_text="Position", row=1, col=1)
        fig.update_yaxes(title_text="Importance", row=1, col=1)
        
        fig.update_xaxes(title_text="Position", row=2, col=1)
        fig.update_yaxes(title_text="Hit", row=2, col=1)
        
        fig.update_xaxes(title_text="Position", row=3, col=1)
        fig.update_yaxes(showticklabels=False, row=3, col=1)
        
        fig.update_xaxes(title_text="K-mer Size", row=3, col=2)
        fig.update_yaxes(title_text="Hit Ratio", row=3, col=2)
        
        fig.update_layout(
            height=1000,
            showlegend=False,
            title_text=f"Variant Analysis Dashboard - {prediction_result.get('prediction', 'Unknown') if prediction_result else 'Analysis'}",
            title_x=0.5
        )
        
        return fig
