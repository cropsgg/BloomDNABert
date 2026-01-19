"""
DNABERT Wrapper with Attention Weight Extraction

This module wraps DNABERT-2-117M to extract both embeddings and attention weights
for interpretable variant classification.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
import warnings


class DNABERTWrapper:
    """
    Wrapper for DNABERT-2 with attention weight extraction capabilities.
    
    This class handles:
    - Loading DNABERT-2-117M model
    - Extracting sequence embeddings
    - Extracting and aggregating attention weights
    - Providing interpretable attention maps
    """
    
    def __init__(self, model_name: str = "zhihan1996/DNABERT-2-117M", device: str = None):
        """
        Initialize DNABERT wrapper.
        
        Args:
            model_name: HuggingFace model name/path
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        
        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading DNABERT-2 model on {self.device}...")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            output_attentions=True  # Enable attention output
        ).to(self.device)
        
        self.model.eval()
        
        # Model configuration
        self.hidden_size = 768  # DNABERT-2-117M hidden size
        self.num_layers = 12    # Number of transformer layers
        self.num_heads = 12     # Number of attention heads per layer
        
        print(f"[OK] DNABERT-2 model loaded successfully")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Layers: {self.num_layers}")
        print(f"  Attention heads: {self.num_heads}")
    
    def get_embedding(self, sequence: str, pool_method: str = 'mean') -> np.ndarray:
        """
        Get sequence embedding from DNABERT-2.
        
        Args:
            sequence: DNA sequence (A, T, C, G)
            pool_method: Pooling method ('mean', 'max', 'cls')
            
        Returns:
            Numpy array of shape (hidden_size,)
        """
        sequence = sequence.upper()
        
        # Tokenize
        inputs = self.tokenizer(sequence, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        # Get model output
        with torch.no_grad():
            outputs = self.model(input_ids)
            # Handle different output formats
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            else:
                hidden_states = outputs[0]  # First element is hidden states
        
        # Pool embeddings
        if pool_method == 'mean':
            embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()
        elif pool_method == 'max':
            embedding = hidden_states.max(dim=1)[0].squeeze().cpu().numpy()
        elif pool_method == 'cls':
            embedding = hidden_states[:, 0, :].squeeze().cpu().numpy()
        else:
            raise ValueError(f"Unknown pool_method: {pool_method}")
        
        return embedding
    
    def get_attention_weights(
        self,
        sequence: str,
        layer: int = -1,
        aggregate_heads: str = 'mean'
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract attention weights from DNABERT-2.
        
        Args:
            sequence: DNA sequence
            layer: Which layer to extract from (-1 for last layer)
            aggregate_heads: How to aggregate attention heads ('mean', 'max', 'min')
            
        Returns:
            Tuple of (attention_matrix, tokens)
            - attention_matrix: [seq_len, seq_len] attention weights
            - tokens: List of token strings
        """
        sequence = sequence.upper()
        
        # Tokenize
        inputs = self.tokenizer(sequence, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        # Get model output with attentions
        with torch.no_grad():
            outputs = self.model(input_ids)
            # Handle different output formats
            if hasattr(outputs, 'attentions'):
                attentions = outputs.attentions
            else:
                # Outputs is a tuple: (hidden_states, attentions)
                attentions = outputs[1] if len(outputs) > 1 else None
                
        if attentions is None:
            raise ValueError("Model did not return attention weights. Make sure output_attentions=True")
        
        # Get tokens for visualization
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Select layer
        if layer == -1:
            layer = len(attentions) - 1
        
        attention = attentions[layer].squeeze(0)  # [num_heads, seq_len, seq_len]
        
        # Aggregate across heads
        if aggregate_heads == 'mean':
            attention_matrix = attention.mean(dim=0).cpu().numpy()
        elif aggregate_heads == 'max':
            attention_matrix = attention.max(dim=0)[0].cpu().numpy()
        elif aggregate_heads == 'min':
            attention_matrix = attention.min(dim=0)[0].cpu().numpy()
        else:
            raise ValueError(f"Unknown aggregate_heads: {aggregate_heads}")
        
        return attention_matrix, tokens
    
    def get_all_layer_attentions(
        self,
        sequence: str,
        aggregate_heads: str = 'mean'
    ) -> List[np.ndarray]:
        """
        Get attention matrices from all layers.
        
        Args:
            sequence: DNA sequence
            aggregate_heads: How to aggregate attention heads
            
        Returns:
            List of attention matrices, one per layer
        """
        sequence = sequence.upper()
        
        # Tokenize
        inputs = self.tokenizer(sequence, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        # Get model output with attentions
        with torch.no_grad():
            outputs = self.model(input_ids)
            # Handle different output formats
            if hasattr(outputs, 'attentions'):
                attentions = outputs.attentions
            else:
                attentions = outputs[1] if len(outputs) > 1 else None
        
        attention_matrices = []
        
        for attention in attentions:
            attention = attention.squeeze(0)  # [num_heads, seq_len, seq_len]
            
            # Aggregate across heads
            if aggregate_heads == 'mean':
                attention_matrix = attention.mean(dim=0).cpu().numpy()
            elif aggregate_heads == 'max':
                attention_matrix = attention.max(dim=0)[0].cpu().numpy()
            else:
                attention_matrix = attention.mean(dim=0).cpu().numpy()
            
            attention_matrices.append(attention_matrix)
        
        return attention_matrices
    
    def get_token_importance(
        self,
        sequence: str,
        layer: int = -1,
        method: str = 'attention_sum'
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Get importance score for each token in the sequence.
        
        Args:
            sequence: DNA sequence
            layer: Which layer to use (-1 for last)
            method: 'attention_sum' (sum of attention to token) or
                   'attention_entropy' (entropy of attention from token)
            
        Returns:
            Tuple of (importance_scores, tokens)
            - importance_scores: [seq_len] array of importance values
            - tokens: List of token strings
        """
        attention_matrix, tokens = self.get_attention_weights(sequence, layer=layer)
        
        if method == 'attention_sum':
            # Sum of attention weights received by each token
            importance = attention_matrix.sum(axis=0)
        elif method == 'attention_entropy':
            # Entropy of attention distribution from each token
            importance = np.zeros(attention_matrix.shape[0])
            for i in range(attention_matrix.shape[0]):
                attn = attention_matrix[i, :]
                attn = attn + 1e-10  # Avoid log(0)
                entropy = -np.sum(attn * np.log(attn))
                importance[i] = entropy
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Normalize to [0, 1]
        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-10)
        
        return importance, tokens
    
    def map_attention_to_sequence(
        self,
        sequence: str,
        layer: int = -1
    ) -> Tuple[np.ndarray, str]:
        """
        Map attention weights back to original sequence positions.
        
        Since DNABERT uses BPE tokenization, multiple characters may be
        in one token. This method approximates attention per nucleotide.
        
        Args:
            sequence: DNA sequence
            layer: Which layer to use
            
        Returns:
            Tuple of (nucleotide_importance, sequence)
            - nucleotide_importance: [len(sequence)] array
            - sequence: Original sequence
        """
        importance, tokens = self.get_token_importance(sequence, layer=layer)
        
        # Map tokens back to sequence positions
        # This is approximate since BPE tokenization is complex
        sequence = sequence.upper()
        nucleotide_importance = np.zeros(len(sequence))
        
        # Simple heuristic: distribute token importance across its characters
        current_pos = 0
        for token, imp in zip(tokens, importance):
            # Skip special tokens
            if token.startswith('[') or token.startswith('<'):
                continue
            
            # Get token text (remove special markers)
            token_text = token.replace('▁', '')  # BPE marker
            token_len = len(token_text)
            
            # Distribute importance
            if current_pos + token_len <= len(sequence):
                nucleotide_importance[current_pos:current_pos + token_len] = imp
                current_pos += token_len
        
        return nucleotide_importance, sequence
    
    def get_embedding_and_attention(
        self,
        sequence: str,
        pool_method: str = 'mean',
        attention_layer: int = -1
    ) -> Dict[str, np.ndarray]:
        """
        Get both embedding and attention in a single forward pass.
        
        Args:
            sequence: DNA sequence
            pool_method: Pooling method for embedding
            attention_layer: Layer to extract attention from
            
        Returns:
            Dictionary with 'embedding' and 'attention' keys
        """
        sequence = sequence.upper()
        
        # Tokenize
        inputs = self.tokenizer(sequence, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        # Get model output
        with torch.no_grad():
            outputs = self.model(input_ids)
            # Handle different output formats
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
                attentions = outputs.attentions if hasattr(outputs, 'attentions') else outputs[1]
            else:
                hidden_states = outputs[0]
                attentions = outputs[1] if len(outputs) > 1 else None
        
        # Get embedding
        if pool_method == 'mean':
            embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()
        elif pool_method == 'max':
            embedding = hidden_states.max(dim=1)[0].squeeze().cpu().numpy()
        elif pool_method == 'cls':
            embedding = hidden_states[:, 0, :].squeeze().cpu().numpy()
        else:
            embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()
        
        # Get attention
        if attention_layer == -1:
            attention_layer = len(attentions) - 1
        
        attention = attentions[attention_layer].squeeze(0)
        attention_matrix = attention.mean(dim=0).cpu().numpy()
        
        return {
            'embedding': embedding,
            'attention': attention_matrix
        }
