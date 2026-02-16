"""
Hybrid Classifier combining Bloom Filter features and DNABERT embeddings

This module implements two classification architectures:

1. HybridClassifier (baseline): Simple concatenation + MLP
2. BloomGuidedPipeline (novel BGPCA): Position-aware cross-modal attention

The novel BGPCA architecture preserves positional correspondence between
Bloom filter hits and DNABERT token representations, using cross-attention
with Bloom-derived attention biases for knowledge-guided classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import pickle
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from .bloom_attention_bridge import BloomGuidedClassifier


class HybridClassifier(nn.Module):
    """
    Hybrid neural network classifier combining Bloom filter features
    and DNABERT embeddings for pathogenic variant classification.
    
    Architecture:
    - Bloom features (18-dim) + DNABERT embedding (768-dim) = 786-dim input
    - 2-layer MLP with dropout
    - Binary classification output
    """
    
    def __init__(
        self,
        dnabert_dim: int = 768,
        bloom_dim: int = 18,
        hidden_dim: int = 256,
        dropout: float = 0.3
    ):
        """
        Initialize hybrid classifier.
        
        Args:
            dnabert_dim: DNABERT embedding dimension
            bloom_dim: Bloom filter feature dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super(HybridClassifier, self).__init__()
        
        self.dnabert_dim = dnabert_dim
        self.bloom_dim = bloom_dim
        input_dim = dnabert_dim + bloom_dim
        
        # MLP layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)
        
    def forward(self, dnabert_embedding: torch.Tensor, bloom_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            dnabert_embedding: [batch_size, dnabert_dim]
            bloom_features: [batch_size, bloom_dim]
            
        Returns:
            Logits [batch_size, 1]
        """
        # Concatenate features
        x = torch.cat([dnabert_embedding, bloom_features], dim=1)
        
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Output layer
        x = self.fc3(x)
        
        return x
    
    def predict_proba(self, dnabert_embedding: torch.Tensor, bloom_features: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions.
        
        Args:
            dnabert_embedding: [batch_size, dnabert_dim]
            bloom_features: [batch_size, bloom_dim]
            
        Returns:
            Probabilities [batch_size]
        """
        logits = self.forward(dnabert_embedding, bloom_features)
        probs = torch.sigmoid(logits).squeeze()
        return probs


class HybridClassifierPipeline:
    """
    Complete pipeline for training and inference with the hybrid classifier.
    
    This class orchestrates:
    - Bloom filter initialization
    - DNABERT model loading
    - Feature extraction
    - Model training
    - Inference
    """
    
    def __init__(
        self,
        bloom_filter=None,
        dnabert_wrapper=None,
        device: str = None
    ):
        """
        Initialize pipeline.
        
        Args:
            bloom_filter: MultiScaleBloomFilter instance
            dnabert_wrapper: DNABERTWrapper instance
            device: Device for PyTorch
        """
        self.bloom_filter = bloom_filter
        self.dnabert_wrapper = dnabert_wrapper
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.trained = False
        
    def extract_features(self, sequence: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract both Bloom and DNABERT features from a sequence.
        
        Args:
            sequence: DNA sequence
            
        Returns:
            Tuple of (bloom_features, dnabert_embedding)
        """
        # Extract Bloom filter features
        bloom_features = self.bloom_filter.get_feature_vector(sequence)
        
        # Extract DNABERT embedding
        dnabert_embedding = self.dnabert_wrapper.get_embedding(sequence)
        
        return bloom_features, dnabert_embedding
    
    def prepare_dataset(self, sequences: list, labels: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare dataset for training/evaluation.
        
        Args:
            sequences: List of DNA sequences
            labels: List of labels (0=benign, 1=pathogenic)
            
        Returns:
            Tuple of (bloom_features, dnabert_embeddings, labels) as tensors
        """
        bloom_features_list = []
        dnabert_embeddings_list = []
        
        print(f"Extracting features from {len(sequences)} sequences...")
        for i, seq in enumerate(sequences):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(sequences)}")
            
            bloom_feat, dnabert_emb = self.extract_features(seq)
            bloom_features_list.append(bloom_feat)
            dnabert_embeddings_list.append(dnabert_emb)
        
        # Convert to tensors
        bloom_features = torch.tensor(np.array(bloom_features_list), dtype=torch.float32)
        dnabert_embeddings = torch.tensor(np.array(dnabert_embeddings_list), dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        
        return bloom_features, dnabert_embeddings, labels_tensor
    
    def train(
        self,
        train_sequences: list,
        train_labels: list,
        val_sequences: list = None,
        val_labels: list = None,
        epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        patience: int = 10,
        max_grad_norm: float = 1.0
    ) -> Dict[str, list]:
        """
        Train the hybrid classifier with early stopping and gradient clipping.
        
        Args:
            train_sequences: Training sequences
            train_labels: Training labels
            val_sequences: Validation sequences (optional)
            val_labels: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: L2 regularization
            patience: Early stopping patience (epochs without val loss improvement)
            max_grad_norm: Max gradient norm for clipping
            
        Returns:
            Dictionary with training history
        """
        print("\n" + "=" * 60)
        print("Training Hybrid Classifier")
        print("=" * 60)
        
        train_bloom, train_dnabert, train_labels_t = self.prepare_dataset(train_sequences, train_labels)
        
        if val_sequences is not None:
            val_bloom, val_dnabert, val_labels_t = self.prepare_dataset(val_sequences, val_labels)
        
        self.model = HybridClassifier(
            dnabert_dim=train_dnabert.shape[1],
            bloom_dim=train_bloom.shape[1]
        ).to(self.device)
        
        # Compute class weights for imbalanced labels
        n_pos = sum(train_labels)
        n_neg = len(train_labels) - n_pos
        if n_pos > 0 and n_neg > 0:
            pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(self.device)
        else:
            pos_weight = torch.tensor([1.0], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=learning_rate * 0.01
        )
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        n_samples = len(train_sequences)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        # Early stopping state
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        print(f"\nTraining settings:")
        print(f"  Samples: {n_samples}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Class weight (pos): {pos_weight.item():.3f}")
        print(f"  Gradient clipping: {max_grad_norm}")
        print(f"  Early stopping patience: {patience}")
        print(f"  Device: {self.device}")
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            
            indices = torch.randperm(n_samples)
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_bloom = train_bloom[batch_indices].to(self.device)
                batch_dnabert = train_dnabert[batch_indices].to(self.device)
                batch_labels = train_labels_t[batch_indices].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_dnabert, batch_bloom).squeeze()
                loss = criterion(outputs, batch_labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                
                epoch_loss += loss.item() * len(batch_indices)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                epoch_correct += (preds == batch_labels).sum().item()
            
            scheduler.step()
            
            avg_loss = epoch_loss / n_samples
            avg_acc = epoch_correct / n_samples
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(avg_acc)
            
            if val_sequences is not None:
                val_loss, val_acc = self._evaluate(
                    val_bloom, val_dnabert, val_labels_t, criterion
                )
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 5 == 0:
                    lr = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                          f"LR: {lr:.6f}")
                
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1} "
                          f"(no val loss improvement for {patience} epochs)")
                    break
            else:
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
        
        # Restore best model if we have validation
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Restored best model (val loss: {best_val_loss:.4f})")
        
        self.trained = True
        print(f"\nTraining completed! ({epoch+1} epochs)")
        
        return history
    
    def _evaluate(
        self,
        bloom_features: torch.Tensor,
        dnabert_embeddings: torch.Tensor,
        labels: torch.Tensor,
        criterion
    ) -> Tuple[float, float]:
        """Evaluate on a dataset."""
        self.model.eval()
        
        with torch.no_grad():
            bloom_features = bloom_features.to(self.device)
            dnabert_embeddings = dnabert_embeddings.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(dnabert_embeddings, bloom_features).squeeze()
            loss = criterion(outputs, labels).item()
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            acc = (preds == labels).float().mean().item()
        
        return loss, acc
    
    MAX_SEQUENCE_LENGTH = 5000

    def _validate_sequence(self, sequence: str) -> str:
        """Validate and sanitize a DNA sequence."""
        if not sequence or not isinstance(sequence, str):
            raise ValueError("Sequence must be a non-empty string")
        sequence = sequence.upper().strip()
        if len(sequence) < 10:
            raise ValueError(f"Sequence too short ({len(sequence)} bp). Minimum is 10.")
        if len(sequence) > self.MAX_SEQUENCE_LENGTH:
            raise ValueError(
                f"Sequence too long ({len(sequence)} bp). "
                f"Maximum is {self.MAX_SEQUENCE_LENGTH}."
            )
        invalid = set(sequence) - set('ATCGN')
        if invalid:
            raise ValueError(f"Invalid characters in sequence: {invalid}. Only A,T,C,G,N allowed.")
        return sequence

    def predict(self, sequence: str) -> Dict[str, float]:
        """
        Predict whether a sequence is pathogenic.
        
        Args:
            sequence: DNA sequence (10-5000 bp, ATCGN only)
            
        Returns:
            Dictionary with prediction results
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        
        sequence = self._validate_sequence(sequence)
        self.model.eval()
        
        bloom_feat, dnabert_emb = self.extract_features(sequence)
        
        bloom_feat = torch.tensor(bloom_feat, dtype=torch.float32).unsqueeze(0).to(self.device)
        dnabert_emb = torch.tensor(dnabert_emb, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(dnabert_emb, bloom_feat)
            prob = torch.sigmoid(output).item()
        
        return {
            'probability': prob,
            'prediction': 'Pathogenic' if prob > 0.5 else 'Benign',
            'confidence': prob if prob > 0.5 else (1 - prob)
        }
    
    def evaluate(self, sequences: list, labels: list) -> Dict[str, float]:
        """
        Evaluate model on a test set.
        
        Args:
            sequences: Test sequences
            labels: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        bloom_features, dnabert_embeddings, labels_tensor = self.prepare_dataset(sequences, labels)
        
        self.model.eval()
        
        with torch.no_grad():
            bloom_features = bloom_features.to(self.device)
            dnabert_embeddings = dnabert_embeddings.to(self.device)
            
            outputs = self.model(dnabert_embeddings, bloom_features).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
        
        # Calculate metrics
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        auc = roc_auc_score(labels, probs)
        
        metrics = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc
        }
        
        return metrics
    
    def save(self, path: str):
        """Save model to disk."""
        if not self.trained:
            raise ValueError("Cannot save untrained model")
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'dnabert_dim': self.model.dnabert_dim,
                'bloom_dim': self.model.bloom_dim,
                'hidden_dim': self.model.fc1.out_features,
                'dropout': self.model.dropout1.p,
            },
            'architecture': 'Baseline'
        }
        
        torch.save(save_dict, path)
        print(f"Baseline model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        save_dict = torch.load(path, map_location=self.device, weights_only=False)

        if 'model_config' not in save_dict or 'model_state_dict' not in save_dict:
            raise ValueError(f"Invalid model file format: {path}")

        self.model = HybridClassifier(**save_dict['model_config']).to(self.device)
        self.model.load_state_dict(save_dict['model_state_dict'])
        self.trained = True
        
        print(f"Model loaded from {path}")


class BloomGuidedPipeline:
    """
    Training and inference pipeline for the novel Bloom-Guided Positional
    Cross-Attention (BGPCA) architecture.

    Unlike HybridClassifierPipeline which extracts flat features and
    concatenates them, this pipeline:
    1. Extracts per-TOKEN hidden states from DNABERT (preserving position)
    2. Computes per-TOKEN Bloom activation (aligned to token positions)
    3. Also extracts flat Bloom summary features (for gated fusion)
    4. Pads/truncates to uniform length for batched training
    5. Feeds all three to the BloomGuidedClassifier

    This preserves spatial correspondence between what the Bloom filter
    detects and what DNABERT represents at each position.
    """

    def __init__(
        self,
        bloom_filter=None,
        dnabert_wrapper=None,
        device: str = None,
        max_tokens: int = 128,
        d_bloom: int = 64,
        n_cross_attn_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.2
    ):
        self.bloom_filter = bloom_filter
        self.dnabert_wrapper = dnabert_wrapper
        self.max_tokens = max_tokens
        self.d_bloom = d_bloom
        self.n_cross_attn_layers = n_cross_attn_layers
        self.n_heads = n_heads
        self.dropout = dropout

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = None
        self.trained = False

    def extract_positional_features(
        self,
        sequence: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract position-aware features from a single sequence.

        Returns:
            hidden_states:  [num_tokens, 768] DNABERT per-token
            bloom_signal:   [num_tokens, 3]   token-aligned Bloom activation
            bloom_summary:  [18]              flat Bloom features
        """
        hidden_np, token_spans, _ = self.dnabert_wrapper.get_token_level_outputs(sequence)
        bloom_signal = self.bloom_filter.get_token_aligned_signal(sequence, token_spans)
        bloom_summary = self.bloom_filter.get_feature_vector(sequence)

        return hidden_np, bloom_signal, bloom_summary

    def _pad_or_truncate(
        self,
        tensor: np.ndarray,
        max_len: int,
        pad_value: float = 0.0
    ) -> np.ndarray:
        """Pad or truncate first dimension to max_len."""
        if len(tensor) >= max_len:
            return tensor[:max_len]
        else:
            pad_shape = list(tensor.shape)
            pad_shape[0] = max_len - len(tensor)
            padding = np.full(pad_shape, pad_value, dtype=tensor.dtype)
            return np.concatenate([tensor, padding], axis=0)

    def prepare_dataset(
        self,
        sequences: list,
        labels: list
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare dataset with position-aware features.

        Returns:
            hidden_states:  [N, max_tokens, 768]
            bloom_signals:  [N, max_tokens, 3]
            bloom_summaries: [N, 18]
            labels:         [N]
        """
        all_hidden = []
        all_bloom_signal = []
        all_bloom_summary = []

        print(f"Extracting positional features from {len(sequences)} sequences...")
        for i, seq in enumerate(sequences):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(sequences)}")

            hidden, bloom_sig, bloom_sum = self.extract_positional_features(seq)

            hidden = self._pad_or_truncate(hidden, self.max_tokens)
            bloom_sig = self._pad_or_truncate(bloom_sig, self.max_tokens)

            all_hidden.append(hidden)
            all_bloom_signal.append(bloom_sig)
            all_bloom_summary.append(bloom_sum)

        hidden_t = torch.tensor(np.array(all_hidden), dtype=torch.float32)
        bloom_sig_t = torch.tensor(np.array(all_bloom_signal), dtype=torch.float32)
        bloom_sum_t = torch.tensor(np.array(all_bloom_summary), dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.float32)

        return hidden_t, bloom_sig_t, bloom_sum_t, labels_t

    def train(
        self,
        train_sequences: list,
        train_labels: list,
        val_sequences: list = None,
        val_labels: list = None,
        epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        patience: int = 10,
        max_grad_norm: float = 1.0
    ) -> Dict[str, list]:
        """
        Train the BGPCA classifier with early stopping and class-weighted loss.
        """
        print("\n" + "=" * 60)
        print("Training Bloom-Guided Positional Cross-Attention (BGPCA)")
        print("=" * 60)

        train_hidden, train_bloom_sig, train_bloom_sum, train_labels_t = \
            self.prepare_dataset(train_sequences, train_labels)

        if val_sequences is not None:
            val_hidden, val_bloom_sig, val_bloom_sum, val_labels_t = \
                self.prepare_dataset(val_sequences, val_labels)

        d_model = train_hidden.shape[2]
        n_bloom_scales = train_bloom_sig.shape[2]
        bloom_summary_dim = train_bloom_sum.shape[1]

        self.model = BloomGuidedClassifier(
            d_model=d_model,
            d_bloom=self.d_bloom,
            n_bloom_scales=n_bloom_scales,
            bloom_summary_dim=bloom_summary_dim,
            n_heads=self.n_heads,
            n_cross_attn_layers=self.n_cross_attn_layers,
            dropout=self.dropout
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nBGPCA Model Architecture:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Cross-attention layers: {self.n_cross_attn_layers}")
        print(f"  Attention heads: {self.n_heads}")
        print(f"  Bloom encoding dim: {self.d_bloom}")
        print(f"  Max tokens: {self.max_tokens}")

        # Compute class weights for imbalanced labels
        n_pos = sum(train_labels)
        n_neg = len(train_labels) - n_pos
        if n_pos > 0 and n_neg > 0:
            pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(self.device)
        else:
            pos_weight = torch.tensor([1.0], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=learning_rate * 0.01
        )

        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

        n_samples = len(train_sequences)
        n_batches = (n_samples + batch_size - 1) // batch_size

        # Early stopping state
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        print(f"\nTraining settings:")
        print(f"  Samples: {n_samples}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Class weight (pos): {pos_weight.item():.3f}")
        print(f"  Gradient clipping: {max_grad_norm}")
        print(f"  Early stopping patience: {patience}")
        print(f"  Optimizer: AdamW + CosineAnnealing")
        print(f"  Device: {self.device}")

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_correct = 0

            indices = torch.randperm(n_samples)

            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                batch_idx = indices[start_idx:end_idx]

                b_hidden = train_hidden[batch_idx].to(self.device)
                b_bloom_sig = train_bloom_sig[batch_idx].to(self.device)
                b_bloom_sum = train_bloom_sum[batch_idx].to(self.device)
                b_labels = train_labels_t[batch_idx].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(b_hidden, b_bloom_sig, b_bloom_sum).squeeze()
                loss = criterion(outputs, b_labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

                epoch_loss += loss.item() * len(batch_idx)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                epoch_correct += (preds == b_labels).sum().item()

            scheduler.step()

            avg_loss = epoch_loss / n_samples
            avg_acc = epoch_correct / n_samples
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(avg_acc)

            if val_sequences is not None:
                val_loss, val_acc = self._evaluate(
                    val_hidden, val_bloom_sig, val_bloom_sum, val_labels_t, criterion
                )
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {
                        k: v.clone() for k, v in self.model.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1

                if (epoch + 1) % 5 == 0:
                    lr = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                          f"LR: {lr:.6f}")

                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1} "
                          f"(no val loss improvement for {patience} epochs)")
                    break
            else:
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

        # Restore best model if we have validation
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Restored best model (val loss: {best_val_loss:.4f})")

        self.trained = True
        print(f"\nTraining completed! ({epoch+1} epochs)")

        return history

    def _evaluate(
        self,
        hidden: torch.Tensor,
        bloom_sig: torch.Tensor,
        bloom_sum: torch.Tensor,
        labels: torch.Tensor,
        criterion
    ) -> Tuple[float, float]:
        """Evaluate on a pre-extracted dataset."""
        self.model.eval()

        with torch.no_grad():
            hidden = hidden.to(self.device)
            bloom_sig = bloom_sig.to(self.device)
            bloom_sum = bloom_sum.to(self.device)
            labels_d = labels.to(self.device)

            outputs = self.model(hidden, bloom_sig, bloom_sum).squeeze()
            loss = criterion(outputs, labels_d).item()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            acc = (preds == labels_d).float().mean().item()

        return loss, acc

    MAX_SEQUENCE_LENGTH = 5000

    def _validate_sequence(self, sequence: str) -> str:
        """Validate and sanitize a DNA sequence."""
        if not sequence or not isinstance(sequence, str):
            raise ValueError("Sequence must be a non-empty string")
        sequence = sequence.upper().strip()
        if len(sequence) < 10:
            raise ValueError(f"Sequence too short ({len(sequence)} bp). Minimum is 10.")
        if len(sequence) > self.MAX_SEQUENCE_LENGTH:
            raise ValueError(
                f"Sequence too long ({len(sequence)} bp). "
                f"Maximum is {self.MAX_SEQUENCE_LENGTH}."
            )
        invalid = set(sequence) - set('ATCGN')
        if invalid:
            raise ValueError(f"Invalid characters in sequence: {invalid}. Only A,T,C,G,N allowed.")
        return sequence

    def _prepare_single_sequence(self, sequence: str):
        """Extract and tensorize features for a single validated sequence."""
        hidden, bloom_sig, bloom_sum = self.extract_positional_features(sequence)
        hidden = self._pad_or_truncate(hidden, self.max_tokens)
        bloom_sig = self._pad_or_truncate(bloom_sig, self.max_tokens)

        hidden_t = torch.tensor(hidden, dtype=torch.float32).unsqueeze(0).to(self.device)
        bloom_sig_t = torch.tensor(bloom_sig, dtype=torch.float32).unsqueeze(0).to(self.device)
        bloom_sum_t = torch.tensor(bloom_sum, dtype=torch.float32).unsqueeze(0).to(self.device)

        return hidden_t, bloom_sig_t, bloom_sum_t, bloom_sig

    def predict(self, sequence: str) -> Dict[str, float]:
        """
        Predict pathogenicity for a single sequence.

        Args:
            sequence: DNA sequence (10-5000 bp, ATCGN only)
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")

        sequence = self._validate_sequence(sequence)
        self.model.eval()

        hidden_t, bloom_sig_t, bloom_sum_t, _ = self._prepare_single_sequence(sequence)

        with torch.no_grad():
            output = self.model(hidden_t, bloom_sig_t, bloom_sum_t)
            prob = torch.sigmoid(output).item()

        return {
            'probability': prob,
            'prediction': 'Pathogenic' if prob > 0.5 else 'Benign',
            'confidence': prob if prob > 0.5 else (1 - prob)
        }

    def predict_with_uncertainty(
        self,
        sequence: str,
        n_samples: int = 20
    ) -> Dict[str, float]:
        """
        Predict with Monte Carlo dropout uncertainty estimation.

        Returns prediction, confidence, AND epistemic uncertainty.
        High uncertainty = model is unsure (important for clinical use).

        Args:
            sequence: DNA sequence (10-5000 bp, ATCGN only)
            n_samples: Number of MC dropout forward passes
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")

        sequence = self._validate_sequence(sequence)
        hidden_t, bloom_sig_t, bloom_sum_t, _ = self._prepare_single_sequence(sequence)

        mean_pred, uncertainty = self.model.predict_with_uncertainty(
            hidden_t, bloom_sig_t, bloom_sum_t, n_samples=n_samples
        )

        prob = mean_pred.item()
        unc = uncertainty.item()

        return {
            'probability': prob,
            'prediction': 'Pathogenic' if prob > 0.5 else 'Benign',
            'confidence': prob if prob > 0.5 else (1 - prob),
            'uncertainty': unc,
            'uncertainty_level': 'Low' if unc < 0.05 else ('Medium' if unc < 0.15 else 'High')
        }

    def predict_with_interpretability(
        self,
        sequence: str
    ) -> Dict:
        """
        Predict with full interpretability outputs.

        Returns prediction plus:
        - Position importance weights (which positions matter most)
        - Cross-attention weights (Bloom-guided attention patterns)
        - Gate values (Bloom vs DNABERT trust balance)

        Args:
            sequence: DNA sequence (10-5000 bp, ATCGN only)
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")

        sequence = self._validate_sequence(sequence)
        self.model.eval()

        hidden_t, bloom_sig_t, bloom_sum_t, bloom_sig_np = \
            self._prepare_single_sequence(sequence)

        with torch.no_grad():
            result = self.model.forward_with_interpretability(
                hidden_t, bloom_sig_t, bloom_sum_t
            )

        prob = torch.sigmoid(result['logits']).item()

        return {
            'probability': prob,
            'prediction': 'Pathogenic' if prob > 0.5 else 'Benign',
            'confidence': prob if prob > 0.5 else (1 - prob),
            'position_importance': result['position_importance'].squeeze(0).cpu().numpy(),
            'cross_attn_weights': [w.squeeze(0).cpu().numpy() for w in result['cross_attn_weights']],
            'gate_values': result['gate_values'].squeeze(0).cpu().numpy(),
            'bloom_positional_signal': bloom_sig_np
        }

    def evaluate(self, sequences: list, labels: list) -> Dict[str, float]:
        """Evaluate model on a test set with comprehensive metrics."""
        hidden, bloom_sig, bloom_sum, labels_t = self.prepare_dataset(sequences, labels)

        self.model.eval()

        with torch.no_grad():
            hidden = hidden.to(self.device)
            bloom_sig = bloom_sig.to(self.device)
            bloom_sum = bloom_sum.to(self.device)

            outputs = self.model(hidden, bloom_sig, bloom_sum).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)

        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary'
        )
        auc = roc_auc_score(labels, probs)

        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc
        }

    def calibration_analysis(
        self,
        sequences: list,
        labels: list,
        n_bins: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Compute calibration metrics to assess whether predicted probabilities
        match observed frequencies. Critical for clinical reliability.

        A well-calibrated model that predicts 80% pathogenic should be correct
        ~80% of the time. Poor calibration means the confidence scores are
        misleading, which is dangerous in clinical settings.

        Args:
            sequences: Test sequences
            labels: True labels (0/1)
            n_bins: Number of probability bins

        Returns:
            Dictionary with:
            - bin_edges: Probability bin boundaries
            - bin_counts: Number of samples per bin
            - bin_accuracy: Observed fraction of positives per bin
            - bin_confidence: Mean predicted probability per bin
            - ece: Expected Calibration Error (lower is better)
            - mce: Maximum Calibration Error
        """
        if not self.trained:
            raise ValueError("Model must be trained before calibration analysis")

        hidden, bloom_sig, bloom_sum, labels_t = self.prepare_dataset(sequences, labels)

        self.model.eval()
        with torch.no_grad():
            hidden = hidden.to(self.device)
            bloom_sig = bloom_sig.to(self.device)
            bloom_sum = bloom_sum.to(self.device)

            outputs = self.model(hidden, bloom_sig, bloom_sum).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()

        labels_arr = np.array(labels)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_accuracy = np.zeros(n_bins)
        bin_confidence = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins, dtype=int)

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if i == n_bins - 1:
                mask = (probs >= lo) & (probs <= hi)
            else:
                mask = (probs >= lo) & (probs < hi)

            bin_counts[i] = mask.sum()
            if bin_counts[i] > 0:
                bin_accuracy[i] = labels_arr[mask].mean()
                bin_confidence[i] = probs[mask].mean()

        # Expected Calibration Error (weighted by bin count)
        total = bin_counts.sum()
        if total > 0:
            ece = np.sum(
                bin_counts / total * np.abs(bin_accuracy - bin_confidence)
            )
        else:
            ece = 0.0

        # Maximum Calibration Error
        nonzero = bin_counts > 0
        if nonzero.any():
            mce = np.max(np.abs(bin_accuracy[nonzero] - bin_confidence[nonzero]))
        else:
            mce = 0.0

        print(f"\nCalibration Analysis ({n_bins} bins):")
        print(f"  Expected Calibration Error (ECE): {ece:.4f}")
        print(f"  Maximum Calibration Error (MCE): {mce:.4f}")
        print(f"  {'Bin':>8s} {'Count':>6s} {'Conf':>8s} {'Acc':>8s} {'|Gap|':>8s}")
        for i in range(n_bins):
            if bin_counts[i] > 0:
                gap = abs(bin_accuracy[i] - bin_confidence[i])
                print(f"  {bin_edges[i]:.1f}-{bin_edges[i+1]:.1f} "
                      f"{bin_counts[i]:6d} "
                      f"{bin_confidence[i]:8.3f} "
                      f"{bin_accuracy[i]:8.3f} "
                      f"{gap:8.3f}")

        return {
            'bin_edges': bin_edges,
            'bin_counts': bin_counts,
            'bin_accuracy': bin_accuracy,
            'bin_confidence': bin_confidence,
            'ece': ece,
            'mce': mce,
        }

    def cross_validate(
        self,
        sequences: list,
        labels: list,
        n_folds: int = 5,
        epochs: int = 50,
        batch_size: int = 16,
        **train_kwargs
    ) -> Dict[str, list]:
        """
        Perform k-fold cross-validation for more reliable performance estimates.

        Useful for small datasets where a single train/val/test split may
        produce high-variance metrics.

        Args:
            sequences: All sequences to use
            labels: All labels
            n_folds: Number of folds (default 5)
            epochs: Training epochs per fold
            batch_size: Batch size per fold
            **train_kwargs: Additional args passed to train()

        Returns:
            Dictionary with per-fold metrics and aggregated statistics
        """
        print(f"\n{'=' * 60}")
        print(f"{n_folds}-Fold Cross-Validation (BGPCA)")
        print(f"{'=' * 60}")

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(sequences, labels)):
            print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")

            train_seqs = [sequences[i] for i in train_idx]
            train_labs = [labels[i] for i in train_idx]
            val_seqs = [sequences[i] for i in val_idx]
            val_labs = [labels[i] for i in val_idx]

            self.train(
                train_sequences=train_seqs,
                train_labels=train_labs,
                val_sequences=val_seqs,
                val_labels=val_labs,
                epochs=epochs,
                batch_size=batch_size,
                **train_kwargs
            )

            metrics = self.evaluate(val_seqs, val_labs)
            fold_metrics.append(metrics)
            print(f"  Fold {fold_idx + 1}: "
                  f"Acc={metrics['accuracy']:.3f}, "
                  f"F1={metrics['f1_score']:.3f}, "
                  f"AUC={metrics['auc_roc']:.3f}")

        # Aggregate results
        metric_names = fold_metrics[0].keys()
        aggregated = {}
        for name in metric_names:
            values = [m[name] for m in fold_metrics]
            aggregated[f'{name}_mean'] = np.mean(values)
            aggregated[f'{name}_std'] = np.std(values)
            aggregated[f'{name}_folds'] = values

        print(f"\n{'=' * 60}")
        print(f"Cross-Validation Summary ({n_folds} folds):")
        for name in metric_names:
            mean = aggregated[f'{name}_mean']
            std = aggregated[f'{name}_std']
            print(f"  {name}: {mean:.3f} +/- {std:.3f}")
        print(f"{'=' * 60}")

        return aggregated

    def save(self, path: str):
        """Save model to disk."""
        if not self.trained:
            raise ValueError("Cannot save untrained model")

        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'd_model': self.model.d_model,
                'd_bloom': self.model.d_bloom,
                'n_bloom_scales': self.model.bloom_encoder.n_scales,
                'bloom_summary_dim': self.model.bloom_summary_dim,
                'n_heads': self.model.cross_attn_layers[0].n_heads,
                'n_cross_attn_layers': self.model.n_cross_attn_layers,
                'dropout': self.model.classifier[2].p,
            },
            'pipeline_config': {
                'max_tokens': self.max_tokens,
            },
            'architecture': 'BGPCA'
        }

        torch.save(save_dict, path)
        print(f"BGPCA model saved to {path}")

    def load(self, path: str):
        """Load model from disk."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        save_dict = torch.load(path, map_location=self.device, weights_only=False)

        if save_dict.get('architecture') != 'BGPCA':
            raise ValueError(
                f"Expected BGPCA model, got: {save_dict.get('architecture', 'unknown')}"
            )

        if 'model_config' not in save_dict or 'model_state_dict' not in save_dict:
            raise ValueError(f"Invalid model file format: {path}")

        config = save_dict['model_config']
        self.model = BloomGuidedClassifier(**config).to(self.device)
        self.model.load_state_dict(save_dict['model_state_dict'])
        self.trained = True

        if 'pipeline_config' in save_dict:
            pipe_cfg = save_dict['pipeline_config']
            self.max_tokens = pipe_cfg.get('max_tokens', self.max_tokens)

        print(f"BGPCA model loaded from {path}")
