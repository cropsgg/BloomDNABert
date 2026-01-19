"""
Hybrid Classifier combining Bloom Filter features and DNABERT embeddings

This module implements a neural classifier that fuses:
1. Bloom filter k-mer hit features (fast pattern matching)
2. DNABERT-2 sequence embeddings (deep contextual understanding)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import pickle
from pathlib import Path


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
        
        # Batch normalization for stability
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        
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
        
        # First layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Second layer
        x = self.fc2(x)
        x = self.bn2(x)
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
        weight_decay: float = 0.01
    ) -> Dict[str, list]:
        """
        Train the hybrid classifier.
        
        Args:
            train_sequences: Training sequences
            train_labels: Training labels
            val_sequences: Validation sequences (optional)
            val_labels: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: L2 regularization
            
        Returns:
            Dictionary with training history
        """
        print("\n" + "="*60)
        print("Training Hybrid Classifier")
        print("="*60)
        
        # Prepare datasets
        train_bloom, train_dnabert, train_labels_t = self.prepare_dataset(train_sequences, train_labels)
        
        if val_sequences is not None:
            val_bloom, val_dnabert, val_labels_t = self.prepare_dataset(val_sequences, val_labels)
        
        # Initialize model
        self.model = HybridClassifier(
            dnabert_dim=train_dnabert.shape[1],
            bloom_dim=train_bloom.shape[1]
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Training loop
        n_samples = len(train_sequences)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        print(f"\nTraining settings:")
        print(f"  Samples: {n_samples}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Device: {self.device}")
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            
            # Shuffle data
            indices = torch.randperm(n_samples)
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch
                batch_bloom = train_bloom[batch_indices].to(self.device)
                batch_dnabert = train_dnabert[batch_indices].to(self.device)
                batch_labels = train_labels_t[batch_indices].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_dnabert, batch_bloom).squeeze()
                loss = criterion(outputs, batch_labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                epoch_loss += loss.item() * len(batch_indices)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                epoch_correct += (preds == batch_labels).sum().item()
            
            # Epoch statistics
            avg_loss = epoch_loss / n_samples
            avg_acc = epoch_correct / n_samples
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(avg_acc)
            
            # Validation
            if val_sequences is not None:
                val_loss, val_acc = self._evaluate(
                    val_bloom, val_dnabert, val_labels_t, criterion
                )
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
        
        self.trained = True
        print("\n✓ Training completed!")
        
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
    
    def predict(self, sequence: str) -> Dict[str, float]:
        """
        Predict whether a sequence is pathogenic.
        
        Args:
            sequence: DNA sequence
            
        Returns:
            Dictionary with prediction results
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        
        self.model.eval()
        
        # Extract features
        bloom_feat, dnabert_emb = self.extract_features(sequence)
        
        # Convert to tensors
        bloom_feat = torch.tensor(bloom_feat, dtype=torch.float32).unsqueeze(0).to(self.device)
        dnabert_emb = torch.tensor(dnabert_emb, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Predict
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
                'bloom_dim': self.model.bloom_dim
            }
        }
        
        torch.save(save_dict, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        save_dict = torch.load(path, map_location=self.device)
        
        self.model = HybridClassifier(**save_dict['model_config']).to(self.device)
        self.model.load_state_dict(save_dict['model_state_dict'])
        self.trained = True
        
        print(f"Model loaded from {path}")
