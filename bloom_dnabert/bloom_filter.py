"""
Multi-scale Bloom Filter Module for Pathogenic K-mer Detection

This module implements Bloom filters at multiple k-mer sizes (k=6, 8, 10)
to enable fast O(1) lookup of known pathogenic k-mers from the HBB gene region.
"""

import numpy as np
from typing import List, Dict, Tuple
from pybloom_live import BloomFilter


class MultiScaleBloomFilter:
    """
    Multi-scale Bloom filter for detecting pathogenic k-mers at different scales.
    
    Uses three Bloom filters with k=6, 8, 10 to capture patterns at different
    sequence granularities. This provides robust detection of known pathogenic
    variants including the sickle cell mutation.
    """
    
    def __init__(self, capacity: int = 100000, error_rate: float = 0.001):
        """
        Initialize multi-scale Bloom filters.
        
        Args:
            capacity: Expected number of k-mers per filter
            error_rate: False positive rate (default 0.1%)
        """
        self.k_sizes = [6, 8, 10]
        self.filters = {
            k: BloomFilter(capacity=capacity, error_rate=error_rate)
            for k in self.k_sizes
        }
        self.pathogenic_kmer_count = {k: 0 for k in self.k_sizes}
        
    def add_pathogenic_kmers(self, sequence: str, k: int = None):
        """
        Add all k-mers from a pathogenic sequence to the Bloom filters.
        
        Args:
            sequence: DNA sequence containing pathogenic variant
            k: Specific k-mer size (if None, adds to all filters)
        """
        sequence = sequence.upper().replace('N', '')  # Remove ambiguous bases
        
        k_list = [k] if k else self.k_sizes
        
        for k_size in k_list:
            if k_size not in self.filters:
                continue
                
            # Extract all k-mers
            for i in range(len(sequence) - k_size + 1):
                kmer = sequence[i:i + k_size]
                # Only add valid k-mers (A, T, C, G)
                if all(base in 'ATCG' for base in kmer):
                    self.filters[k_size].add(kmer)
                    self.pathogenic_kmer_count[k_size] += 1
    
    def add_pathogenic_kmer(self, kmer: str):
        """
        Add a single pathogenic k-mer to the appropriate filter.
        
        Args:
            kmer: K-mer sequence to add
        """
        kmer = kmer.upper()
        k = len(kmer)
        
        if k in self.filters and all(base in 'ATCG' for base in kmer):
            self.filters[k].add(kmer)
            self.pathogenic_kmer_count[k] += 1
    
    def check_sequence(self, sequence: str) -> Dict[int, List[bool]]:
        """
        Check which k-mers in a sequence match known pathogenic k-mers.
        
        Args:
            sequence: DNA sequence to check
            
        Returns:
            Dictionary mapping k-size to list of boolean flags indicating
            whether each k-mer at that position is in the pathogenic set
        """
        sequence = sequence.upper()
        results = {}
        
        for k in self.k_sizes:
            hits = []
            for i in range(len(sequence) - k + 1):
                kmer = sequence[i:i + k]
                if all(base in 'ATCG' for base in kmer):
                    hits.append(kmer in self.filters[k])
                else:
                    hits.append(False)
            results[k] = hits
            
        return results
    
    def get_hit_features(self, sequence: str) -> Dict[str, float]:
        """
        Extract Bloom filter hit features for classification.
        
        Features include:
        - Hit counts at each k-size
        - Hit ratios (hits / total k-mers)
        - Positional features (where hits occur)
        - Multi-scale hit density
        
        Args:
            sequence: DNA sequence to analyze
            
        Returns:
            Dictionary of feature names and values
        """
        sequence = sequence.upper()
        hits = self.check_sequence(sequence)
        features = {}
        
        # Basic hit statistics for each k-size
        for k in self.k_sizes:
            hit_list = hits[k]
            if len(hit_list) > 0:
                features[f'hit_count_k{k}'] = sum(hit_list)
                features[f'hit_ratio_k{k}'] = sum(hit_list) / len(hit_list)
                
                # Positional features - where do hits occur?
                hit_positions = [i for i, h in enumerate(hit_list) if h]
                if hit_positions:
                    features[f'first_hit_pos_k{k}'] = hit_positions[0] / len(hit_list)
                    features[f'last_hit_pos_k{k}'] = hit_positions[-1] / len(hit_list)
                    features[f'hit_span_k{k}'] = (hit_positions[-1] - hit_positions[0]) / len(hit_list)
                else:
                    features[f'first_hit_pos_k{k}'] = 0.0
                    features[f'last_hit_pos_k{k}'] = 0.0
                    features[f'hit_span_k{k}'] = 0.0
            else:
                features[f'hit_count_k{k}'] = 0
                features[f'hit_ratio_k{k}'] = 0.0
                features[f'first_hit_pos_k{k}'] = 0.0
                features[f'last_hit_pos_k{k}'] = 0.0
                features[f'hit_span_k{k}'] = 0.0
        
        # Multi-scale aggregated features
        all_ratios = [features[f'hit_ratio_k{k}'] for k in self.k_sizes]
        features['mean_hit_ratio'] = np.mean(all_ratios)
        features['max_hit_ratio'] = np.max(all_ratios)
        features['min_hit_ratio'] = np.min(all_ratios)
        
        return features
    
    def get_feature_vector(self, sequence: str) -> np.ndarray:
        """
        Get feature vector as numpy array for model input.
        
        Args:
            sequence: DNA sequence to analyze
            
        Returns:
            Numpy array of features (fixed length)
        """
        features = self.get_hit_features(sequence)
        
        # Fixed order for consistency
        feature_order = []
        for k in self.k_sizes:
            feature_order.extend([
                f'hit_count_k{k}',
                f'hit_ratio_k{k}',
                f'first_hit_pos_k{k}',
                f'last_hit_pos_k{k}',
                f'hit_span_k{k}'
            ])
        feature_order.extend(['mean_hit_ratio', 'max_hit_ratio', 'min_hit_ratio'])
        
        return np.array([features[f] for f in feature_order], dtype=np.float32)
    
    def get_hit_positions(self, sequence: str, k: int = 8) -> List[Tuple[int, str]]:
        """
        Get positions and sequences of pathogenic k-mer hits.
        
        Args:
            sequence: DNA sequence to check
            k: K-mer size to use (default 8)
            
        Returns:
            List of (position, k-mer) tuples for hits
        """
        sequence = sequence.upper()
        hits = []
        
        if k not in self.filters:
            return hits
            
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i + k]
            if all(base in 'ATCG' for base in kmer) and kmer in self.filters[k]:
                hits.append((i, kmer))
                
        return hits
    
    def load_hbb_pathogenic_variants(self):
        """
        Load known HBB pathogenic variants including sickle cell mutation.
        
        Mutations verified against NCBI ClinVar / NM_000518.5:
        - E6V (HbS): c.20A>T, codon 7 GAG→GTG, position 19 (0-indexed)
        - E6K (HbC): c.19G>A, codon 7 GAG→AAG, position 18 (0-indexed)
        - E26K (HbE): c.79G>A, codon 27 GAG→AAG, position 78 (0-indexed)
        
        Normal codon 7 region:  ...CATCTGACTCCT GAG GAGAAGTCTGCC...
        HbS mutant:             ...CATCTGACTCCT GTG GAGAAGTCTGCC...
        HbC mutant:             ...CATCTGACTCCT AAG GAGAAGTCTGCC...
        """
        # Reference region around codon 7 (positions 6-29):
        # CATCTGACTCCTGAGGAGAAGTCTGCC
        #                  ^^^ codon 7 = GAG (Glu) at positions 18-20
        
        pathogenic_sequences = [
            # Sickle cell (HbS) - E6V: GAG→GTG (pos 19: A→T)
            # Mutant context: ...CCTGTGGAG... (GTG at codon 7)
            "CATCTGACTCCTGTGGAGAAGTCTGCC",  # Full mutant context
            "ACTCCTGTGGAG",   # Shorter context around mutation
            "CCTGTG",         # Key mutant k-mer (k=6)
            "CCTGTGGA",       # k=8
            "CCTGTGGAGA",     # k=10
            "TGACTCCTGTGGAGAA",  # Extended context
            
            # HbC disease - E6K: GAG→AAG (pos 18: G→A)
            # Mutant context: ...CCTAAGGAG... (AAG at codon 7)
            "CATCTGACTCCTAAGGAGAAGTCTGCC",  # Full mutant context
            "ACTCCTAAGGAG",   # Shorter context
            "CCTAAG",         # Key mutant k-mer (k=6)
            "CCTAAGGA",       # k=8
            "CCTAAGGAGA",     # k=10
            
            # HbE disease - E26K: GAG→AAG (pos 78: G→A)
            # In exon 1, codon 27
            "GCCCTGGGCAAGTTGGTATCAAGGTTACAAG",  # HbE region (approximate)
        ]
        
        for seq in pathogenic_sequences:
            self.add_pathogenic_kmers(seq)
        
        print(f"Loaded pathogenic variants into Bloom filters:")
        for k in self.k_sizes:
            print(f"  k={k}: {self.pathogenic_kmer_count[k]} k-mers added")
    
    def get_positional_signal(self, sequence: str) -> np.ndarray:
        """
        Compute per-position Bloom filter activation signal.
        
        For each nucleotide position, computes the fraction of k-mers
        overlapping that position that are found in the pathogenic Bloom filter.
        This preserves spatial information about WHERE pathogenic patterns
        are detected, unlike summary statistics that collapse position info.
        
        Args:
            sequence: DNA sequence
            
        Returns:
            numpy array of shape [seq_len, n_k_sizes] with activation scores
            in [0, 1] for each position and k-mer scale
        """
        sequence = sequence.upper()
        seq_len = len(sequence)
        n_scales = len(self.k_sizes)
        signal = np.zeros((seq_len, n_scales), dtype=np.float32)
        
        hits = self.check_sequence(sequence)
        
        for k_idx, k in enumerate(self.k_sizes):
            hit_list = hits[k]
            hit_counts = np.zeros(seq_len, dtype=np.float32)
            overlap_counts = np.zeros(seq_len, dtype=np.float32)
            
            for i in range(len(hit_list)):
                end_pos = min(i + k, seq_len)
                overlap_counts[i:end_pos] += 1.0
                if hit_list[i]:
                    hit_counts[i:end_pos] += 1.0
            
            # Activation = fraction of overlapping k-mers that are hits
            mask = overlap_counts > 0
            signal[mask, k_idx] = hit_counts[mask] / overlap_counts[mask]
        
        return signal
    
    def get_token_aligned_signal(
        self,
        sequence: str,
        token_spans: List[Tuple[int, int]]
    ) -> np.ndarray:
        """
        Compute Bloom activation signal aligned to DNABERT token positions.
        
        Aggregates per-nucleotide Bloom signals to match the tokenization
        used by DNABERT, enabling cross-modal position-level alignment.
        
        Args:
            sequence: DNA sequence
            token_spans: List of (start, end) nucleotide positions per token
            
        Returns:
            numpy array of shape [num_tokens, n_k_sizes]
        """
        nuc_signal = self.get_positional_signal(sequence)
        n_tokens = len(token_spans)
        n_scales = len(self.k_sizes)
        token_signal = np.zeros((n_tokens, n_scales), dtype=np.float32)
        
        for t_idx, (start, end) in enumerate(token_spans):
            if end > start and start < len(nuc_signal):
                actual_end = min(end, len(nuc_signal))
                token_signal[t_idx] = nuc_signal[start:actual_end].mean(axis=0)
        
        return token_signal
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get statistics about the Bloom filters.
        
        Returns:
            Dictionary with filter statistics
        """
        return {
            f'pathogenic_kmers_k{k}': self.pathogenic_kmer_count[k]
            for k in self.k_sizes
        }
