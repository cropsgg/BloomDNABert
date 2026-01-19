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
        
        The sickle cell mutation is at codon 6: GAG -> GTG (E6V)
        This method populates the Bloom filters with k-mers covering this
        and other known pathogenic variants in the HBB gene.
        """
        # HBB gene reference sequence (partial, around codon 6)
        # Normal: ...CAC GTG GAC TAC CCT...
        #         His Val Asp Tyr Pro
        # Sickle: ...CAC GTG GTC TAC CCT... (GAG->GTG in codon 6)
        #         His Val Val Tyr Pro
        
        # Known pathogenic variants in HBB gene
        pathogenic_sequences = [
            # Sickle cell (HbS) - E6V mutation region
            "CACGTGGTCTACCCTGAGGT",  # Mutant with GTG at codon 6
            "GTGGTCTACCCT",
            "CACGTGGTCTAC",
            "GTGGTCTACCCTGAGG",
            
            # Additional sickle cell k-mers
            "GTGGTC",  # Key mutant k-mer (k=6)
            "GTGGTCTA",  # k=8
            "GTGGTCTACC",  # k=10
            
            # HbC disease - E6K mutation
            "CACGTGAAGTACCCTGAGGT",
            "GTGAAGTAC",
            
            # Beta-thalassemia mutations (common ones)
            # These are insertions/deletions that create novel k-mers
            "CTGAGGAGAAGTCTGCCGTT",  # IVS-I-110 mutation region
            
            # Add more pathogenic variant k-mers as needed
        ]
        
        for seq in pathogenic_sequences:
            self.add_pathogenic_kmers(seq)
        
        print(f"Loaded pathogenic variants into Bloom filters:")
        for k in self.k_sizes:
            print(f"  k={k}: {self.pathogenic_kmer_count[k]} k-mers added")
    
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
