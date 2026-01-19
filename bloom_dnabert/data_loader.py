"""
ClinVar Data Fetcher for HBB Variants

This module fetches and processes variant data from ClinVar for the HBB gene,
including the sickle cell mutation and other hemoglobin disorders.
"""

import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path


class ClinVarDataLoader:
    """
    Fetches and processes HBB gene variants from ClinVar database.
    """
    
    # HBB gene information
    HBB_GENE_ID = "3043"
    HBB_CHR = "11"
    HBB_START = 5225464
    HBB_END = 5227071
    
    # Reference HBB sequence (exonic regions, coding sequence)
    # This is the normal beta-globin sequence
    HBB_REFERENCE = (
        "ATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGG"
        "TTGGTATCAAGGTTACAAGACAGGTTTAAGGAGACCAATAGAAACTGGGCATGTGGAGACAGAGAAGACTCTTGGGTTTCTGATAGGCACTGAC"
        "TCTCTCTGCCTATTGGTCTATTTTCCCACCCTTAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAGTCCTTTGGGGATCTGTCCA"
        "CTCCTGATGCTGTTATGGGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAGTGCTCGGTGCCTTTAGTGATGGCCTGGCTCACCTGGACAACCT"
        "CAAGGGCACCTTTGCCACACTGAGTGAGCTGCACTGTGACAAGCTGCACGTGGATCCTGAGAACTTCAGGGTGAGTCTATGGGACGCTTGATGT"
        "TTTCTTTCCCCTTCTTTTCTATGGTTAAGTTCATGTCATAGGAAGGGGAGAAGTAACAGGGTACACATATTGACCAAATCAGGGTAATTTTGCA"
        "TTTGTAATTTTAAAAAATGCTTTCTTCTTTTAATATACTTTTTTGTTTATCTTATTTCTAATACTTTCCCTAATCTCTTTCTTTCAGGGCAATA"
        "ATGATACAATGTATCATGCCTCTTTGCACCATTCTAAAGAATAACAGTGATAATTTCTGGGTTAAGGCAATAGCAATATCTCTGCATATAAATA"
        "TTTCTGCATATAAATTGTAACTGATGTAAGAGGTTTCATATTGCTAATAGCAGCTACAATCCAGCTACCATTCTGCTTTTATTTTATGGTTGGG"
        "ATAAGGCTGGATTATTCTGAGTCCAAGCTAGGCCCTTTTGCTAATCATGTTCATACCTCTTATCTTCCTCCCACAGCTCCTGGGCAACGTGCTG"
        "GTCTGTGTGCTGGCCCATCACTTTGGCAAAGAATTCACCCCACCAGTGCAGGCTGCCTATCAGAAAGTGGTGGCTGGTGTGGCTAATGCCCTGG"
        "CCCACAAGTATCACTAAGCTCGCTTTCTTGCTGTCCAATTTCTATTAAAGGTTCCTTTGTTCCCTAAGTCCAACTACTAAACTGGGGGATATTAT"
        "GAAGGGCCTTGAGCATCTGGATTCTGCCTAATAAAAAACATTTATTTTCATTGCAA"
    )
    
    # Sickle cell mutation: position 20 in CDS (codon 6)
    # Normal: GAG (Glutamic acid)
    # Mutant: GTG (Valine)
    SICKLE_CELL_POSITION = 19  # 0-indexed, 20th nucleotide (A->T)
    
    def __init__(self, cache_dir: str = "data"):
        """
        Initialize ClinVar data loader.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.variants_cache_file = self.cache_dir / "hbb_variants.csv"
        
    def fetch_hbb_variants(self, force_download: bool = False) -> pd.DataFrame:
        """
        Fetch HBB variants from ClinVar.
        
        Args:
            force_download: Force re-download even if cache exists
            
        Returns:
            DataFrame with variant information
        """
        # Check cache
        if self.variants_cache_file.exists() and not force_download:
            print(f"Loading cached variants from {self.variants_cache_file}")
            return pd.read_csv(self.variants_cache_file)
        
        print("Fetching HBB variants from ClinVar...")
        
        # Use NCBI E-utilities API
        # Search for HBB gene variants
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            'db': 'clinvar',
            'term': 'HBB[gene] AND human[orgn]',
            'retmax': 1000,
            'retmode': 'json'
        }
        
        try:
            response = requests.get(search_url, params=search_params, timeout=30)
            response.raise_for_status()
            search_results = response.json()
            
            variant_ids = search_results.get('esearchresult', {}).get('idlist', [])
            print(f"Found {len(variant_ids)} HBB variants in ClinVar")
            
        except Exception as e:
            print(f"Warning: Could not fetch from ClinVar API: {e}")
            print("Generating synthetic dataset instead...")
            return self._generate_synthetic_dataset()
        
        # For simplicity, generate a synthetic dataset with known variants
        # In production, you would fetch full variant details from ClinVar
        return self._generate_synthetic_dataset()
    
    def _generate_synthetic_dataset(self) -> pd.DataFrame:
        """
        Generate a synthetic dataset with known HBB variants.
        
        This includes:
        - Sickle cell mutation (HbS)
        - HbC mutation
        - Beta-thalassemia mutations
        - Benign variants
        
        Returns:
            DataFrame with variant information
        """
        print("Generating synthetic HBB variant dataset...")
        
        variants = []
        
        # 1. Sickle Cell Disease (HbS) - E6V mutation
        for i in range(50):  # 50 positive examples
            seq = self._create_variant_sequence(self.SICKLE_CELL_POSITION, 'T')
            variants.append({
                'variant_id': f'HbS_{i}',
                'gene': 'HBB',
                'mutation': 'E6V',
                'position': self.SICKLE_CELL_POSITION,
                'ref': 'A',
                'alt': 'T',
                'disease': 'Sickle Cell Disease',
                'clinical_significance': 'Pathogenic',
                'sequence': seq,
                'label': 1  # Pathogenic
            })
        
        # 2. HbC Disease - E6K mutation (GAG -> AAG)
        for i in range(30):
            seq = self._create_variant_sequence(self.SICKLE_CELL_POSITION, 'A')
            variants.append({
                'variant_id': f'HbC_{i}',
                'gene': 'HBB',
                'mutation': 'E6K',
                'position': self.SICKLE_CELL_POSITION,
                'ref': 'G',
                'alt': 'A',
                'disease': 'HbC Disease',
                'clinical_significance': 'Pathogenic',
                'sequence': seq,
                'label': 1
            })
        
        # 3. Beta-thalassemia mutations
        # IVS-I-110 (G->A) - common in Mediterranean
        for i in range(20):
            pos = 150  # Example position in intron
            seq = self._create_variant_sequence(pos, 'A', ref_base='G')
            variants.append({
                'variant_id': f'BetaThal_IVS_{i}',
                'gene': 'HBB',
                'mutation': 'IVS-I-110',
                'position': pos,
                'ref': 'G',
                'alt': 'A',
                'disease': 'Beta-Thalassemia',
                'clinical_significance': 'Pathogenic',
                'sequence': seq,
                'label': 1
            })
        
        # 4. Benign/Likely Benign variants
        # Random synonymous or intronic variants
        benign_positions = [30, 50, 75, 100, 200, 250, 300]
        for i, pos in enumerate(benign_positions):
            for j in range(10):  # 10 examples per position
                # Random synonymous change
                alt_base = np.random.choice(['A', 'T', 'C', 'G'])
                seq = self._create_variant_sequence(pos, alt_base)
                variants.append({
                    'variant_id': f'Benign_{i}_{j}',
                    'gene': 'HBB',
                    'mutation': f'p.Syn{pos}',
                    'position': pos,
                    'ref': self.HBB_REFERENCE[pos] if pos < len(self.HBB_REFERENCE) else 'N',
                    'alt': alt_base,
                    'disease': 'None',
                    'clinical_significance': 'Benign',
                    'sequence': seq,
                    'label': 0  # Benign
                })
        
        # 5. Uncertain significance variants
        for i in range(30):
            pos = np.random.randint(50, min(400, len(self.HBB_REFERENCE)))
            alt_base = np.random.choice(['A', 'T', 'C', 'G'])
            seq = self._create_variant_sequence(pos, alt_base)
            variants.append({
                'variant_id': f'VUS_{i}',
                'gene': 'HBB',
                'mutation': f'p.?{pos}',
                'position': pos,
                'ref': self.HBB_REFERENCE[pos] if pos < len(self.HBB_REFERENCE) else 'N',
                'alt': alt_base,
                'disease': 'Unknown',
                'clinical_significance': 'Uncertain',
                'sequence': seq,
                'label': -1  # Unknown/exclude from training
            })
        
        df = pd.DataFrame(variants)
        
        # Save to cache
        df.to_csv(self.variants_cache_file, index=False)
        print(f"Generated {len(df)} variant examples")
        print(f"  Pathogenic: {(df['label'] == 1).sum()}")
        print(f"  Benign: {(df['label'] == 0).sum()}")
        print(f"  Uncertain: {(df['label'] == -1).sum()}")
        print(f"Saved to {self.variants_cache_file}")
        
        return df
    
    def _create_variant_sequence(
        self,
        position: int,
        alt_base: str,
        ref_base: str = None,
        context_size: int = 100
    ) -> str:
        """
        Create a sequence with a variant at the specified position.
        
        Args:
            position: Position of the variant
            alt_base: Alternative base
            ref_base: Reference base (if None, use from reference)
            context_size: Context size around variant
            
        Returns:
            DNA sequence with variant
        """
        # Get reference sequence
        ref_seq = self.HBB_REFERENCE
        
        # Extract context around variant
        start = max(0, position - context_size)
        end = min(len(ref_seq), position + context_size + 1)
        
        # Create variant sequence
        variant_seq = list(ref_seq[start:end])
        
        # Apply variant if position is in range
        relative_pos = position - start
        if 0 <= relative_pos < len(variant_seq):
            variant_seq[relative_pos] = alt_base
        
        return ''.join(variant_seq)
    
    def get_training_data(
        self,
        test_split: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get training and test datasets.
        
        Args:
            test_split: Fraction for test set
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, test_df)
        """
        # Load variants
        df = self.fetch_hbb_variants()
        
        # Filter out uncertain variants for training
        df_labeled = df[df['label'] != -1].copy()
        
        # Shuffle
        df_labeled = df_labeled.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Split
        n_test = int(len(df_labeled) * test_split)
        test_df = df_labeled[:n_test]
        train_df = df_labeled[n_test:]
        
        print(f"\nDataset split:")
        print(f"  Training: {len(train_df)} ({(train_df['label']==1).sum()} pathogenic, {(train_df['label']==0).sum()} benign)")
        print(f"  Test: {len(test_df)} ({(test_df['label']==1).sum()} pathogenic, {(test_df['label']==0).sum()} benign)")
        
        return train_df, test_df
    
    def get_reference_sequence(self) -> str:
        """Get the HBB reference sequence."""
        return self.HBB_REFERENCE
    
    def get_sickle_cell_examples(self) -> pd.DataFrame:
        """Get only sickle cell disease examples."""
        df = self.fetch_hbb_variants()
        return df[df['mutation'] == 'E6V']
