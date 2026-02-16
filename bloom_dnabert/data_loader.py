"""
ClinVar Data Loader for HBB Variants

Generates biologically accurate training data for the HBB gene region,
including the sickle cell mutation and other hemoglobin disorders.

Phase 1 data quality fixes applied:
- Verified HBB reference against NCBI NM_000518.5
- Fixed HbC (E6K) mutation position: c.19G>A = position 18 (0-indexed)
- Confirmed sickle cell (E6V) position: c.20A>T = position 19 (0-indexed)
- Added sequence diversity (variable context, background SNPs)
- Verified benign variants are truly synonymous
- Proper 60/20/20 stratified train/val/test split with no leakage
"""

import pandas as pd
import numpy as np
import requests
import time
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split


# Standard genetic code (DNA codons -> amino acids)
CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}


class ClinVarDataLoader:
    """
    Generates biologically accurate HBB gene variant datasets.
    """

    HBB_GENE_ID = "3043"
    HBB_CHR = "11"
    HBB_START = 5225464
    HBB_END = 5227071

    # HBB reference: genomic sequence starting at ATG, including introns.
    # Exon 1 coding region: positions 0-91 (30 codons + 2 nt of codon 31)
    # Verified: positions 0-91 match NCBI NM_000518.5 CDS exactly.
    HBB_REFERENCE = (
        "ATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTG"
        "AACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGG"
        "TTGGTATCAAGGTTACAAGACAGGTTTAAGGAGACCAATAGAAACTGGGCATGTGGAG"
        "ACAGAGAAGACTCTTGGGTTTCTGATAGGCACTGAC"
        "TCTCTCTGCCTATTGGTCTATTTTCCCACCCTTAGGCTGCTGGTGGTCTACCCTTGG"
        "ACCCAGAGGTTCTTTGAGTCCTTTGGGGATCTGTCCA"
        "CTCCTGATGCTGTTATGGGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAGTGCTCG"
        "GTGCCTTTAGTGATGGCCTGGCTCACCTGGACAACCT"
        "CAAGGGCACCTTTGCCACACTGAGTGAGCTGCACTGTGACAAGCTGCACGTGGATCC"
        "TGAGAACTTCAGGGTGAGTCTATGGGACGCTTGATGT"
        "TTTCTTTCCCCTTCTTTTCTATGGTTAAGTTCATGTCATAGGAAGGGGAGAAGTAACA"
        "GGGTACACATATTGACCAAATCAGGGTAATTTTGCA"
        "TTTGTAATTTTAAAAAATGCTTTCTTCTTTTAATATACTTTTTTGTTTATCTTATTTC"
        "TAATACTTTCCCTAATCTCTTTCTTTCAGGGCAATA"
        "ATGATACAATGTATCATGCCTCTTTGCACCATTCTAAAGAATAACAGTGATAATTTCT"
        "GGGTTAAGGCAATAGCAATATCTCTGCATATAAATA"
        "TTTCTGCATATAAATTGTAACTGATGTAAGAGGTTTCATATTGCTAATAGCAGCTACA"
        "ATCCAGCTACCATTCTGCTTTTATTTTATGGTTGGG"
        "ATAAGGCTGGATTATTCTGAGTCCAAGCTAGGCCCTTTTGCTAATCATGTTCATACCT"
        "CTTATCTTCCTCCCACAGCTCCTGGGCAACGTGCTG"
        "GTCTGTGTGCTGGCCCATCACTTTGGCAAAGAATTCACCCCACCAGTGCAGGCTGCCT"
        "ATCAGAAAGTGGTGGCTGGTGTGGCTAATGCCCTGG"
        "CCCACAAGTATCACTAAGCTCGCTTTCTTGCTGTCCAATTTCTATTAAAGGTTCCTTT"
        "GTTCCCTAAGTCCAACTACTAAACTGGGGGATATTAT"
        "GAAGGGCCTTGAGCATCTGGATTCTGCCTAATAAAAAACATTTATTTTCATTGCAA"
    )

    # Exon 1 boundary: positions 0-91 are coding (verified against NCBI)
    EXON1_END = 92

    # Sickle cell mutation: c.20A>T (p.Glu7Val / E6V in traditional numbering)
    # NCBI ClinVar: NM_000518.5(HBB):c.20A>T
    # CDS position 20 (1-indexed) = position 19 (0-indexed)
    # Codon 7: G(18)-A(19)-G(20) = GAG (Glu) -> GTG (Val)
    SICKLE_CELL_POSITION = 19  # 0-indexed, verified against NCBI

    # HbC mutation: c.19G>A (p.Glu7Lys / E6K in traditional numbering)
    # CDS position 19 (1-indexed) = position 18 (0-indexed)
    # Codon 7: G(18)-A(19)-G(20) = GAG (Glu) -> AAG (Lys)
    HBC_POSITION = 18  # 0-indexed, verified against NCBI

    # IVS-I-110: Intronic mutation, approximately position 201 in this reference
    # (Intron 1 starts at ~position 92; IVS-I-110 = position 92 + 109 = 201)
    IVS1_110_POSITION = 201

    def __init__(self, cache_dir: str = "data", random_seed: int = 42):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.variants_cache_file = self.cache_dir / "hbb_variants.csv"
        self.rng = np.random.RandomState(random_seed)

    def fetch_hbb_variants(self, force_download: bool = False) -> pd.DataFrame:
        """
        Fetch HBB variant dataset.

        Attempts to download real ClinVar variants first, then supplements
        with synthetic data. Falls back entirely to synthetic if API fails.
        """
        if self.variants_cache_file.exists() and not force_download:
            print(f"Loading cached variants from {self.variants_cache_file}")
            df = pd.read_csv(self.variants_cache_file)
            if 'version' in df.columns and df['version'].iloc[0] >= 3:
                return df
            print("Stale cache detected. Regenerating...")

        # Try real ClinVar data, supplement with synthetic
        clinvar_df = self._fetch_clinvar_variants()
        synthetic_df = self._generate_synthetic_dataset()

        if clinvar_df is not None and len(clinvar_df) > 0:
            # Combine: real ClinVar + synthetic for diversity
            combined = pd.concat([clinvar_df, synthetic_df], ignore_index=True)
            combined['version'] = 3
            combined.to_csv(self.variants_cache_file, index=False)
            n_real = len(clinvar_df)
            n_synth = len(synthetic_df)
            print(f"\nCombined dataset: {n_real} ClinVar + {n_synth} synthetic = {len(combined)}")
            return combined

        # Fallback: synthetic only
        return synthetic_df

    def _fetch_clinvar_variants(self) -> Optional[pd.DataFrame]:
        """
        Fetch real HBB variants from ClinVar via NCBI E-utilities API.

        Queries for single nucleotide variants in the HBB gene with known
        clinical significance (pathogenic, likely pathogenic, benign,
        likely benign). Returns None if the API is unreachable.
        """
        ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

        print("Fetching real HBB variants from ClinVar...")
        try:
            # Search ClinVar for HBB single nucleotide variants
            search_params = {
                'db': 'clinvar',
                'term': 'HBB[gene] AND "single nucleotide variant"[variant type] AND ("pathogenic"[clinical significance] OR "likely pathogenic"[clinical significance] OR "benign"[clinical significance] OR "likely benign"[clinical significance])',
                'retmax': 500,
                'retmode': 'json',
            }

            resp = requests.get(ESEARCH_URL, params=search_params, timeout=15)
            resp.raise_for_status()
            search_data = resp.json()

            id_list = search_data.get('esearchresult', {}).get('idlist', [])
            if not id_list:
                print("  No ClinVar results found.")
                return None

            print(f"  Found {len(id_list)} ClinVar variant IDs")

            # Fetch summaries in batches
            variants = []
            batch_size = 50
            for batch_start in range(0, len(id_list), batch_size):
                batch_ids = id_list[batch_start:batch_start + batch_size]
                summary_params = {
                    'db': 'clinvar',
                    'id': ','.join(batch_ids),
                    'retmode': 'json',
                }
                resp = requests.get(ESUMMARY_URL, params=summary_params, timeout=15)
                resp.raise_for_status()
                summary_data = resp.json()

                for uid in batch_ids:
                    doc = summary_data.get('result', {}).get(uid)
                    if doc is None:
                        continue
                    variant = self._parse_clinvar_record(doc)
                    if variant is not None:
                        variants.append(variant)

                # Respect NCBI rate limit
                time.sleep(0.35)

            if not variants:
                print("  No usable variants parsed from ClinVar.")
                return None

            df = pd.DataFrame(variants)
            n_path = (df['label'] == 1).sum()
            n_ben = (df['label'] == 0).sum()
            print(f"  Parsed {len(df)} ClinVar variants ({n_path} pathogenic, {n_ben} benign)")
            return df

        except (requests.RequestException, KeyError, ValueError) as e:
            print(f"  ClinVar API unavailable ({type(e).__name__}: {e}). Using synthetic data only.")
            return None

    def _parse_clinvar_record(self, doc: dict) -> Optional[dict]:
        """
        Parse a single ClinVar eSummary record into a variant dict.
        Returns None if the record can't be used.
        """
        try:
            title = doc.get('title', '')
            clinical_sig = doc.get('clinical_significance', {})
            if isinstance(clinical_sig, dict):
                sig_desc = clinical_sig.get('description', '').lower()
            else:
                sig_desc = str(clinical_sig).lower()

            # Skip ambiguous/conflicting significance upfront
            skip_terms = ['conflicting', 'uncertain', 'not provided', 'other']
            if any(term in sig_desc for term in skip_terms):
                return None

            # Determine label from clean significance
            if 'pathogenic' in sig_desc and 'benign' not in sig_desc:
                label = 1
                variant_type = 'ClinVar_pathogenic'
            elif 'benign' in sig_desc and 'pathogenic' not in sig_desc:
                label = 0
                variant_type = 'ClinVar_benign'
            else:
                return None  # Skip dual annotations

            # Try to extract HGVS coding notation (e.g., c.20A>T)
            hgvs_c = ''
            variation_set = doc.get('variation_set', [])
            if isinstance(variation_set, list):
                for vs in variation_set:
                    cdna_change = vs.get('cdna_change', '')
                    if cdna_change:
                        hgvs_c = cdna_change
                        break

            # Extract the CDS position from HGVS c. notation
            pos_match = re.search(r'c\.(\d+)', hgvs_c)
            if pos_match:
                cds_pos_1indexed = int(pos_match.group(1))
                cds_pos_0indexed = cds_pos_1indexed - 1
            else:
                cds_pos_0indexed = -1

            # Generate a sequence context around the variant
            if 0 <= cds_pos_0indexed < len(self.HBB_REFERENCE):
                alt_match = re.search(r'>([ATCG])$', hgvs_c)
                alt_base = alt_match.group(1) if alt_match else None

                if alt_base and alt_base != self.HBB_REFERENCE[cds_pos_0indexed]:
                    seq = self._create_variant_sequence(
                        position=cds_pos_0indexed,
                        alt_base=alt_base,
                        context_size=100,
                        context_jitter=10,
                        background_snp_rate=0.002,
                        protected_positions=[cds_pos_0indexed]
                    )
                else:
                    seq = None
            else:
                seq = None

            if seq is None:
                # Fallback: use reference with some context if we can't build the variant
                start = max(0, cds_pos_0indexed - 100) if cds_pos_0indexed >= 0 else 0
                end = min(len(self.HBB_REFERENCE), start + 200)
                seq = self.HBB_REFERENCE[start:end]
                if len(seq) < 20:
                    return None

            return {
                'variant_id': f'ClinVar_{doc.get("uid", "unknown")}',
                'gene': 'HBB',
                'mutation': title[:60] if title else 'unknown',
                'hgvs_c': hgvs_c,
                'hgvs_p': doc.get('protein_change', ''),
                'position': cds_pos_0indexed if cds_pos_0indexed >= 0 else -1,
                'ref': self.HBB_REFERENCE[cds_pos_0indexed] if 0 <= cds_pos_0indexed < len(self.HBB_REFERENCE) else '',
                'alt': alt_base if 'alt_base' in dir() and alt_base else '',
                'disease': title[:80] if title else '',
                'clinical_significance': sig_desc,
                'variant_type': variant_type,
                'sequence': seq,
                'label': label,
                'version': 3,
            }
        except Exception:
            return None

    def _get_codon_at_cds_position(self, cds_pos: int) -> Tuple[str, int, int]:
        """
        Get the codon containing a CDS position.

        Args:
            cds_pos: 0-indexed position in the CDS

        Returns:
            (codon_str, codon_start, position_within_codon)
        """
        codon_index = cds_pos // 3
        codon_start = codon_index * 3
        pos_in_codon = cds_pos - codon_start
        codon = self.HBB_REFERENCE[codon_start:codon_start + 3]
        return codon, codon_start, pos_in_codon

    def _is_synonymous(self, cds_pos: int, alt_base: str) -> bool:
        """
        Check if a substitution at a CDS position is synonymous
        (does not change the amino acid).
        """
        if cds_pos >= self.EXON1_END:
            return True  # Intronic positions are not in coding region

        codon, codon_start, pos_in_codon = self._get_codon_at_cds_position(cds_pos)
        if len(codon) < 3:
            return False

        ref_aa = CODON_TABLE.get(codon, '?')
        mut_codon = list(codon)
        mut_codon[pos_in_codon] = alt_base
        mut_codon_str = ''.join(mut_codon)
        mut_aa = CODON_TABLE.get(mut_codon_str, '?')

        return ref_aa == mut_aa and ref_aa != '?' and ref_aa != '*'

    def _get_synonymous_change(self, cds_pos: int) -> Optional[str]:
        """
        Find a synonymous base change at a CDS position.
        Returns the alt base, or None if no synonymous change exists.
        """
        if cds_pos >= self.EXON1_END or cds_pos >= len(self.HBB_REFERENCE):
            return None

        ref_base = self.HBB_REFERENCE[cds_pos]
        candidates = [b for b in 'ATCG' if b != ref_base and self._is_synonymous(cds_pos, b)]

        if candidates:
            return self.rng.choice(candidates)
        return None

    def _create_variant_sequence(
        self,
        position: int,
        alt_base: str,
        context_size: int = 100,
        context_jitter: int = 0,
        background_snp_rate: float = 0.0,
        protected_positions: List[int] = None
    ) -> str:
        """
        Create a variant sequence with controlled diversity.

        Args:
            position: Variant position in the reference
            alt_base: Alternative base to introduce
            context_size: Base context window around the variant
            context_jitter: Random offset applied to the context window
            background_snp_rate: Rate of random background SNPs for diversity
            protected_positions: Positions that must not receive background SNPs
        """
        ref_seq = self.HBB_REFERENCE
        if position >= len(ref_seq):
            return ref_seq[:2 * context_size]

        # Validate alt differs from ref
        ref_base = ref_seq[position]
        if alt_base == ref_base:
            return None  # No-op mutation, skip

        # Apply context jitter for diversity
        jitter = 0
        if context_jitter > 0:
            jitter = self.rng.randint(-context_jitter, context_jitter + 1)

        start = max(0, position - context_size + jitter)
        end = min(len(ref_seq), position + context_size + 1 + jitter)

        variant_seq = list(ref_seq[start:end])

        # Apply the primary variant
        relative_pos = position - start
        if 0 <= relative_pos < len(variant_seq):
            variant_seq[relative_pos] = alt_base

        # Apply background SNPs for diversity (simulating individual variation)
        if background_snp_rate > 0 and len(variant_seq) > 0:
            n_snps = max(0, self.rng.poisson(background_snp_rate * len(variant_seq)))
            if protected_positions is None:
                protected_positions = []
            protected_relative = {p - start for p in protected_positions if start <= p < end}
            protected_relative.add(relative_pos)  # Never mutate the primary variant

            available = [
                i for i in range(len(variant_seq))
                if i not in protected_relative
            ]

            if available and n_snps > 0:
                snp_positions = self.rng.choice(
                    available, size=min(n_snps, len(available)), replace=False
                )
                for sp in snp_positions:
                    orig = variant_seq[sp]
                    alts = [b for b in 'ATCG' if b != orig]
                    variant_seq[sp] = self.rng.choice(alts)

        return ''.join(variant_seq)

    def _generate_synthetic_dataset(self) -> pd.DataFrame:
        """
        Generate a biologically accurate synthetic dataset.

        Key improvements over v1:
        - HbC mutation at correct position (18, not 19)
        - Each example has unique sequence (variable context + background SNPs)
        - Benign variants verified as synonymous
        - Proper validation of all mutations
        """
        print("Generating synthetic HBB variant dataset (v2 - Phase 1 fixes)...")

        variants = []

        # === 1. Sickle Cell Disease (HbS) - E6V ===
        # c.20A>T at position 19 (0-indexed), GAG→GTG
        # Protect the codon 7 region (positions 18-20)
        hbs_protected = [18, 19, 20]
        for i in range(60):
            seq = self._create_variant_sequence(
                position=self.SICKLE_CELL_POSITION,
                alt_base='T',
                context_size=100,
                context_jitter=20,
                background_snp_rate=0.005,
                protected_positions=hbs_protected
            )
            if seq is None:
                continue
            variants.append({
                'variant_id': f'HbS_{i}',
                'gene': 'HBB',
                'mutation': 'E6V',
                'hgvs_c': 'c.20A>T',
                'hgvs_p': 'p.Glu7Val',
                'position': self.SICKLE_CELL_POSITION,
                'ref': 'A',
                'alt': 'T',
                'disease': 'Sickle Cell Disease',
                'clinical_significance': 'Pathogenic',
                'variant_type': 'HbS',
                'sequence': seq,
                'label': 1,
                'version': 2
            })

        # === 2. HbC Disease - E6K ===
        # c.19G>A at position 18 (0-indexed), GAG→AAG
        # FIXED: was incorrectly using position 19 (A→A = no-op)
        hbc_protected = [18, 19, 20]
        for i in range(40):
            seq = self._create_variant_sequence(
                position=self.HBC_POSITION,
                alt_base='A',
                context_size=100,
                context_jitter=20,
                background_snp_rate=0.005,
                protected_positions=hbc_protected
            )
            if seq is None:
                continue
            variants.append({
                'variant_id': f'HbC_{i}',
                'gene': 'HBB',
                'mutation': 'E6K',
                'hgvs_c': 'c.19G>A',
                'hgvs_p': 'p.Glu7Lys',
                'position': self.HBC_POSITION,
                'ref': 'G',
                'alt': 'A',
                'disease': 'HbC Disease',
                'clinical_significance': 'Pathogenic',
                'variant_type': 'HbC',
                'sequence': seq,
                'label': 1,
                'version': 2
            })

        # === 3. HbE Disease - E26K ===
        # c.79G>A at position 78 (0-indexed), GAG→AAG
        # Common in Southeast Asia
        hbe_position = 78
        hbe_protected = [78, 79, 80]
        for i in range(25):
            ref_base = self.HBB_REFERENCE[hbe_position]
            if ref_base != 'G':
                break
            seq = self._create_variant_sequence(
                position=hbe_position,
                alt_base='A',
                context_size=100,
                context_jitter=15,
                background_snp_rate=0.005,
                protected_positions=hbe_protected
            )
            if seq is None:
                continue
            variants.append({
                'variant_id': f'HbE_{i}',
                'gene': 'HBB',
                'mutation': 'E26K',
                'hgvs_c': 'c.79G>A',
                'hgvs_p': 'p.Glu27Lys',
                'position': hbe_position,
                'ref': 'G',
                'alt': 'A',
                'disease': 'HbE Disease',
                'clinical_significance': 'Pathogenic',
                'variant_type': 'HbE',
                'sequence': seq,
                'label': 1,
                'version': 2
            })

        # === 4. Beta-thalassemia (intronic) ===
        # IVS-I-110: G>A in intron 1 (position ~201 in our reference)
        ivs_position = self.IVS1_110_POSITION
        if ivs_position < len(self.HBB_REFERENCE):
            for i in range(25):
                seq = self._create_variant_sequence(
                    position=ivs_position,
                    alt_base='A',
                    context_size=100,
                    context_jitter=15,
                    background_snp_rate=0.005,
                    protected_positions=[ivs_position]
                )
                if seq is None:
                    continue
                variants.append({
                    'variant_id': f'BetaThal_IVS_{i}',
                    'gene': 'HBB',
                    'mutation': 'IVS-I-110',
                    'hgvs_c': 'c.93-21G>A',
                    'hgvs_p': 'splicing',
                    'position': ivs_position,
                    'ref': self.HBB_REFERENCE[ivs_position],
                    'alt': 'A',
                    'disease': 'Beta-Thalassemia',
                    'clinical_significance': 'Pathogenic',
                    'variant_type': 'BetaThal',
                    'sequence': seq,
                    'label': 1,
                    'version': 2
                })

        # === 5. Benign variants (verified synonymous) ===
        # Only use third-codon-position (wobble) changes that preserve amino acid
        synonymous_positions = []
        for codon_idx in range(1, 30):  # Codons 2-30 in exon 1
            third_pos = codon_idx * 3 + 2  # Third position of each codon
            if third_pos < self.EXON1_END:
                alt = self._get_synonymous_change(third_pos)
                if alt is not None:
                    synonymous_positions.append((third_pos, alt))

        # Also add intronic positions far from splice sites (safe benign)
        intronic_safe_positions = list(range(
            self.EXON1_END + 20,
            min(self.EXON1_END + 100, len(self.HBB_REFERENCE))
        ))

        benign_count = 0
        for pos, alt in synonymous_positions:
            n_per_pos = 8
            for j in range(n_per_pos):
                seq = self._create_variant_sequence(
                    position=pos,
                    alt_base=alt,
                    context_size=100,
                    context_jitter=20,
                    background_snp_rate=0.003,
                    protected_positions=[pos]
                )
                if seq is None:
                    continue
                benign_count += 1
                variants.append({
                    'variant_id': f'Benign_syn_{pos}_{j}',
                    'gene': 'HBB',
                    'mutation': f'c.{pos+1}synonymous',
                    'hgvs_c': f'c.{pos+1}{self.HBB_REFERENCE[pos]}>{alt}',
                    'hgvs_p': 'p.=',
                    'position': pos,
                    'ref': self.HBB_REFERENCE[pos],
                    'alt': alt,
                    'disease': 'None',
                    'clinical_significance': 'Benign',
                    'variant_type': 'Benign_synonymous',
                    'sequence': seq,
                    'label': 0,
                    'version': 2
                })

        # Intronic benign variants (far from splice sites)
        for j in range(30):
            if not intronic_safe_positions:
                break
            pos = self.rng.choice(intronic_safe_positions)
            ref_base = self.HBB_REFERENCE[pos]
            alt = self.rng.choice([b for b in 'ATCG' if b != ref_base])
            seq = self._create_variant_sequence(
                position=pos,
                alt_base=alt,
                context_size=100,
                context_jitter=20,
                background_snp_rate=0.003,
                protected_positions=[pos]
            )
            if seq is None:
                continue
            variants.append({
                'variant_id': f'Benign_intronic_{j}',
                'gene': 'HBB',
                'mutation': f'intronic_{pos}',
                'hgvs_c': f'c.93+{pos - self.EXON1_END + 1}{ref_base}>{alt}',
                'hgvs_p': 'p.?',
                'position': pos,
                'ref': ref_base,
                'alt': alt,
                'disease': 'None',
                'clinical_significance': 'Benign',
                'variant_type': 'Benign_intronic',
                'sequence': seq,
                'label': 0,
                'version': 2
            })

        # === 6. Variants of Uncertain Significance (VUS) ===
        # Missense variants at non-critical positions
        for i in range(30):
            # Pick a coding position that is NOT in the critical codon 7 region
            pos = self.rng.choice([
                p for p in range(3, self.EXON1_END - 2)
                if p not in [18, 19, 20, 78, 79, 80]
            ])
            ref_base = self.HBB_REFERENCE[pos]
            # Pick a NON-synonymous change (missense)
            non_syn = [
                b for b in 'ATCG'
                if b != ref_base and not self._is_synonymous(pos, b)
            ]
            if not non_syn:
                continue
            alt = self.rng.choice(non_syn)
            seq = self._create_variant_sequence(
                position=pos,
                alt_base=alt,
                context_size=100,
                context_jitter=15,
                background_snp_rate=0.003,
                protected_positions=[pos]
            )
            if seq is None:
                continue
            variants.append({
                'variant_id': f'VUS_{i}',
                'gene': 'HBB',
                'mutation': f'missense_{pos}',
                'hgvs_c': f'c.{pos+1}{ref_base}>{alt}',
                'hgvs_p': 'p.?',
                'position': pos,
                'ref': ref_base,
                'alt': alt,
                'disease': 'Unknown',
                'clinical_significance': 'Uncertain',
                'variant_type': 'VUS',
                'sequence': seq,
                'label': -1,
                'version': 2
            })

        df = pd.DataFrame(variants)

        # Save to cache
        df.to_csv(self.variants_cache_file, index=False)

        n_path = (df['label'] == 1).sum()
        n_benign = (df['label'] == 0).sum()
        n_vus = (df['label'] == -1).sum()
        n_unique = df['sequence'].nunique()

        print(f"Generated {len(df)} variant examples ({n_unique} unique sequences)")
        print(f"  Pathogenic: {n_path}")
        for vt in df[df['label'] == 1]['variant_type'].unique():
            ct = (df['variant_type'] == vt).sum()
            print(f"    {vt}: {ct}")
        print(f"  Benign: {n_benign}")
        print(f"  VUS (excluded from training): {n_vus}")
        print(f"Saved to {self.variants_cache_file}")

        return df

    def get_training_data(
        self,
        val_split: float = 0.2,
        test_split: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get stratified train/val/test datasets with no leakage.

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        df = self.fetch_hbb_variants(force_download=True)

        # Separate labeled data from VUS
        df_labeled = df[df['label'] != -1].copy()
        df_vus = df[df['label'] == -1].copy()

        # Stratified split by variant_type to ensure each split has all types
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df_labeled,
            test_size=test_split,
            random_state=random_state,
            stratify=df_labeled['variant_type']
        )

        # Second split: train vs val (from the train+val portion)
        relative_val = val_split / (1 - test_split)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val,
            random_state=random_state,
            stratify=train_val_df['variant_type']
        )

        # Enforce zero leakage: remove any duplicate sequences across splits
        # (can occur when random background SNPs produce identical results)
        train_seqs = set(train_df['sequence'].tolist())
        val_keep = val_df[~val_df['sequence'].isin(train_seqs)]
        combined_seen = train_seqs | set(val_keep['sequence'].tolist())
        test_keep = test_df[~test_df['sequence'].isin(combined_seen)]

        if len(val_keep) < len(val_df) or len(test_keep) < len(test_df):
            removed = (len(val_df) - len(val_keep)) + (len(test_df) - len(test_keep))
            print(f"  Removed {removed} duplicate sequences across splits")

        val_df = val_keep.reset_index(drop=True)
        test_df = test_keep.reset_index(drop=True)

        print(f"\nDataset split (stratified, no leakage):")
        for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            n_p = (split_df['label'] == 1).sum()
            n_b = (split_df['label'] == 0).sum()
            print(f"  {name}: {len(split_df)} ({n_p} pathogenic, {n_b} benign)")
        print(f"  VUS (held out): {len(df_vus)}")

        return train_df, val_df, test_df

    def get_reference_sequence(self) -> str:
        """Get the HBB reference sequence."""
        return self.HBB_REFERENCE

    def get_sickle_cell_examples(self) -> pd.DataFrame:
        """Get only sickle cell disease examples."""
        df = self.fetch_hbb_variants()
        return df[df['mutation'] == 'E6V']
