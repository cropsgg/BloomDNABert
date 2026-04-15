from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from .codon import load_codon_table
from .reference import load_fasta_sequence, verify_file_sha256
from .settings import AppSettings, load_settings

CODON_TABLE = load_codon_table()


class ClinVarDataLoader:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.cache_dir = Path(settings.data.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tpl = settings.data.variants_cache_template
        self.variants_cache_file = self.cache_dir / tpl.format(
            gene_symbol=settings.gene.symbol.lower()
        )
        if settings.reference.sha256:
            verify_file_sha256(
                settings.reference.fasta_path,
                settings.reference.sha256,
            )
        _, self._reference = load_fasta_sequence(settings.reference.fasta_path)
        if (
            settings.annotations.reject_ambiguous_bases_in_reference
            and "N" in self._reference.upper()
        ):
            raise ValueError(
                "Reference FASTA must not contain N when "
                "reject_ambiguous_bases_in_reference is true."
            )
        if self.cds_end_exclusive > len(self._reference):
            raise ValueError(
                "annotations.cds_end_exclusive_0 exceeds reference length."
            )
        for tmpl in settings.synthetic.pathogenic:
            if tmpl.position < 0 or tmpl.position >= len(self._reference):
                raise ValueError(
                    f"synthetic.pathogenic {tmpl.variant_id_prefix}: "
                    f"position {tmpl.position} out of reference range."
                )
            if tmpl.require_ref_base is not None:
                if self._reference[tmpl.position] != tmpl.require_ref_base:
                    raise ValueError(
                        f"synthetic.pathogenic {tmpl.variant_id_prefix}: "
                        f"reference base at {tmpl.position} is not "
                        f"{tmpl.require_ref_base}."
                    )
            elif tmpl.ref and len(tmpl.ref) == 1:
                if self._reference[tmpl.position] != tmpl.ref:
                    raise ValueError(
                        f"synthetic.pathogenic {tmpl.variant_id_prefix}: "
                        f"ref {tmpl.ref!r} does not match reference at position."
                    )
        self._codon_path = settings.data.codon_table_path
        self._codon = (
            load_codon_table(self._codon_path)
            if self._codon_path
            else CODON_TABLE
        )
        self.rng = np.random.RandomState(settings.data.random_seed)

    @classmethod
    def from_config_path(cls, path: Optional[Path] = None) -> ClinVarDataLoader:
        return cls(load_settings(path))

    @property
    def reference_sequence(self) -> str:
        return self._reference

    @property
    def cds_end_exclusive(self) -> int:
        return self.settings.annotations.cds_end_exclusive_0

    @property
    def HBB_REFERENCE(self) -> str:
        return self._reference

    @property
    def EXON1_END(self) -> int:
        return self.cds_end_exclusive

    def fetch_variants(self, force_download: bool = False) -> pd.DataFrame:
        need_version = self.settings.data.cache_format_version
        if self.variants_cache_file.exists() and not force_download:
            print(f"Loading cached variants from {self.variants_cache_file}")
            df = pd.read_csv(self.variants_cache_file)
            if "version" in df.columns and df["version"].iloc[0] >= need_version:
                return df
            print("Stale cache detected. Regenerating...")

        clinvar_df = self._fetch_clinvar_variants()
        synthetic_df = self._generate_synthetic_dataset()

        if clinvar_df is not None and len(clinvar_df) > 0:
            combined = pd.concat([clinvar_df, synthetic_df], ignore_index=True)
            combined["version"] = need_version
            combined.to_csv(self.variants_cache_file, index=False)
            print(
                f"\nCombined dataset: {len(clinvar_df)} ClinVar + "
                f"{len(synthetic_df)} synthetic = {len(combined)}"
            )
            return combined

        return synthetic_df

    def fetch_hbb_variants(self, force_download: bool = False) -> pd.DataFrame:
        return self.fetch_variants(force_download=force_download)

    def _fetch_clinvar_variants(self) -> Optional[pd.DataFrame]:
        cv = self.settings.clinvar
        print(f"Fetching ClinVar variants ({self.settings.gene.symbol})...")
        try:
            search_params = {
                "db": "clinvar",
                "term": cv.search_term,
                "retmax": cv.retmax,
                "retmode": "json",
            }
            resp = requests.get(cv.esearch_url, params=search_params, timeout=cv.timeout_sec)
            resp.raise_for_status()
            search_data = resp.json()
            id_list = search_data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                print("  No ClinVar results found.")
                return None
            print(f"  Found {len(id_list)} ClinVar variant IDs")
            variants = []
            bs = cv.esummary_batch_size
            for batch_start in range(0, len(id_list), bs):
                batch_ids = id_list[batch_start : batch_start + bs]
                summary_params = {
                    "db": "clinvar",
                    "id": ",".join(batch_ids),
                    "retmode": "json",
                }
                resp = requests.get(cv.esummary_url, params=summary_params, timeout=cv.timeout_sec)
                resp.raise_for_status()
                summary_data = resp.json()
                for uid in batch_ids:
                    doc = summary_data.get("result", {}).get(uid)
                    if doc is None:
                        continue
                    variant = self._parse_clinvar_record(doc)
                    if variant is not None:
                        variants.append(variant)
                time.sleep(cv.request_delay_sec)
            if not variants:
                print("  No usable variants parsed from ClinVar.")
                return None
            df = pd.DataFrame(variants)
            n_path = (df["label"] == 1).sum()
            n_ben = (df["label"] == 0).sum()
            print(f"  Parsed {len(df)} ClinVar variants ({n_path} pathogenic, {n_ben} benign)")
            return df
        except (requests.RequestException, KeyError, ValueError) as e:
            print(
                f"  ClinVar API unavailable ({type(e).__name__}: {e}). "
                "Using synthetic data only."
            )
            return None

    @staticmethod
    def _normalize_hgvs_c(hgvs: str) -> str:
        return re.sub(r"\s+", "", (hgvs or "").upper())

    def _resolve_clinvar_snv(
        self, hgvs_c: str
    ) -> Optional[Tuple[int, str, str]]:
        norm = self._normalize_hgvs_c(hgvs_c)
        if not norm:
            return None
        cv = self.settings.clinvar
        for key, pos in cv.hgvs_linear_overrides.items():
            if self._normalize_hgvs_c(key) == norm:
                m = re.search(r"([ATCG])>([ATCG])$", norm)
                if not m:
                    return None
                return pos, m.group(1), m.group(2)
        m = re.match(r"^C\.(\d+)([ATCG])>([ATCG])$", norm)
        if m:
            return int(m.group(1)) - 1, m.group(2), m.group(3)
        m = re.match(r"^C\.(\d+)\-(\d+)([ATCG])>([ATCG])$", norm)
        if m:
            linear = self.cds_end_exclusive + int(m.group(2)) - 1
            return linear, m.group(3), m.group(4)
        m = re.match(r"^C\.(\d+)\+(\d+)([ATCG])>([ATCG])$", norm)
        if m:
            linear = int(m.group(1)) - 1 + int(m.group(2))
            return linear, m.group(3), m.group(4)
        return None

    def _parse_clinvar_record(self, doc: dict) -> Optional[dict]:
        try:
            title = doc.get("title", "")
            cv = self.settings.clinvar
            acc = self.settings.gene.refseq_transcript.split(".")[0].upper()
            if cv.require_refseq_in_title and acc not in title.replace(" ", "").upper():
                return None
            clinical_sig = doc.get("clinical_significance", {})
            if isinstance(clinical_sig, dict):
                sig_desc = clinical_sig.get("description", "").lower()
            else:
                sig_desc = str(clinical_sig).lower()
            skip_terms = ["conflicting", "uncertain", "not provided", "other"]
            if any(term in sig_desc for term in skip_terms):
                return None
            if "pathogenic" in sig_desc and "benign" not in sig_desc:
                label = 1
                variant_type = "ClinVar_pathogenic"
            elif "benign" in sig_desc and "pathogenic" not in sig_desc:
                label = 0
                variant_type = "ClinVar_benign"
            else:
                return None
            hgvs_c = ""
            variation_set = doc.get("variation_set", [])
            if isinstance(variation_set, list):
                for vs in variation_set:
                    cdna_change = vs.get("cdna_change", "")
                    if cdna_change:
                        hgvs_c = cdna_change
                        break
            resolved = self._resolve_clinvar_snv(hgvs_c)
            if resolved is None:
                return None
            cds_pos_0indexed, ref_base, alt_base = resolved
            if cds_pos_0indexed < 0 or cds_pos_0indexed >= len(self._reference):
                return None
            if self._reference[cds_pos_0indexed] != ref_base:
                return None
            seq = self._create_variant_sequence(
                position=cds_pos_0indexed,
                alt_base=alt_base,
                context_size=cv.sequence_context_size,
                context_jitter=cv.sequence_context_jitter,
                background_snp_rate=cv.background_snp_rate,
                protected_positions=[cds_pos_0indexed],
            )
            if seq is None:
                return None
            return {
                "variant_id": f'ClinVar_{doc.get("uid", "unknown")}',
                "gene": self.settings.gene.symbol,
                "mutation": title[:60] if title else "unknown",
                "hgvs_c": hgvs_c,
                "hgvs_p": doc.get("protein_change", ""),
                "position": cds_pos_0indexed,
                "ref": ref_base,
                "alt": alt_base,
                "disease": title[:80] if title else "",
                "clinical_significance": sig_desc,
                "variant_type": variant_type,
                "sequence": seq,
                "label": label,
                "version": self.settings.data.cache_format_version,
            }
        except Exception:
            return None

    def _get_codon_at_cds_position(self, cds_pos: int) -> Tuple[str, int, int]:
        codon_index = cds_pos // 3
        codon_start = codon_index * 3
        pos_in_codon = cds_pos - codon_start
        codon = self._reference[codon_start : codon_start + 3]
        return codon, codon_start, pos_in_codon

    def _is_synonymous(self, cds_pos: int, alt_base: str) -> bool:
        if cds_pos >= self.cds_end_exclusive:
            return True
        codon, codon_start, pos_in_codon = self._get_codon_at_cds_position(cds_pos)
        if len(codon) < 3:
            return False
        ref_aa = self._codon.get(codon, "?")
        mut_codon = list(codon)
        mut_codon[pos_in_codon] = alt_base
        mut_codon_str = "".join(mut_codon)
        mut_aa = self._codon.get(mut_codon_str, "?")
        return ref_aa == mut_aa and ref_aa != "?" and ref_aa != "*"

    def _get_synonymous_change(self, cds_pos: int) -> Optional[str]:
        if cds_pos >= self.cds_end_exclusive or cds_pos >= len(self._reference):
            return None
        ref_base = self._reference[cds_pos]
        candidates = [b for b in "ATCG" if b != ref_base and self._is_synonymous(cds_pos, b)]
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
        protected_positions: Optional[List[int]] = None,
    ) -> Optional[str]:
        ref_seq = self._reference
        if position >= len(ref_seq):
            return ref_seq[: 2 * context_size]
        ref_base = ref_seq[position]
        if alt_base == ref_base:
            return None
        jitter = 0
        if context_jitter > 0:
            jitter = self.rng.randint(-context_jitter, context_jitter + 1)
        start = max(0, position - context_size + jitter)
        end = min(len(ref_seq), position + context_size + 1 + jitter)
        variant_seq = list(ref_seq[start:end])
        relative_pos = position - start
        if 0 <= relative_pos < len(variant_seq):
            variant_seq[relative_pos] = alt_base
        if background_snp_rate > 0 and len(variant_seq) > 0:
            n_snps = max(0, self.rng.poisson(background_snp_rate * len(variant_seq)))
            if protected_positions is None:
                protected_positions = []
            protected_relative = {p - start for p in protected_positions if start <= p < end}
            protected_relative.add(relative_pos)
            available = [i for i in range(len(variant_seq)) if i not in protected_relative]
            if available and n_snps > 0:
                snp_positions = self.rng.choice(
                    available, size=min(n_snps, len(available)), replace=False
                )
                for sp in snp_positions:
                    orig = variant_seq[sp]
                    alts = [b for b in "ATCG" if b != orig]
                    variant_seq[sp] = self.rng.choice(alts)
        return "".join(variant_seq)

    def _generate_synthetic_dataset(self) -> pd.DataFrame:
        print(f"Generating synthetic dataset ({self.settings.gene.symbol})...")
        variants: List[dict] = []
        syn = self.settings.synthetic
        ver = self.settings.data.cache_format_version
        gene = self.settings.gene.symbol
        cds_end = self.cds_end_exclusive

        for tmpl in syn.pathogenic:
            ref_display = tmpl.ref
            if not ref_display and 0 <= tmpl.position < len(self._reference):
                ref_display = self._reference[tmpl.position]
            if tmpl.require_ref_base is not None:
                if tmpl.position >= len(self._reference):
                    continue
                if self._reference[tmpl.position] != tmpl.require_ref_base:
                    continue
            for i in range(tmpl.count):
                seq = self._create_variant_sequence(
                    position=tmpl.position,
                    alt_base=tmpl.alt,
                    context_size=tmpl.context_size,
                    context_jitter=tmpl.context_jitter,
                    background_snp_rate=tmpl.background_snp_rate,
                    protected_positions=list(tmpl.protected_positions),
                )
                if seq is None:
                    continue
                variants.append(
                    {
                        "variant_id": f"{tmpl.variant_id_prefix}_{i}",
                        "gene": gene,
                        "mutation": tmpl.mutation,
                        "hgvs_c": tmpl.hgvs_c,
                        "hgvs_p": tmpl.hgvs_p,
                        "position": tmpl.position,
                        "ref": ref_display or (self._reference[tmpl.position] if tmpl.position < len(self._reference) else ""),
                        "alt": tmpl.alt,
                        "disease": tmpl.disease,
                        "clinical_significance": tmpl.clinical_significance,
                        "variant_type": tmpl.variant_type,
                        "sequence": seq,
                        "label": tmpl.label,
                        "version": ver,
                    }
                )

        bs = syn.benign_synonymous
        for codon_idx in range(bs.first_codon_index, bs.last_codon_index + 1):
            third_pos = codon_idx * 3 + 2
            if third_pos < self.cds_end_exclusive:
                alt = self._get_synonymous_change(third_pos)
                if alt is not None:
                    for j in range(bs.samples_per_position):
                        seq = self._create_variant_sequence(
                            position=third_pos,
                            alt_base=alt,
                            context_size=bs.context_size,
                            context_jitter=bs.context_jitter,
                            background_snp_rate=bs.background_snp_rate,
                            protected_positions=[third_pos],
                        )
                        if seq is None:
                            continue
                        variants.append(
                            {
                                "variant_id": f"Benign_syn_{third_pos}_{j}",
                                "gene": gene,
                                "mutation": f"c.{third_pos+1}synonymous",
                                "hgvs_c": f"c.{third_pos+1}{self._reference[third_pos]}>{alt}",
                                "hgvs_p": "p.=",
                                "position": third_pos,
                                "ref": self._reference[third_pos],
                                "alt": alt,
                                "disease": "None",
                                "clinical_significance": "Benign",
                                "variant_type": "Benign_synonymous",
                                "sequence": seq,
                                "label": 0,
                                "version": ver,
                            }
                        )

        bi = syn.benign_intronic
        intronic_safe = list(
            range(
                cds_end + bi.region_start_offset,
                min(cds_end + bi.region_end_offset, len(self._reference)),
            )
        )
        anchor = bi.hgvs_intron_anchor
        for j in range(bi.count):
            if not intronic_safe:
                break
            pos = self.rng.choice(intronic_safe)
            ref_base = self._reference[pos]
            alt = self.rng.choice([b for b in "ATCG" if b != ref_base])
            seq = self._create_variant_sequence(
                position=pos,
                alt_base=alt,
                context_size=bi.context_size,
                context_jitter=bi.context_jitter,
                background_snp_rate=bi.background_snp_rate,
                protected_positions=[pos],
            )
            if seq is None:
                continue
            offset = pos - cds_end + 1
            variants.append(
                {
                    "variant_id": f"Benign_intronic_{j}",
                    "gene": gene,
                    "mutation": f"intronic_{pos}",
                    "hgvs_c": f"c.{anchor}{offset}{ref_base}>{alt}",
                    "hgvs_p": "p.?",
                    "position": pos,
                    "ref": ref_base,
                    "alt": alt,
                    "disease": "None",
                    "clinical_significance": "Benign",
                    "variant_type": "Benign_intronic",
                    "sequence": seq,
                    "label": 0,
                    "version": ver,
                }
            )

        vu = syn.vus
        excl = set(vu.exclude_positions)
        coding_positions = [
            p for p in range(3, self.cds_end_exclusive - 2) if p not in excl
        ]
        for i in range(vu.count):
            if not coding_positions:
                break
            pos = self.rng.choice(coding_positions)
            ref_base = self._reference[pos]
            non_syn = [b for b in "ATCG" if b != ref_base and not self._is_synonymous(pos, b)]
            if not non_syn:
                continue
            alt = self.rng.choice(non_syn)
            seq = self._create_variant_sequence(
                position=pos,
                alt_base=alt,
                context_size=vu.context_size,
                context_jitter=vu.context_jitter,
                background_snp_rate=vu.background_snp_rate,
                protected_positions=[pos],
            )
            if seq is None:
                continue
            variants.append(
                {
                    "variant_id": f"VUS_{i}",
                    "gene": gene,
                    "mutation": f"missense_{pos}",
                    "hgvs_c": f"c.{pos+1}{ref_base}>{alt}",
                    "hgvs_p": "p.?",
                    "position": pos,
                    "ref": ref_base,
                    "alt": alt,
                    "disease": "Unknown",
                    "clinical_significance": "Uncertain",
                    "variant_type": "VUS",
                    "sequence": seq,
                    "label": -1,
                    "version": ver,
                }
            )

        df = pd.DataFrame(variants)
        df.to_csv(self.variants_cache_file, index=False)
        n_path = (df["label"] == 1).sum()
        n_benign = (df["label"] == 0).sum()
        n_vus = (df["label"] == -1).sum()
        print(
            f"Generated {len(df)} examples ({df['sequence'].nunique()} unique sequences)\n"
            f"  Pathogenic: {n_path}\n  Benign: {n_benign}\n  VUS: {n_vus}\n"
            f"Saved to {self.variants_cache_file}"
        )
        return df

    def get_training_data(
        self,
        val_split: Optional[float] = None,
        test_split: Optional[float] = None,
        random_state: Optional[int] = None,
        use_cache: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        val_split = self.settings.data.val_split if val_split is None else val_split
        test_split = self.settings.data.test_split if test_split is None else test_split
        rs = self.settings.data.random_seed if random_state is None else random_state
        df = self.fetch_variants(force_download=not use_cache)
        df_labeled = df[df["label"] != -1].copy()
        df_vus = df[df["label"] == -1].copy()
        train_val_df, test_df = train_test_split(
            df_labeled,
            test_size=test_split,
            random_state=rs,
            stratify=df_labeled["variant_type"],
        )
        relative_val = val_split / (1 - test_split)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val,
            random_state=rs,
            stratify=train_val_df["variant_type"],
        )
        train_seqs = set(train_df["sequence"].tolist())
        val_keep = val_df[~val_df["sequence"].isin(train_seqs)]
        combined_seen = train_seqs | set(val_keep["sequence"].tolist())
        test_keep = test_df[~test_df["sequence"].isin(combined_seen)]
        if len(val_keep) < len(val_df) or len(test_keep) < len(test_df):
            removed = (len(val_df) - len(val_keep)) + (len(test_df) - len(test_keep))
            print(f"  Removed {removed} duplicate sequences across splits")
        val_df = val_keep.reset_index(drop=True)
        test_df = test_keep.reset_index(drop=True)
        print("\nDataset split (stratified, no leakage):")
        for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            n_p = (split_df["label"] == 1).sum()
            n_b = (split_df["label"] == 0).sum()
            print(f"  {name}: {len(split_df)} ({n_p} pathogenic, {n_b} benign)")
        print(f"  VUS (held out): {len(df_vus)}")
        return train_df, val_df, test_df

    def get_reference_sequence(self) -> str:
        return self._reference

    def get_sickle_cell_examples(self) -> pd.DataFrame:
        df = self.fetch_variants()
        name = self.settings.gene.example_mutation_name
        if name:
            return df[df["mutation"] == name]
        return df.iloc[0:0]
