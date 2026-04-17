"""
Pan-gene ClinVar SNV windows: stream variant_summary.txt.gz and extract ±context
from a local GRCh38 FASTA (e.g. UCSC hg38.fa).

Used to build tens of thousands of labeled sequences for training without relying
on the truncated in-repo HBB genomic slice.
"""

from __future__ import annotations

import csv
import gzip
import random
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

CLINVAR_VARIANT_SUMMARY_GZ = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)


def clinical_significance_to_label(text: str) -> Optional[int]:
    """Map ClinVar ClinicalSignificance text to 1=pathogenic, 0=benign, None=skip."""
    if not text:
        return None
    t = text.strip().lower()
    if any(x in t for x in ("conflicting", "uncertain", "not provided", "no classification")):
        return None
    has_path = "pathogenic" in t
    has_ben = "benign" in t
    if has_path and has_ben:
        return None
    if has_path:
        return 1
    if has_ben:
        return 0
    return None


def normalize_clinvar_chrom(chrom: str) -> str:
    c = chrom.strip()
    if c in ("MT", "M", "chrM", "chrMT"):
        return "chrM"
    if c.startswith("chr"):
        return c
    return f"chr{c}"


def iter_variant_summary_rows(path: Path) -> Iterator[Dict[str, str]]:
    if str(path).endswith(".gz"):
        f = gzip.open(path, "rt", encoding="utf-8", errors="replace")
    else:
        f = open(path, "rt", encoding="utf-8", errors="replace")
    with f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield row


def reservoir_sample_clinvar_snvs(
    summary_path: Path,
    max_keep: int,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """
    Uniform random sample (reservoir) of rows matching GRCh38 labeled SNV filters.
    Streams the full file once; memory O(max_keep).
    """
    rng = random.Random(seed)
    pool: List[Dict[str, str]] = []
    n_match = 0
    for row in iter_variant_summary_rows(summary_path):
        if row.get("Assembly") != "GRCh38":
            continue
        if row.get("Type") != "single nucleotide variant":
            continue
        ra = (row.get("ReferenceAlleleVCF") or "").upper()
        aa = (row.get("AlternateAlleleVCF") or "").upper()
        if len(ra) != 1 or len(aa) != 1 or ra not in "ATCG" or aa not in "ATCG":
            continue
        if clinical_significance_to_label(row.get("ClinicalSignificance") or "") is None:
            continue
        if not (row.get("Chromosome") or "").strip():
            continue
        pos_s = row.get("PositionVCF") or row.get("Start") or ""
        if not str(pos_s).isdigit():
            continue
        n_match += 1
        if len(pool) < max_keep:
            pool.append(row)
            continue
        j = rng.randint(0, n_match - 1)
        if j < max_keep:
            pool[j] = row
    return pool


def download_variant_summary(dest: Path, show_progress: bool = True) -> None:
    """Download ClinVar variant_summary.txt.gz to ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if show_progress:
        print(f"Downloading {CLINVAR_VARIANT_SUMMARY_GZ} ...")
    with urllib.request.urlopen(CLINVAR_VARIANT_SUMMARY_GZ, timeout=600) as resp:
        total = resp.headers.get("Content-Length")
        chunk = 1024 * 1024
        read = 0
        with open(dest, "wb") as out:
            while True:
                b = resp.read(chunk)
                if not b:
                    break
                out.write(b)
                read += len(b)
                if show_progress and total and read % (10 * chunk) < chunk:
                    print(f"  ... {read // (1024 * 1024)} MiB", flush=True)
    if show_progress:
        print(f"Saved {dest}")


def rows_to_training_records(
    rows: List[Dict[str, str]],
    fasta_path: Path,
    half_window: int = 100,
) -> List[Dict[str, Any]]:
    """Extract mutated reference windows; drops rows where FASTA ref != ClinVar ref."""
    try:
        from pyfaidx import Fasta
    except ImportError as e:
        raise ImportError("Install pyfaidx: pip install pyfaidx") from e

    fa = Fasta(str(fasta_path), as_raw=True)
    records: List[Dict[str, Any]] = []
    for row in rows:
        chrom = normalize_clinvar_chrom(row["Chromosome"])
        pos = int(row["PositionVCF"] or row["Start"])
        ref = (row["ReferenceAlleleVCF"] or "").upper()
        alt = (row["AlternateAlleleVCF"] or "").upper()
        sig = row.get("ClinicalSignificance") or ""
        label = clinical_significance_to_label(sig)
        if label is None:
            continue
        if chrom not in fa:
            continue
        chrom_len = len(fa[chrom])
        idx0 = pos - 1
        if idx0 < 0 or idx0 >= chrom_len:
            continue
        obs = str(fa[chrom][idx0 : idx0 + 1]).upper()
        if obs != ref:
            continue
        start = max(0, idx0 - half_window)
        end = min(chrom_len, idx0 + half_window + 1)
        ref_window = str(fa[chrom][start:end]).upper()
        local = idx0 - start
        if local < 0 or local >= len(ref_window) or ref_window[local] != ref:
            continue
        mut = list(ref_window)
        mut[local] = alt
        seq = "".join(mut)
        if len(seq) < 80:
            continue
        # Header column is ``#AlleleID`` in ClinVar's TSV.
        allele = row.get("#AlleleID") or row.get("AlleleID") or "na"
        vid = row.get("VariationID") or "na"
        gene = row.get("GeneSymbol") or ""
        records.append(
            {
                "variant_id": f"ClinVarAllele_{allele}_Var_{vid}",
                "gene": gene,
                "mutation": (row.get("Name") or "")[:200],
                "hgvs_c": "",
                "hgvs_p": "",
                "position": idx0,
                "ref": ref,
                "alt": alt,
                "disease": (row.get("PhenotypeList") or "")[:200],
                "clinical_significance": sig[:200],
                "variant_type": "ClinVar_pan_pathogenic" if label == 1 else "ClinVar_pan_benign",
                "sequence": seq,
                "label": label,
                "version": 5,
            }
        )
    return records
