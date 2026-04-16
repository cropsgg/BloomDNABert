# Datasets for Bloom-Enhanced DNABERT (HBB / sickle-cell-style classification)

This project scores **short DNA windows** as **pathogenic vs benign** using Bloom-filter k-mers plus DNABERT-2. The demo UI still centers on *HBB*, but training can use either a **small *HBB*-only ClinVar slice** or a **large pan-gene ClinVar GRCh38 SNV table** (tens of thousands of windows). Below is a survey of public sources and how to build each tier.

---

## What the code expects

- **Input:** variable-length DNA strings over **A/C/G/T** (IUPAC ambiguity is not modeled).
- **Labels:** binary **1 = pathogenic / likely pathogenic**, **0 = benign / likely benign** (VUS and conflicting entries are dropped).
- **Reference:** an in-repo linear *HBB* sequence (`ClinVarDataLoader.HBB_REFERENCE`) aligned so that **CDS positions 1–91** match **indices 0–90** in that string (exon 1 coding). Full **CDS→genomic** mapping for later exons is *not* implemented yet; including those ClinVar rows without liftover would mis-place alleles.

---

## Recommended choice for this repo (smallest *decent* real set)

| Property | Value |
|----------|--------|
| **Source** | [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) via NCBI [E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/) |
| **Scope** | *HBB* + **single-nucleotide variant** + pathogenic/likely pathogenic/benign/likely benign |
| **Refinement** | **NM_000518.5** in title; **simple exonic** `c.\d+[ATCG]>[ATCG]`; **CDS position ≤ 91** (exon 1 coding only) so coordinates match the project reference |
| **Shipped file** | `data/hbb_clinvar_refined.csv` (regenerate with `python scripts/build_hbb_clinvar_dataset.py`) |
| **Typical size** | On the order of **tens to low hundreds** of rows (class-balanced enough for small-model fine-tuning on an M1) |
| **Why this size** | Large enough to train the **hybrid MLP head** and sanity-check BGPCA; small enough to avoid huge download/processing and long DNABERT forward passes during experimentation |

Larger public corpora exist (below). A **single-gene** corpus (*HBB* only) cannot reach hundreds of thousands of **distinct** clinically labeled SNVs—there are not that many in ClinVar—so scaling up means **either** (a) pan-gene ClinVar SNVs on a real reference genome, **or** (b) heavy label-preserving augmentation of a smaller real seed set.

---

## Large-scale training (≈10k–100k+ labeled windows)

Use this when you want **orders of magnitude more data** than the *HBB*-only CSV.

| Property | Value |
|----------|--------|
| **Source** | ClinVar [variant_summary.txt.gz](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz) (tab-delimited; updated regularly) |
| **Assembly** | **GRCh38** rows only; alleles taken from `ReferenceAlleleVCF` / `AlternateAlleleVCF` / `PositionVCF` |
| **Labels** | `ClinicalSignificance` parsed to pathogenic/likely pathogenic (**1**) vs benign/likely benign (**0**); uncertain/conflicting skipped |
| **Sequences** | ±100 bp (configurable) around each SNV on a local **UCSC-style hg38 FASTA** (`chr1`, `chr2`, …). The reference base is **verified** against the FASTA before the alt is applied. |
| **Output** | `data/clinvar_pan_grch38_snvs.csv` — **loaded first** by `ClinVarDataLoader` when present |
| **Build command** | `python scripts/build_clinvar_pan_dataset.py --reference /path/to/hg38.fa --max-output 25000` |
| **Typical scale** | Default **25k** rows; raise `--max-output` (e.g. **100000**) and `--pool-multiplier` if you want more (longer build, larger ClinVar scan). This is the practical way to approach “**~10⁴×**” more examples than the ~60-row *HBB* exon-1 table—not by finding “more *HBB* labels”, but by **training across genes**. |
| **Caveat** | The Bloom filter is still seeded with *HBB* pathogenic k-mers, so pan-gene rows will often have **weaker Bloom signal**; DNABERT carries most of the load. That is expected for a general SNV classifier head. |

**One-time reference:** download and decompress [hg38.fa.gz](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/latest/hg38.fa.gz) from UCSC (several GB). The build script can also download the ClinVar summary into `data/clinvar_variant_summary.txt.gz` on first run.

---

## Catalog of relevant public datasets & resources

### Clinically annotated variant databases (best label quality for this task)

| Resource | What it is | Access | Fit for this project |
|----------|------------|--------|----------------------|
| **[ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/)** | Expert/submitter assertions of clinical significance | [FTP](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/), [API](https://www.ncbi.nlm.nih.gov/books/NBK25501/) | **Primary.** Gene-specific VCF/TSV or API → filter *HBB* → NM_000518.5. **Used in-repo.** |
| **[ClinGen / Variant Curation Interface](https://clinicalgenome.org/)** | Expert panels, allele representations | Web, some exports via partners | Good for **curated interpretations**; usually consumed **via ClinVar** rather than a separate training dump. |
| **[LOVD (e.g. HBB locus)](https://www.lovd.nl/)** | Locus-specific variant databases | SQL/API varies by install | Can add **orthogonal** rows; cleaning and license vary by instance. |
| **[HGMD (commercial)](https://www.hgmd.cf.ac.uk/)** | Literature-reported disease mutations | Paid | Strong for pathology research; **not redistributable**; use only under license. |

### Population frequency (weak labels or priors, not pathogenicity alone)

| Resource | What it is | Access | Fit for this project |
|----------|------------|--------|----------------------|
| **[gnomAD](https://gnomad.broadinstitute.org/)** | Allele frequencies in large cohorts | [Downloads](https://gnomad.broadinstitute.org/downloads) | **Excellent** for *HBB* variant context and **rare vs common** priors; **does not** replace ClinVar labels. Often used as a **feature** (AF) layered on top of ClinVar-labeled examples. |
| **[1000 Genomes](https://www.internationalgenome.org/)** | Global SNP/indels | [FTP](https://ftp.1000genomes.ebi.ac.uk/) | Population variation; **not** clinical labels. |
| **[dbSNP](https://www.ncbi.nlm.nih.gov/snp/)** | RefSNP catalog | [FTP](https://ftp.ncbi.nlm.nih.gov/snp/) | Stable rs IDs; clinical assertions usually **via ClinVar cross-refs**. |
| **[UK Biobank / All of Us / other biobanks]** | Phenotype + WGS/WES in cohorts | Controlled access | Powerful but **governance-heavy**; overkill for this demo pipeline. |

### Variant effect / pathogenicity predictors (scores, not raw training pairs)

| Resource | What it is | Access | Fit for this project |
|----------|------------|--------|----------------------|
| **[CADD](https://cadd.gs.washington.edu/)** | Deleteriousness scores | Whole-genome tracks | Use as **extra input features** or baselines. |
| **[AlphaMissense / ESM-variants](https://github.com/google-deepmind/alphamissense)** | Model-based pathogenicity | Downloads / APIs | Useful for **comparison** or auxiliary loss; different license/scale. |
| **[Ensembl VEP / VCF annotation](https://www.ensembl.org/vep)** | Consequence annotation | Tooling | Helps **normalize** HGVS and consequences; not a labeled training set by itself. |

### Cancer-focused (generally wrong disease domain for HBB)

| Resource | Notes |
|----------|--------|
| **[COSMIC](https://cancer.sanger.ac.uk/cosmic)** | Somatic cancer mutations; **not** appropriate as germline *HBB* pathology labels without careful filtering. |

### Benchmark suites (method comparison, not HBB-only)

| Resource | Notes |
|----------|--------|
| **[VarBench](https://www.ncbi.nlm.nih.gov/clinvar/docs/varbench/)** / **[ClinVar benchmarks](https://www.ncbi.nlm.nih.gov/clinvar/docs/maintenance_use/)** | Good for **evaluating** variant classifiers across genes; can inspire **evaluation splits**, not necessarily training data for this narrow app. |
| **[GEMINI / civicpy / other integrators](https://docs.civicpy.org/en/latest/)** | Aggregate evidence; useful **ETL** layers if you expand beyond *HBB*. |

---

## Why we did *not* pick these as the default training corpus

- **gnomAD / 1000G / dbSNP alone:** frequency ≠ pathogenicity; you still need ClinVar (or similar) for **supervised** 0/1 training.
- **Full ClinVar *HBB* without coordinate hygiene:** many rows are intronic or lie in later exons; naïvely treating `c.` positions as indices into a genomic string **mis-places** the alternate allele (the bug we avoided by restricting to **exon 1 coding** until a full mapper exists).
- **Massive multi-gene VCFs (UKB, All of Us):** excellent science, but **large downloads**, access controls, and **GPU/time** costs conflict with a lightweight M1 workflow.
- **HGMD / proprietary panels:** strong content but **not** freely redistributable as a project CSV.

---

## Operations in this repository

| Action | Command / path |
|--------|------------------|
| **Large pan-gene set (loaded first if present)** | `data/clinvar_pan_grch38_snvs.csv` — `python scripts/build_clinvar_pan_dataset.py --reference /path/to/hg38.fa` |
| **Small *HBB*-only slice** | `data/hbb_clinvar_refined.csv` — `python scripts/build_hbb_clinvar_dataset.py` |
| **API-only cache** | If neither file exists, `fetch_hbb_variants()` may query NCBI and write `data/hbb_variants.csv` |
| **Synthetic fallback** | Only if ClinVar is unreachable — **offline** checks, not publication-grade training |

---

## Future improvements (if you need more real rows without synthetic data)

1. **CDS→genomic map** for NM_000518.5 across all exons (using annotated exon intervals on GRCh38) so *HBB*-specific ClinVar rows beyond exon 1 are placed correctly.
2. **Liftover** of ClinVar `canonical_spdi` or chromosomal positions to the exact reference string used in code (verify against NCBI *HBB* genomic FASTA).
3. **gnomAD AF** as extra scalar features alongside Bloom + DNABERT (still keep ClinVar labels).

---

## References (URLs)

- ClinVar: https://www.ncbi.nlm.nih.gov/clinvar/  
- ClinVar FTP primer: https://www.ncbi.nlm.nih.gov/clinvar/docs/ftp_primer/  
- NCBI E-utilities: https://www.ncbi.nlm.nih.gov/books/NBK25501/  
- RefSeq NM_000518.5: https://www.ncbi.nlm.nih.gov/nuccore/NM_000518.5  
- gnomAD: https://gnomad.broadinstitute.org/  
- dbSNP: https://www.ncbi.nlm.nih.gov/snp/  

---

*Last updated: aligned with the loader behavior in `bloom_dnabert/data_loader.py` (germline significance parsing, exon-1 coordinate guard, refined CSV path).*
