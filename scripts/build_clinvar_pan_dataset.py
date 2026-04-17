#!/usr/bin/env python3
"""
Build a large pan-gene ClinVar SNV training table (default ~25k rows).

Requires:
  - ClinVar ``variant_summary.txt.gz`` (downloaded automatically into ``data/`` if missing)
  - UCSC GRCh38 ``hg38.fa`` (and ``.fai`` from ``samtools faidx`` or pyfaidx will index on first open)

Example:
  python scripts/build_clinvar_pan_dataset.py \\
    --reference ~/refs/hg38.fa \\
    --max-output 25000 \\
    --out data/clinvar_pan_grch38_snvs.csv

Download hg38 (one-time, ~3 GiB compressed):
  curl -O https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/latest/hg38.fa.gz
  gunzip hg38.fa.gz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bloom_seq.plugins.clinvar_hbb.clinvar_pan import (  # noqa: E402
    CLINVAR_VARIANT_SUMMARY_GZ,
    download_variant_summary,
    reservoir_sample_clinvar_snvs,
    rows_to_training_records,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Build pan-gene ClinVar SNV windows (GRCh38).")
    p.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Path to GRCh38 FASTA (e.g. hg38.fa from UCSC; chrom names chr1, chr2, …).",
    )
    p.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "data" / "clinvar_variant_summary.txt.gz",
        help="Local path to variant_summary.txt.gz (download if missing).",
    )
    p.add_argument(
        "--max-output",
        type=int,
        default=25_000,
        help="Target number of rows after FASTA validation (may be slightly lower).",
    )
    p.add_argument(
        "--pool-multiplier",
        type=int,
        default=4,
        help="Reservoir pool size = min(max-pool, max_output * multiplier) before FASTA filtering.",
    )
    p.add_argument(
        "--max-pool",
        type=int,
        default=200_000,
        help="Cap on reservoir size (raise for very large --max-output, e.g. 500000).",
    )
    p.add_argument(
        "--half-window",
        type=int,
        default=100,
        help="Bases on each side of the SNV (sequence length ≈ 2*half_window + 1).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "clinvar_pan_grch38_snvs.csv",
        help="Output CSV path.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-download", action="store_true", help="Fail if summary gzip is missing.")
    args = p.parse_args()

    if not args.reference.is_file():
        print(f"Reference FASTA not found: {args.reference}", file=sys.stderr)
        sys.exit(1)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    if not args.summary.is_file():
        if args.no_download:
            print(f"Missing {args.summary} and --no-download set.", file=sys.stderr)
            sys.exit(1)
        print(f"Expected ClinVar file at {args.summary}")
        print(f"Source: {CLINVAR_VARIANT_SUMMARY_GZ}")
        download_variant_summary(args.summary)

    pool = min(
        args.max_pool,
        max(args.max_output * args.pool_multiplier, args.max_output + 5_000),
    )
    print(f"Reservoir sampling up to {pool} candidate SNV rows from {args.summary} ...")
    import pandas as pd

    rows = reservoir_sample_clinvar_snvs(args.summary, max_keep=pool, seed=args.seed)
    print(f"Extracting windows from {args.reference} (half_window={args.half_window}) ...")
    recs = rows_to_training_records(rows, args.reference, half_window=args.half_window)
    df = pd.DataFrame(recs)
    if df.empty:
        print("No rows extracted. Check hg38.fa chromosome naming (chr1, …) vs ClinVar.", file=sys.stderr)
        sys.exit(1)
    if len(df) > args.max_output:
        df = df.sample(n=args.max_output, random_state=args.seed).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
    print(df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
