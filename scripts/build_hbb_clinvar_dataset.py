#!/usr/bin/env python3
"""
Refresh ``data/hbb_clinvar_refined.csv`` from NCBI ClinVar (HBB exonic SNVs).

Requires project dependencies (``pip install -r requirements.txt``).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "bloom_data_loader",
    ROOT / "bloom_dnabert" / "data_loader.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
ClinVarDataLoader = _mod.ClinVarDataLoader


def main() -> None:
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    loader = ClinVarDataLoader(cache_dir=str(data_dir))
    df = loader._fetch_clinvar_variants()
    if df is None or df.empty:
        print("ClinVar fetch failed or returned no rows.", file=sys.stderr)
        sys.exit(1)
    df = loader._dedupe_by_hgvs(df)
    df["version"] = 4
    out = data_dir / loader.REFINED_CLINVAR_FILE
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")
    print(df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
