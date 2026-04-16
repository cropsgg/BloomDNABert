"""
Statistical plausibility of a DNA string vs human reference-style sequence.

This does **not** check whether a window exists at specific GRCh38 coordinates.
It scores whether overlapping trinucleotide usage and per-base composition resemble
a random stretch of human genomic DNA (background from an hg38 sample).

Use the returned ``probability_statistically_spurious`` as a sanity check for pasted
or invented sequences: high values mean the string is unlikely to have been drawn
from a typical human genomic context *under this model* — not that a variant is
biologically impossible.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any


@lru_cache(maxsize=1)
def _human_trinuc_background() -> tuple:
    path = Path(__file__).resolve().parent / "human_trinuc_bg.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path.name}; reinstall package or regenerate from hg38."
        )
    with path.open() as f:
        payload = json.load(f)
    probs: Dict[str, float] = payload["prob"]
    logp = {k: math.log(probs[k]) for k in probs}
    ll_iid_trinuc = sum(logp[k] for k in logp) / 64.0
    ll_typical_human = sum(probs[k] * logp[k] for k in probs)
    return probs, logp, ll_iid_trinuc, ll_typical_human


def assess_sequence_genomic_plausibility(sequence: str) -> Dict[str, Any]:
    """
    Score how much a sequence resembles human genomic DNA (k-mer + diversity).

    Returns:
        genomic_plausibility_score: float in [0, 1] — higher = more typical.
        probability_statistically_spurious: float in [0, 1] — higher = less typical
            (informal "chance this is nonsense / not genomic-like" under this model).
        mean_trinucleotide_log_prob: average log P(trinuc | human reference sample).
        base_diversity_score: normalized Shannon entropy of A,C,G,T (0..1).
        trinucleotides_scored: count of overlapping 3-mers used (no N in k-mer).
        reliability_note: str — empty if enough k-mers were scored.
    """
    _, logp, ll_iid, ll_human = _human_trinuc_background()

    s = sequence.upper().strip()
    acc = 0.0
    n_tri = 0
    for i in range(len(s) - 2):
        mer = s[i : i + 3]
        if "N" in mer:
            continue
        if any(b not in "ACGT" for b in mer):
            continue
        acc += logp[mer]
        n_tri += 1

    c = Counter(ch for ch in s if ch in "ACGT")
    tot_b = sum(c.values())
    if tot_b > 0:
        h = -sum((cnt / tot_b) * math.log(cnt / tot_b) for cnt in c.values())
        diversity = h / math.log(4.0)
    else:
        diversity = 0.0

    note = ""
    if n_tri < 3:
        note = "Too few informative trinucleotides (length or excess N); score is unreliable."
        plaus = 0.0
    else:
        mean_lp = acc / n_tri
        denom = ll_human - ll_iid
        if denom <= 1e-9:
            raise RuntimeError("Invalid trinucleotide background span.")
        tri_fit = (mean_lp - ll_iid) / denom
        tri_fit = max(0.0, min(1.0, tri_fit))
        plaus = tri_fit * max(0.0, min(1.0, diversity))

    spurious = max(0.0, min(1.0, 1.0 - plaus))

    return {
        "genomic_plausibility_score": round(plaus, 6),
        "probability_statistically_spurious": round(spurious, 6),
        "mean_trinucleotide_log_prob": round(acc / n_tri, 6) if n_tri else None,
        "base_diversity_score": round(diversity, 6),
        "trinucleotides_scored": n_tri,
        "reliability_note": note,
    }
