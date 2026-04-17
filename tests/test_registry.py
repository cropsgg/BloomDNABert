"""Entry-point discovery (requires editable install)."""

import subprocess
import sys
from pathlib import Path

from bloom_seq.registry import list_all_plugins

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_registry_lists_reference_plugins():
    p = list_all_plugins()
    assert "dnabert2" in p.get("backbones", [])
    assert "multiscale_bloom" in p.get("pattern_indexes", [])
    assert "clinvar_hbb" in p.get("data_sources", [])
    assert "dna" in p.get("alphabets", [])
    assert "dna_trinuc" in p.get("plausibility", [])


def test_bloom_dnabert_shim_emits_single_deprecation_warning():
    """Fresh interpreter so the shim import is not cached without the warning."""
    code = r"""
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    import bloom_dnabert  # noqa: F401
    dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(dep) == 1, dep
"""
    subprocess.check_call(
        [sys.executable, "-W", "always", "-c", code],
        cwd=_REPO_ROOT,
    )
