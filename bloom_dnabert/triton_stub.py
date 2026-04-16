"""
DNABERT-2 / PyTorch Triton cleanup (CPU and macOS).

**Do not** ship a fake ``triton`` package: ``torch`` treats ``import triton`` as a real
install and then imports ``triton.backends``, which breaks with a minimal shim.

Instead we:

1. Remove any previously installed **shim** ``site-packages/triton`` tree.
2. Patch HuggingFace cached ``bert_layers.py`` so it never imports ``flash_attn_triton``
   (the only DNABERT code path that needs Triton).
3. After load, force ``flash_attn_qkvpacked_func = None`` on CPU so only PyTorch attention runs.

On Linux with a **real** Triton wheel, no shim exists and Flash may remain enabled on CUDA.
"""

from __future__ import annotations

import os
import re
import shutil
import site
import sys
from pathlib import Path
from typing import Iterable

_SHIM_MARKERS = (
    "0.0.0-stub",
    "Minimal Triton stub",
    "CPU/macOS-safe Triton stubs",
    "bloom_dnabert.triton_stub",
)

_FLASH_TRY_RE = re.compile(
    r"try\s*:\s*\n"
    r"\s*from \.flash_attn_triton import flash_attn_qkvpacked_func\s*\n"
    r"\s*except\s+ImportError(?:\s+as\s+\w+)?\s*:\s*\n"
    r"\s*flash_attn_qkvpacked_func\s*=\s*None\s*",
    re.MULTILINE,
)

_REPLACEMENT = (
    "# BloomDNABert: PyTorch attention only — do not import flash_attn_triton / Triton.\n"
    "flash_attn_qkvpacked_func = None\n"
)


def _site_roots() -> Iterable[Path]:
    roots: list[Path] = []
    try:
        roots.append(Path(site.getusersitepackages()))
    except Exception:
        pass
    try:
        for p in site.getsitepackages():
            roots.append(Path(p))
    except Exception:
        pass
    hf = os.environ.get("HF_HOME")
    if hf:
        roots.append(Path(hf))
    roots.append(Path.home() / ".cache" / "huggingface")
    roots.append(Path(__file__).resolve().parents[1])
    seen: set[Path] = set()
    for r in roots:
        if r and r.is_dir() and r not in seen:
            seen.add(r)
            yield r


def remove_shim_triton_packages() -> None:
    """Delete ``site-packages/triton`` if it is our (or another) minimal shim, not a real wheel."""
    for root in _site_roots():
        tdir = root / "triton"
        init = tdir / "__init__.py"
        if not init.is_file():
            continue
        try:
            head = init.read_text(encoding="utf-8", errors="replace")[:4000]
        except OSError:
            continue
        if not any(m in head for m in _SHIM_MARKERS):
            continue
        try:
            shutil.rmtree(tdir)
        except OSError:
            pass
    for key in list(sys.modules):
        if key == "triton" or key.startswith("triton."):
            del sys.modules[key]


def patch_dnabert_bert_layers_no_flash() -> None:
    """
    Rewrite cached DNABERT ``bert_layers.py`` files so the Flash/Triton module is never imported.

    Safe to run repeatedly; skips files already patched.
    """
    seen: set[Path] = set()
    for root in _site_roots():
        if not root.is_dir():
            continue
        try:
            candidates = list(root.rglob("bert_layers.py"))
        except OSError:
            continue
        for path in candidates:
            if path in seen:
                continue
            key = str(path).lower()
            if "zhihan1996" not in key and "dnabert" not in key:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if "BloomDNABert: PyTorch attention only" in text:
                seen.add(path)
                continue
            if "flash_attn_triton" not in text:
                seen.add(path)
                continue
            new_text, n = _FLASH_TRY_RE.subn(_REPLACEMENT, text, count=1)
            if n:
                path.write_text(new_text, encoding="utf-8")
            seen.add(path)


def disable_dnabert_flash_attention() -> None:
    """Force PyTorch attention on CPU even if an old process left Flash enabled."""
    import torch

    if torch.cuda.is_available():
        return
    for _name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if not _name.endswith("bert_layers"):
            continue
        if hasattr(mod, "flash_attn_qkvpacked_func"):
            setattr(mod, "flash_attn_qkvpacked_func", None)


def prepare_dnabert_environment() -> None:
    """Call before ``AutoModel.from_pretrained`` for zhihan1996/DNABERT-2-117M."""
    remove_shim_triton_packages()
    patch_dnabert_bert_layers_no_flash()
