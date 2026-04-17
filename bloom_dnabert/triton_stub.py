"""Deprecated shim."""

from bloom_seq.plugins.dnabert2.triton_compat import (
    disable_dnabert_flash_attention,
    prepare_dnabert_environment,
)

__all__ = ["prepare_dnabert_environment", "disable_dnabert_flash_attention"]
