"""Deprecated shim."""

from bloom_seq.errors import DataSourceError
from bloom_seq.plugins.clinvar_hbb.source import (
    SYNTHETIC_ENV_VARS,
    ClinVarDataLoader,
)

__all__ = ["ClinVarDataLoader", "DataSourceError", "SYNTHETIC_ENV_VARS"]
