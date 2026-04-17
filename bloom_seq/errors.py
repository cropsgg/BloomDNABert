"""Framework errors."""

from pathlib import Path


class DataSourceError(RuntimeError):
    """No real data source available (files, cache, or remote API)."""

    @staticmethod
    def default_message(cache_dir: Path) -> str:
        return (
            "No training variant data available. Add one of:\n"
            f"  - {cache_dir / 'clinvar_pan_grch38_snvs.csv'} (pan-gene windows; see DATASETS.md)\n"
            f"  - {cache_dir / 'hbb_clinvar_refined.csv'} (HBB-only; build with "
            "scripts/build_hbb_clinvar_dataset.py)\n"
            f"  - {cache_dir / 'hbb_variants.csv'} (API cache, version >= 4)\n"
            "Or fetch from ClinVar with network access. See DATASETS.md for sources and commands.\n"
            "For unit tests only, set BLOOM_DNABERT_ALLOW_SYNTHETIC=1 or BLOOM_SEQ_ALLOW_SYNTHETIC=1 "
            "or pass allow_synthetic=True."
        )


class PluginNotFoundError(LookupError):
    """No plugin registered under the given name."""


class PluginLoadError(RuntimeError):
    """Failed to import or construct a plugin."""


class AlphabetMismatchError(ValueError):
    """Backbone alphabet does not match pattern index or data source."""
