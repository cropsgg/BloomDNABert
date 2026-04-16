# Contributing to BloomDNABert

Thank you for helping improve this project. This document describes how to set up a development environment, run checks, and open pull requests.

## Development setup

1. **Python** 3.10 or newer (3.11–3.13 are routinely used with the pinned dependency stack).
2. Create a virtual environment and install the package in editable mode with dev tools:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
   ```

   Alternatively, install runtime deps only:

   ```bash
   pip install -r requirements.txt
   pip install pytest
   ```

3. **Training data** is not silently synthesized. Before running training or `test_system.py`, prepare data per [DATASETS.md](DATASETS.md), for example:

   ```bash
   python scripts/build_hbb_clinvar_dataset.py
   ```

   For unit tests, the suite mocks heavy dependencies and does not require full DNABERT weights. Some loader tests use synthetic data only when explicitly enabled via fixtures or `BLOOM_DNABERT_ALLOW_SYNTHETIC=1`.

## Running tests

```bash
pytest tests/ -q
```

Pytest is configured to collect only the `tests/` package. **Root-level smoke scripts** (see below) are not part of `pytest` and must be run manually.

## Smoke / integration scripts (repository root)

These files are **manual** checks that load real models and optional network data. They are named with a `test_` prefix for historical reasons but are **not** pytest modules:

| Script | Purpose |
|--------|---------|
| [test_system.py](test_system.py) | End-to-end smoke: Bloom, DNABERT, data loader, feature extraction. |
| [test_train.py](test_train.py) | Short BGPCA training smoke (one epoch). |
| [test_end_to_end.py](test_end_to_end.py) | Loads pipeline and runs an untrained forward pass. |

Run with `python test_system.py` (etc.) from the repo root with dependencies and data available. Document failures with logs when opening an issue.

## Pull requests

- **Scope:** Keep changes focused on a single concern (bugfix, docs, feature).
- **Tests:** Add or update tests for behavior changes. Run `pytest tests/` locally.
- **Style:** Match surrounding code (imports, naming, typing). Avoid drive-by refactors unrelated to the PR.
- **Data:** Do not commit large reference genomes, Hugging Face caches, or generated pan-genome CSVs unless the maintainers explicitly agree. See [.gitignore](.gitignore) and [DATASETS.md](DATASETS.md).

## Code of conduct

All participants are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Security

Report sensitive issues privately as described in [SECURITY.md](SECURITY.md).
