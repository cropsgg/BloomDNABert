# Project directory map

Short reference for where things live in **BloomDNABert**. The canonical manual is [README.md](README.md); dataset provenance is [DATASETS.md](DATASETS.md).

## Root

| File | Role |
|------|------|
| [app.py](app.py) | Gradio dashboard: training and sequence analysis. |
| [launch_dashboard.py](launch_dashboard.py) | Launcher that imports `app.main` (UTF-8 console tweaks on Windows). |
| [create_triton_stub.py](create_triton_stub.py) | Optional environment prep for DNABERT-2 without a real Triton install. |
| [run_training.py](run_training.py) | CLI entry to train the baseline pipeline without the UI. |
| [test_system.py](test_system.py) | Manual full-stack smoke script (not pytest). |
| [test_train.py](test_train.py), [test_end_to_end.py](test_end_to_end.py) | Manual smoke scripts (not pytest). |
| [requirements.txt](requirements.txt) | Pinned runtime dependencies. |
| [pyproject.toml](pyproject.toml) | Package metadata and optional `[dev]` extras (`pytest`). |
| [README.md](README.md) | Main documentation. |
| [QUICKSTART.md](QUICKSTART.md) | Short setup path. |
| [DATASETS.md](DATASETS.md) | Data sources, file formats, build commands. |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to develop, test, and open PRs. |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), [SECURITY.md](SECURITY.md) | Community and security policy. |
| [LICENSE](LICENSE) | MIT license. |

## `bloom_dnabert/`

Core library: Bloom filters, DNABERT wrapper, BGPCA bridge, classifiers, ClinVar loader (`DataSourceError` when no real data), visualizer, sequence plausibility, ClinVar pan helpers, Triton compatibility stub.

## `scripts/`

| Script | Role |
|--------|------|
| [build_hbb_clinvar_dataset.py](scripts/build_hbb_clinvar_dataset.py) | Refresh `data/hbb_clinvar_refined.csv` from ClinVar. |
| [build_clinvar_pan_dataset.py](scripts/build_clinvar_pan_dataset.py) | Build optional pan-gene SNV CSV + reference windows. |

## `data/`

| File | Role |
|------|------|
| `hbb_clinvar_refined.csv` | Small curated *HBB* slice (tracked). |
| `hbb_variants.csv` | API cache if you fetch live ClinVar (gitignored). |
| `clinvar_pan_grch38_snvs.csv`, `refs/hg38.fa`, etc. | Optional large assets (see `.gitignore` and DATASETS.md). |

## `tests/`

Pytest suite (`pytest` collects only `tests/`). Mocks heavy DL dependencies where appropriate.

## `.github/`

Issue templates and `workflows/ci.yml` (pytest on Python 3.11 and 3.12).

## Common commands

```bash
pip install -e ".[dev]"
pytest
python app.py
python scripts/build_hbb_clinvar_dataset.py   # refresh HBB CSV if needed
```
