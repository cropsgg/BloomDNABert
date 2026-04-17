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
| [pyproject.toml](pyproject.toml) | Package metadata (`bloom-seq`), optional extras `[dl]`, `[ui]`, `[dev]`, `[all]`, and `bloom_seq.*` entry points. |
| [README.md](README.md) | Main documentation. |
| [QUICKSTART.md](QUICKSTART.md) | Short setup path. |
| [DATASETS.md](DATASETS.md) | Data sources, file formats, build commands. |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to develop, test, and open PRs. |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), [SECURITY.md](SECURITY.md) | Community and security policy. |
| [LICENSE](LICENSE) | MIT license. |

## `bloom_seq/`

**Core framework:** [protocols.py](bloom_seq/protocols.py) (typed plugin contracts), [registry.py](bloom_seq/registry.py) (entry-point discovery), [errors.py](bloom_seq/errors.py), [alphabets.py](bloom_seq/alphabets.py), [splits.py](bloom_seq/splits.py), [bridge.py](bloom_seq/bridge.py) (BGPCA layers), [pipeline.py](bloom_seq/pipeline.py) (baseline + BGPCA pipelines), [viz.py](bloom_seq/viz.py).

**Bundled reference plugins** under [bloom_seq/plugins/](bloom_seq/plugins/):

| Plugin | Role |
|--------|------|
| [dnabert2/](bloom_seq/plugins/dnabert2/) | DNABERT-2 backbone + optional Triton compat. |
| [multiscale_bloom/](bloom_seq/plugins/multiscale_bloom/) | Multi-scale Bloom pattern index (alphabet-aware). |
| [clinvar_hbb/](bloom_seq/plugins/clinvar_hbb/) | HBB/ClinVar data source, HBB k-mer seeds, pan-gene builder helpers. |
| [plausibility_dna_trinuc/](bloom_seq/plugins/plausibility_dna_trinuc/) | DNA trinucleotide plausibility prior + JSON background. |

See [PLUGINS.md](PLUGINS.md) and [MODELS.md](MODELS.md).

## `bloom_dnabert/` (deprecated)

Thin **shim** that re-exports the public API from `bloom_seq` and emits a `DeprecationWarning`. Prefer `import bloom_seq` for new code.

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

Pytest suite (`pytest` collects `tests/` recursively). Core tests: `test_protocols.py`, `test_registry.py`, `test_bridge.py`, `test_pipeline.py`. Plugin-focused tests live under [`tests/plugins/`](tests/plugins/) (multiscale Bloom, ClinVar/HBB, plausibility, pan-gene helpers). Mocks heavy DL dependencies where appropriate.

## `.github/`

Issue templates and `workflows/ci.yml`: a **minimal** job (core install, no `[dl]`/`[ui]`) and a **full** job with extras and the complete test suite.

## Common commands

```bash
pip install -e ".[dl,ui,dev]"
pytest
python app.py
python scripts/build_hbb_clinvar_dataset.py   # refresh HBB CSV if needed
```
