# Project Directory Structure & File Map

This document provides a clear overview of all files in the **BloomDNABert** project and explains their specific roles within the system.

---

## 📂 Root Directory
| File Name | Purpose |
| :--- | :--- |
| `app.py` | **Main Entry Point**: Launches the Gradio-based web dashboard. Use this for interactive training, sequence analysis, and visualization. |
| `requirements.txt` | Lists all Python dependencies required to run the project (e.g., PyTorch, Transformers, Gradio). |
| `README.md` | Comprehensive project documentation, including architecture details, installation steps, and scientific background. |
| `IMPLEMENTATION_SUMMARY.md` | Technical deep-dive into the implementation details of the BGPCA architecture and the hybrid system. |
| `QUICKSTART.md` | A brief guide to getting the system up and running as quickly as possible. |
| `STATUS.txt` | Tracks the current development status, training logs, and project milestones. |
| `LICENSE` | MIT License file defining the terms of use for the codebase. |
| `run_training.py` | **CLI Training**: Script to train the Baseline model via the command line without using the UI. |
| `test_system.py` | **Sanity Check**: Script that runs end-to-end tests for all core components (Bloom filter, DNABERT, Loader, Visualizer). |
| `run_dnabert.py` | Utility script to run or test the DNABERT-2 model independently. |
| `launch_dashboard.py` | A wrapper script specifically designed to launch `app.py` with predefined settings. |
| `run_app.sh` | Shell script for Unix-based systems to launch the application. |
| `create_triton_stub.py` | Optimization script to create Triton stubs for performance enhancements (primarily for GPU environments). |

---

## 📂 `bloom_dnabert/` (Core Logic)
This directory contains the heart of the project: the machine learning models and data processing logic.

| File Name | Purpose |
| :--- | :--- |
| `bloom_filter.py` | Implements the **Multi-scale Bloom Filter**. It handles the fast lookup of pathogenic k-mers and generates the positional signal used by the model. |
| `dnabert_wrapper.py` | A wrapper for the **DNABERT-2** transformer. It handles DNA sequence tokenization and extracts high-dimensional hidden states. |
| `bloom_attention_bridge.py` | Contains the **Novel BGPCA Architecture**. This is where the Bloom filter signals are fused with DNABERT embeddings using cross-attention. |
| `classifier.py` | Defines the **Classifier Pipelines**. It includes the `HybridClassifierPipeline` (Baseline) and the `BloomGuidedPipeline` (Advanced). |
| `data_loader.py` | Handles **Data Acquisition**. It fetches variant data from ClinVar, generates synthetic sequences, and preparing data for training/testing. |
| `visualizer.py` | **Interpretability Engine**: Generates attention heatmaps and position-importance plots to explain model decisions. |
| `__init__.py` | Package initialization file that exports key classes for easier imports. |

---

## 📂 `data/`
| File Name | Purpose |
| :--- | :--- |
| `clinvar_pan_grch38_snvs.csv` | **Large training set** (optional): tens of thousands of GRCh38 SNV windows from ClinVar `variant_summary` + `hg38.fa`. |
| `clinvar_variant_summary.txt.gz` | Optional local copy of ClinVar FTP `variant_summary.txt.gz` for the pan-genome builder. |
| `hbb_clinvar_refined.csv` | Small *HBB*-only ClinVar slice (exon 1 coding, coordinate-safe). |
| `hbb_variants.csv` | Optional cache from the ClinVar API. |

---

## 📂 `tests/`
Automated test suite to ensure system reliability.

| File Name | Purpose |
| :--- | :--- |
| `conftest.py` | Global configuration and fixtures for Pytest. |
| `test_bloom_filter.py` | Unit tests for Bloom filter accuracy and signal generation. |
| `test_bloom_attention_bridge.py` | Tests for the cross-attention layers and architecture integrity. |
| `test_classifier.py` | Validation tests for the training pipelines and prediction logic. |
| `test_data_loader.py` | Ensures data is correctly loaded, split, and sanitized. |
| `__init__.py` | Makes the directory a Python package for test discovery. |

---

## 🛠️ Key Commands Summary
- **Start Web UI**: `python app.py`
- **Run Quick Tests**: `python test_system.py`
- **Run Full Test Suite**: `pytest tests/`
- **Train CLI Baseline**: `python run_training.py`
