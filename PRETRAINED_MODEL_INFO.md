# Pretrained Model Information: DNABERT-2

This document provides technical details about the primary pretrained machine learning model used in this project.

## 🤖 Model Identity
- **Model Name**: DNABERT-2-117M
- **Hugging Face ID**: [`zhihan1996/DNABERT-2-117M`](https://huggingface.co/zhihan1996/DNABERT-2-117M)
- **Parameters**: 117 Million
- **Architecture**: Transformer-based, optimized for DNA sequence understanding.

---

## 📂 Implementation in Code
The loading and management of this model are handled centrally in the project to ensure consistency.

### Location
- **File**: [`bloom_dnabert/dnabert_wrapper.py`](file:///c:/Users/Darshil%20Agarwal/Downloads/BloomDNABert-main/BloomDNABert-main/bloom_dnabert/dnabert_wrapper.py)
- **Class**: `DNABERTWrapper`

### Core Loading Logic
The model is loaded using the `transformers` library with specific flags to enable attention extraction and custom code execution:

```python
# Implementation snippet from dnabert_wrapper.py
self.tokenizer = AutoTokenizer.from_pretrained(
    "zhihan1996/DNABERT-2-117M",
    trust_remote_code=True
)

self.model = AutoModel.from_pretrained(
    "zhihan1996/DNABERT-2-117M",
    trust_remote_code=True,
    output_attentions=True  # Enables heatmap generation
)
```

---

## ⚙️ Configuration Details
When initialized, the model is configured with the following parameters (standard for DNABERT-2-117M):
- **Hidden Size**: 768
- **Transformer Layers**: 12
- **Attention Heads**: 12
- **Device Support**: Automatically detects and uses **CUDA (GPU)** if available, otherwise falls back to **CPU**.

## 💡 Why DNABERT-2?
Unlike the original DNABERT which used k-mer based tokenization, DNABERT-2 uses **Byte Pair Encoding (BPE)**, allowing it to:
1. Handle much longer sequences efficiently.
2. Capture more complex genomic patterns.
3. Provide richer per-token embeddings for the BGPCA fusion mechanism.
