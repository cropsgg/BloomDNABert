# bloom_seq plugins

Third-party models and data sources integrate by declaring **Python entry points** in `pyproject.toml`. At runtime, `bloom_seq.registry` loads them with `importlib.metadata`.

## Entry-point groups

| Group | TOML section | Typical implementation |
|-------|----------------|-------------------------|
| Alphabets | `[project.entry-points."bloom_seq.alphabets"]` | `bloom_seq.alphabets.Alphabet` instance (e.g. DNA, RNA, protein) |
| Backbones | `[project.entry-points."bloom_seq.backbones"]` | Class wrapping a HuggingFace `AutoModel` / custom encoder |
| Pattern indexes | `[project.entry-points."bloom_seq.pattern_indexes"]` | Class with `get_feature_vector`, `get_positional_signal`, `get_token_aligned_signal` |
| Data sources | `[project.entry-points."bloom_seq.data_sources"]` | Class exposing `get_training_splits(...)` → `(train_df, val_df, test_df)` |
| Plausibility | `[project.entry-points."bloom_seq.plausibility"]` | Optional `score(sequence) -> dict` prior |

## Minimal HuggingFace backbone template

```python
# mypkg/esm2_backbone.py
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from bloom_seq.alphabets import PROTEIN_ALPHABET

class ESM2Backbone:
    name = "esm2_650m"
    alphabet = PROTEIN_ALPHABET
    supports_attention = False
    supports_hidden_states = True

    def __init__(self, model_name: str = "facebook/esm2_t33_650M_UR50D", device: str | None = None):
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        cfg = self.model.config
        self.hidden_size = cfg.hidden_size
        self.num_layers = cfg.num_hidden_layers
        self.num_heads = cfg.num_attention_heads
        self.max_length = getattr(cfg, "max_position_embeddings", 1024)

    def get_embedding(self, sequence: str, pool_method: str = "mean") -> np.ndarray:
        # Implement tokenization + pooling; return (hidden_size,) float32 vector
        ...

    def get_attention_weights(self, sequence: str, layer: int = -1) -> np.ndarray:
        raise NotImplementedError("Optional for models without attentions")

    def get_token_level_outputs(self, sequence: str) -> dict:
        # Must return token hidden states and char/residue spans for BGPCA alignment
        ...
```

Register:

```toml
[project.entry-points."bloom_seq.backbones"]
esm2_650m = "mypkg.esm2_backbone:ESM2Backbone"
```

## Token alignment rules

BGPCA needs, for each tokenizer token, a span `(start, end)` in **raw sequence coordinates** (0-based, end exclusive).

- **Character-level** DNA/RNA models: one token per base; spans are `(i, i+1)`.
- **BPE** (e.g. DNABERT-2): use `return_offsets_mapping=True` when available; fall back to heuristic mapping (see `bloom_seq.plugins.dnabert2.wrapper`).
- **k-mer** tokenizers: span length equals k (or model-specific).
- **Structure-augmented** (e.g. SaProt): spans may be non-contiguous; provide best-effort bounding boxes.

## Pattern index contract

A pattern index should expose:

- `alphabet`: same `Alphabet` as the backbone (or a compatible subset).
- `get_feature_vector(seq) -> np.ndarray` (fixed length, used in baseline MLP and gated fusion).
- `get_positional_signal(seq) -> (seq_len, n_scales)` for BGPCA.
- `get_token_aligned_signal(seq, token_spans)` aggregating nucleotide/residue hits to tokens.

The reference implementation is `bloom_seq.plugins.multiscale_bloom.MultiScaleBloomFilter` (multi k-mer Bloom filters).

## Opting out of BGPCA

If `supports_attention` is false or hidden states are unavailable, use only `HybridClassifierPipeline` (concat + MLP) or supply a backbone that still exposes pooled embeddings plus a pattern summary vector.

## Discovery API

```python
from bloom_seq.registry import backbones, list_all_plugins
print(list_all_plugins())
Cls = backbones.get("dnabert2")
model = Cls()
```

## Reference plugins in this repo

| Name | Module |
|------|--------|
| `dnabert2` | [bloom_seq/plugins/dnabert2/](bloom_seq/plugins/dnabert2/) |
| `multiscale_bloom` | [bloom_seq/plugins/multiscale_bloom/](bloom_seq/plugins/multiscale_bloom/) |
| `clinvar_hbb` | [bloom_seq/plugins/clinvar_hbb/](bloom_seq/plugins/clinvar_hbb/) |
| `dna_trinuc` | [bloom_seq/plugins/plausibility_dna_trinuc/](bloom_seq/plugins/plausibility_dna_trinuc/) |
