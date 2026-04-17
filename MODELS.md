# Model catalog (documentation only)

This table lists **representative** foundation models that can back a `bloom_seq` plugin. It is not exhaustive; check each model’s license and Hugging Face / paper for exact hidden sizes and context limits.

Columns:

- **BGPCA fit**: whether Bloom-guided cross-attention (per-token hidden states + positional pattern signal) is typically meaningful.
- **Tokenizer**: broad tokenizer family for alignment planning.

## DNA

| Model | Example HF id | Hidden | Tokenizer | Context (order of magnitude) | BGPCA fit |
|-------|----------------|--------|-----------|------------------------------|-----------|
| DNABERT-2 | `zhihan1996/DNABERT-2-117M` | 768 | BPE | ~512–2k tokens | Yes (reference) |
| Nucleotide Transformer v2 | `InstaDeepAI/nucleotide_transformer_v2_*` | varies | BPE | multi-kb | Yes |
| HyenaDNA | `LongSafari/hyenadna-*` | varies | character / hyena | very long | Partial (attention differs) |
| Caduceus / Caduceus-Ph | `kulesh-group/caduceus-*` | varies | DNA tokens | long | Partial |
| GENA-LM | `AIRI-Institute/gena-lm-*` | varies | BPE | long | Yes |
| Evo / Evo2 | `togethercomputer/evo-*`, `arcinstitute/evo2-*` | varies | char / model-specific | very long | Partial |
| GROVER, HybriDNA, ModernGENA | checkpoints / papers | varies | varies | long | Partial |

## RNA

| Model | Example source | Notes | BGPCA fit |
|-------|----------------|-------|-----------|
| RNA-FM | `multimolecule/rnafm` | ncRNA-focused FM | Yes (with RNA alphabet + k-mers) |
| mRNA-FM | `multimolecule/mrnafm` | CDS / mRNA | Yes |
| mRNABERT | Nature Communications 2025 | dual tokenization + protein contrast | Partial |

## Protein

| Model | Example HF id | Notes | BGPCA fit |
|-------|----------------|-------|-----------|
| ESM2 | `facebook/esm2_t*` | general PLM | Yes (residue-level) |
| ESM3 | `EvolutionaryScale/esm3-*` | multimodal | Partial |
| ProtT5 | `Rostlab/prot_t5_*` | encoder–decoder | Partial (use encoder) |
| ProteinBERT / DistilProtBert | various | smaller | Yes |
| Tranception | `facebook/rce*`, variant checkpoints | VEP-focused | Yes |
| Ankh, PoET, SaProt, METL, ProtMamba | various | specialized | Case-by-case |

## Out of scope as *sequence* backbones

Structure predictors whose primary output is 3D coordinates (AlphaFold, ESMFold) are not drop-in `Backbone` implementations; use a **structure token** plugin or separate visualization tool.

## Licensing

Verify **model weights**, **data**, and **code** licenses before redistributing a plugin or bundling weights.
