from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parent.parent


class GeneConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str
    refseq_transcript: str
    ncbi_gene_id: str = ""
    chromosome: str = ""
    grch38_start: int = 0
    grch38_end: int = 0
    example_mutation_name: str = ""


class ReferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fasta_path: Path
    sha256: Optional[str] = None


class AnnotationsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cds_end_exclusive_0: int = Field(ge=0)
    reject_ambiguous_bases_in_reference: bool = True


class ClinVarConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    esearch_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    esummary_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    search_term: str
    retmax: int = 500
    esummary_batch_size: int = 50
    request_delay_sec: float = 0.35
    timeout_sec: int = 15
    sequence_context_size: int = 100
    sequence_context_jitter: int = 10
    background_snp_rate: float = 0.002
    require_refseq_in_title: bool = True
    hgvs_linear_overrides: Dict[str, int] = Field(default_factory=dict)


class PathogenicTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    variant_id_prefix: str
    variant_type: str
    label: int
    mutation: str
    hgvs_c: str
    hgvs_p: str
    disease: str
    clinical_significance: str
    position: int
    ref: str
    alt: str
    protected_positions: List[int]
    count: int
    context_size: int
    context_jitter: int
    background_snp_rate: float
    require_ref_base: Optional[str] = None


class BenignSynonymousConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    first_codon_index: int = 1
    last_codon_index: int = 29
    samples_per_position: int = 8
    context_size: int = 100
    context_jitter: int = 20
    background_snp_rate: float = 0.003


class BenignIntronicConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    count: int = 30
    region_start_offset: int = 20
    region_end_offset: int = 100
    context_size: int = 100
    context_jitter: int = 20
    background_snp_rate: float = 0.003
    hgvs_intron_anchor: str = "93+"


class VUSConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    count: int = 30
    context_size: int = 100
    context_jitter: int = 15
    background_snp_rate: float = 0.003
    exclude_positions: List[int]


class SyntheticDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pathogenic: List[PathogenicTemplate]
    benign_synonymous: BenignSynonymousConfig
    benign_intronic: BenignIntronicConfig
    vus: VUSConfig


class BloomConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    k_sizes: List[int] = Field(min_length=1)
    capacity: int = 100_000
    error_rate: float = 0.001
    seeds_path: Path


class DNABERTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str = "zhihan1996/DNABERT-2-117M"
    tokenizer_max_length: int = 512


class BGPCAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    d_bloom: int = 64
    n_cross_attn_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.2


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_tokens: int = 128
    batch_size: int = 16
    encode_batch_size: int = 8
    learning_rate_baseline: float = 1e-3
    learning_rate_bgpca: float = 5e-4
    weight_decay: float = 0.01
    patience: int = 10
    max_grad_norm: float = 1.0
    num_workers: int = 0
    pin_memory: bool = True
    use_amp: bool = True
    baseline_epochs_cli: int = 30
    gradio_epochs_default: int = 30
    gradio_epochs_max: int = 100


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cache_dir: Path = Path("data")
    random_seed: int = 42
    val_split: float = Field(default=0.2, ge=0.0, lt=1.0)
    test_split: float = Field(default=0.2, ge=0.0, lt=1.0)
    cache_format_version: int = 4
    variants_cache_template: str = "{gene_symbol}_variants.csv"
    codon_table_path: Optional[Path] = None
    feature_cache_dir: Optional[Path] = None


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_sequence_length: int = 5000
    mc_dropout_samples: int = 20


class GradioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    server_name: str = "0.0.0.0"
    server_port: Optional[int] = None
    port_scan_start: int = 7860
    port_scan_end: int = 7870


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gene: GeneConfig
    reference: ReferenceConfig
    annotations: AnnotationsConfig
    clinvar: ClinVarConfig
    synthetic: SyntheticDataConfig
    bloom: BloomConfig
    dnabert: DNABERTConfig
    bgpca: BGPCAConfig
    training: TrainingConfig
    data: DataConfig
    inference: InferenceConfig
    gradio: GradioConfig

    @model_validator(mode="after")
    def _resolve_paths(self) -> AppSettings:
        def R(p: Path) -> Path:
            if p.is_absolute():
                return p
            return (ROOT / p).resolve()

        self.reference.fasta_path = R(self.reference.fasta_path)
        self.bloom.seeds_path = R(self.bloom.seeds_path)
        self.data.cache_dir = R(self.data.cache_dir)
        if self.data.codon_table_path is not None:
            self.data.codon_table_path = R(self.data.codon_table_path)
        if self.data.feature_cache_dir is not None and str(self.data.feature_cache_dir).strip():
            self.data.feature_cache_dir = R(self.data.feature_cache_dir)
        else:
            self.data.feature_cache_dir = None
        return self


def load_settings(config_path: Path | None = None) -> AppSettings:
    if config_path is None:
        env = os.environ.get("BLOOM_CONFIG")
        config_path = Path(env) if env else ROOT / "config" / "default.yaml"
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"bloom_dnabert: config not found at {config_path}. "
            "Create config/default.yaml or set BLOOM_CONFIG."
        )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid YAML root in {config_path}")
    return AppSettings.model_validate(raw)


def settings_fingerprint_digest(settings: AppSettings) -> str:
    payload = settings.model_dump_json(
        exclude={"gradio": True},
        exclude_none=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]
