"""Stage C genomic language-model interfaces for the paper-MAC stack."""

from .config import MemoryMode, StageCModelConfig
from .baselines import BaselineResult, MarkovDNABaseline, run_statistical_baselines, uniform_baseline
from .metrics import StageCMetrics, compute_stage_c_metrics
from .model import BlockStates, StageCLMOutput, StageCPaperMACForCausalLM, detach_stream_states
from .trainer import StageCTrainer, StreamBatchScheduler, TrainingStepRecord
from .checkpoints import (
    CHECKPOINT_FORMAT_VERSION,
    load_stage_c_checkpoint,
    save_stage_c_checkpoint,
)
from .reporting import write_cpu_pilot_report, write_tokenizer_report, write_training_history
from .evaluation import EvaluationResult, evaluate_ordered_streams
from .tokenizers import (
    EncodedDNA,
    Evo2CharTokenizer,
    GenomeTokenizer,
    HuggingFaceBPETokenizer,
    SeqTrainerBaseTokenizer,
    SixMerTokenizer,
    TokenizerMetrics,
    TokenizerSpec,
    TrainOnlyBPETokenizer,
    evaluate_tokenizer,
    normalize_dna,
)
from .study import StudyProtocol, amend, canonical_json, initialize, record, report, validate_protocol, verify

__all__ = [
    "EncodedDNA",
    "EvaluationResult",
    "BaselineResult",
    "BlockStates",
    "Evo2CharTokenizer",
    "GenomeTokenizer",
    "HuggingFaceBPETokenizer",
    "MemoryMode",
    "MarkovDNABaseline",
    "SeqTrainerBaseTokenizer",
    "SixMerTokenizer",
    "StageCModelConfig",
    "StageCTrainer",
    "StageCLMOutput",
    "StageCMetrics",
    "StageCPaperMACForCausalLM",
    "TokenizerMetrics",
    "TokenizerSpec",
    "TrainingStepRecord",
    "TrainOnlyBPETokenizer",
    "evaluate_tokenizer",
    "evaluate_ordered_streams",
    "compute_stage_c_metrics",
    "detach_stream_states",
    "CHECKPOINT_FORMAT_VERSION",
    "load_stage_c_checkpoint",
    "normalize_dna",
    "run_statistical_baselines",
    "save_stage_c_checkpoint",
    "StreamBatchScheduler",
    "uniform_baseline",
    "write_cpu_pilot_report",
    "write_tokenizer_report",
    "write_training_history",
    "StudyProtocol",
    "canonical_json",
    "validate_protocol",
    "initialize",
    "record",
    "amend",
    "verify",
    "report",
]
