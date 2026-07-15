"""PyTorch integration points for SeqTrainer."""

from .adapters import to_torch_dataset
from .backbones import (
    HuggingFaceBackbone,
    evo2,
    gemma3_4b,
    hyena_dna,
    nucleotide_transformer_v2,
)
from .finetune import build_finetune_config

__all__ = [
    "to_torch_dataset",
    "build_finetune_config",
    "HuggingFaceBackbone",
    "nucleotide_transformer_v2",
    "hyena_dna",
    "evo2",
    "gemma3_4b",
]

try:  # optional torch extra
    from .titans import DNATokenizer, NeuralLongTermMemory, TitansMIRASConfig, TitansMemoryAsContextClassifier

    __all__.extend(
        [
            "TitansMIRASConfig",
            "DNATokenizer",
            "NeuralLongTermMemory",
            "TitansMemoryAsContextClassifier",
        ]
    )
except ModuleNotFoundError:
    pass

try:  # optional torch extra
    from .titans_mac import (
        DNABaseTokenizer,
        GeneratedSequence,
        TitansMACForCausalLM,
        TitansMACForSequenceClassification,
        TitansMACLMConfig,
        compute_lm_metrics,
        count_parameters,
        generate_dna,
        load_training_checkpoint,
        save_training_checkpoint,
    )

    __all__.extend(
        [
            "DNABaseTokenizer",
            "TitansMACLMConfig",
            "TitansMACForCausalLM",
            "TitansMACForSequenceClassification",
            "GeneratedSequence",
            "count_parameters",
            "generate_dna",
            "save_training_checkpoint",
            "load_training_checkpoint",
            "compute_lm_metrics",
        ]
    )
except ModuleNotFoundError:
    pass
