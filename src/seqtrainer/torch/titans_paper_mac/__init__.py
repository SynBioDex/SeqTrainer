"""Paper-traceable functional neural-memory primitives for Titans MAC.

This package is intentionally separate from :mod:`seqtrainer.torch.titans_mac`.
The latter is a slot-retrieval/EMA baseline; these classes implement only the
functional long-term-memory update needed for the Stage A paper reference.
"""

from .lifecycle import LifecycleTransition, StreamLifecycleHarness
from .synthetic import (
    DEFAULT_VOCABULARY,
    SEGMENT_LENGTH,
    SyntheticScore,
    SyntheticSegment,
    SyntheticTaskFixture,
    SyntheticVocabulary,
    build_stage_a_fixtures,
    context_boundary_reset,
    delayed_key_value_recall,
    overwrite_forgetting,
    score_query_predictions,
)

_MEMORY_EXPORTS = {
    "AdaptiveUpdateGates",
    "FunctionalNeuralMemory",
    "GateValues",
    "ParameterGateValues",
    "PaperResidualMemory",
    "PerLayerChannelUpdateGates",
    "PaperMACStreamState",
}

_MAC_EXPORTS = {
    "PaperMACBlock",
    "PaperMACBlockOutput",
    "block_causal_attention_mask",
}

_BENCHMARK_EXPORTS = {
    "AcceptanceGates",
    "BenchmarkConfig",
    "BenchmarkResults",
    "VariantMetrics",
    "run_stage_a_benchmark",
    "write_benchmark_artifacts",
}

__all__ = [
    "AdaptiveUpdateGates",
    "AcceptanceGates",
    "BenchmarkConfig",
    "BenchmarkResults",
    "DEFAULT_VOCABULARY",
    "FunctionalNeuralMemory",
    "GateValues",
    "ParameterGateValues",
    "PaperResidualMemory",
    "PerLayerChannelUpdateGates",
    "LifecycleTransition",
    "PaperMACStreamState",
    "PaperMACBlock",
    "PaperMACBlockOutput",
    "SEGMENT_LENGTH",
    "StreamLifecycleHarness",
    "SyntheticScore",
    "SyntheticSegment",
    "SyntheticTaskFixture",
    "SyntheticVocabulary",
    "VariantMetrics",
    "build_stage_a_fixtures",
    "block_causal_attention_mask",
    "context_boundary_reset",
    "delayed_key_value_recall",
    "overwrite_forgetting",
    "score_query_predictions",
    "run_stage_a_benchmark",
    "write_benchmark_artifacts",
]


def __getattr__(name: str) -> object:
    """Load Torch-dependent memory primitives only when a caller requests them."""

    if name in _BENCHMARK_EXPORTS:
        from .benchmark import (
            AcceptanceGates,
            BenchmarkConfig,
            BenchmarkResults,
            VariantMetrics,
            run_stage_a_benchmark,
            write_benchmark_artifacts,
        )

        exports = {
            "AcceptanceGates": AcceptanceGates,
            "BenchmarkConfig": BenchmarkConfig,
            "BenchmarkResults": BenchmarkResults,
            "VariantMetrics": VariantMetrics,
            "run_stage_a_benchmark": run_stage_a_benchmark,
            "write_benchmark_artifacts": write_benchmark_artifacts,
        }
        globals().update(exports)
        return exports[name]
    if name in _MAC_EXPORTS:
        from .mac import PaperMACBlock, PaperMACBlockOutput, block_causal_attention_mask

        exports = {
            "PaperMACBlock": PaperMACBlock,
            "PaperMACBlockOutput": PaperMACBlockOutput,
            "block_causal_attention_mask": block_causal_attention_mask,
        }
        globals().update(exports)
        return exports[name]
    if name not in _MEMORY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .memory import (
        AdaptiveUpdateGates,
        FunctionalNeuralMemory,
        GateValues,
        ParameterGateValues,
        PaperResidualMemory,
        PerLayerChannelUpdateGates,
    )
    from .state import PaperMACStreamState

    exports = {
        "AdaptiveUpdateGates": AdaptiveUpdateGates,
        "FunctionalNeuralMemory": FunctionalNeuralMemory,
        "GateValues": GateValues,
        "ParameterGateValues": ParameterGateValues,
        "PaperResidualMemory": PaperResidualMemory,
        "PerLayerChannelUpdateGates": PerLayerChannelUpdateGates,
        "PaperMACStreamState": PaperMACStreamState,
    }
    globals().update(exports)
    return exports[name]
