"""Stage B adapters and evidence around the immutable paper-MAC reference."""

from .a100_pilot import (
    A100PilotUnavailableError,
    evaluate_a100_evidence,
    inspect_a100,
    run_a100_pilot,
    verify_a100_pilot_directory,
)

from .backends import (
    BackendCapability,
    BackendUnavailableError,
    StageBBackendRegistry,
    execute_stage_b,
)
from .attention import (
    integrate_flash_attention,
    integrate_sdpa_attention,
    probe_flash_mask_support,
    sdpa_allowed_attention_mask,
)
from .attention_benchmark import (
    run_attention_backend_evidence,
    write_attention_backend_evidence,
)
from .audit import (
    build_stage_b_audit,
    render_fidelity_performance_audit,
    write_stage_b_audit,
)
from .approximate_scan import (
    ApproximateScanMemoryBackend,
    update_segment_with_stale_windows,
)
from .approximate_scan_benchmark import (
    run_approximate_scan_study,
    write_approximate_scan_study,
)
from .config import (
    APPROXIMATE_WINDOWS,
    ActivationDType,
    AttentionBackend,
    GateBackend,
    MemoryBackend,
    StageBBackendConfig,
)
from .convolution import (
    CausalConvolutionalUpdateGates,
    update_segment_with_convolutional_gates,
)
from .convolution_benchmark import (
    run_convolution_comparison,
    write_convolution_comparison,
)
from .exact_acceleration import ExactAcceleratedMemoryBackend
from .exact_acceleration_benchmark import (
    DEFAULT_SCALES,
    ExactAccelerationMatrix,
    StageBScale,
    run_exact_acceleration_matrix,
    write_exact_acceleration_matrix,
)
from .parity import ParityReport, TensorParity, compare_backends
from .scan_feasibility import (
    AffineStateMap,
    ScanFeasibilityResult,
    affine_scan_linear_recurrence,
    compose_affine,
    run_scan_feasibility_harness,
    sequential_linear_recurrence,
    write_scan_feasibility_artifact,
)
from .telemetry import (
    HardwareTelemetry,
    ModelGeometry,
    StageBBenchmarkResult,
    TimingTelemetry,
    benchmark_stage_b,
    write_stage_b_artifacts,
)
from .training_step_benchmark import (
    A100_TRAINING_STEP_SCALE,
    TRAINING_VARIANTS,
    TrainingStepScale,
    run_training_step_matrix,
    write_training_step_matrix,
)
from .stack import StageBMACStack, StageBStackOutput
from .long_context_benchmark import (
    DEFAULT_LONG_CONTEXT_SCALES,
    DEFAULT_LONG_CONTEXT_VARIANTS,
    LongContextScale,
    run_long_context_study,
    write_long_context_study,
)

__all__ = [
    "APPROXIMATE_WINDOWS",
    "A100PilotUnavailableError",
    "A100_TRAINING_STEP_SCALE",
    "ActivationDType",
    "AffineStateMap",
    "AttentionBackend",
    "ApproximateScanMemoryBackend",
    "BackendCapability",
    "BackendUnavailableError",
    "CausalConvolutionalUpdateGates",
    "DEFAULT_SCALES",
    "DEFAULT_LONG_CONTEXT_SCALES",
    "DEFAULT_LONG_CONTEXT_VARIANTS",
    "ExactAccelerationMatrix",
    "ExactAcceleratedMemoryBackend",
    "HardwareTelemetry",
    "GateBackend",
    "MemoryBackend",
    "LongContextScale",
    "ModelGeometry",
    "ParityReport",
    "ScanFeasibilityResult",
    "StageBBackendConfig",
    "StageBBackendRegistry",
    "StageBBenchmarkResult",
    "StageBMACStack",
    "StageBScale",
    "StageBStackOutput",
    "TensorParity",
    "TimingTelemetry",
    "TRAINING_VARIANTS",
    "TrainingStepScale",
    "affine_scan_linear_recurrence",
    "benchmark_stage_b",
    "build_stage_b_audit",
    "compare_backends",
    "compose_affine",
    "execute_stage_b",
    "evaluate_a100_evidence",
    "integrate_flash_attention",
    "integrate_sdpa_attention",
    "inspect_a100",
    "probe_flash_mask_support",
    "render_fidelity_performance_audit",
    "run_scan_feasibility_harness",
    "run_training_step_matrix",
    "run_exact_acceleration_matrix",
    "run_long_context_study",
    "run_convolution_comparison",
    "run_attention_backend_evidence",
    "run_approximate_scan_study",
    "run_a100_pilot",
    "sequential_linear_recurrence",
    "sdpa_allowed_attention_mask",
    "write_stage_b_artifacts",
    "write_training_step_matrix",
    "write_stage_b_audit",
    "write_scan_feasibility_artifact",
    "write_exact_acceleration_matrix",
    "write_long_context_study",
    "write_convolution_comparison",
    "write_attention_backend_evidence",
    "write_approximate_scan_study",
    "update_segment_with_convolutional_gates",
    "update_segment_with_stale_windows",
    "verify_a100_pilot_directory",
]
