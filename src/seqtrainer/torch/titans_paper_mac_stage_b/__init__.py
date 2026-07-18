"""Stage B adapters and evidence around the immutable paper-MAC reference."""

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
from .stack import StageBMACStack, StageBStackOutput

__all__ = [
    "APPROXIMATE_WINDOWS",
    "ActivationDType",
    "AffineStateMap",
    "AttentionBackend",
    "ApproximateScanMemoryBackend",
    "BackendCapability",
    "BackendUnavailableError",
    "CausalConvolutionalUpdateGates",
    "DEFAULT_SCALES",
    "ExactAccelerationMatrix",
    "ExactAcceleratedMemoryBackend",
    "HardwareTelemetry",
    "GateBackend",
    "MemoryBackend",
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
    "affine_scan_linear_recurrence",
    "benchmark_stage_b",
    "compare_backends",
    "compose_affine",
    "execute_stage_b",
    "integrate_flash_attention",
    "integrate_sdpa_attention",
    "probe_flash_mask_support",
    "run_scan_feasibility_harness",
    "run_exact_acceleration_matrix",
    "run_convolution_comparison",
    "run_attention_backend_evidence",
    "run_approximate_scan_study",
    "sequential_linear_recurrence",
    "sdpa_allowed_attention_mask",
    "write_stage_b_artifacts",
    "write_scan_feasibility_artifact",
    "write_exact_acceleration_matrix",
    "write_convolution_comparison",
    "write_attention_backend_evidence",
    "write_approximate_scan_study",
    "update_segment_with_convolutional_gates",
    "update_segment_with_stale_windows",
]
