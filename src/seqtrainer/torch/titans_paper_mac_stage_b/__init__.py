"""Stage B adapters and evidence around the immutable paper-MAC reference."""

from .backends import (
    BackendCapability,
    BackendUnavailableError,
    StageBBackendRegistry,
    execute_stage_b,
)
from .config import (
    APPROXIMATE_WINDOWS,
    ActivationDType,
    AttentionBackend,
    MemoryBackend,
    StageBBackendConfig,
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
    "BackendCapability",
    "BackendUnavailableError",
    "DEFAULT_SCALES",
    "ExactAccelerationMatrix",
    "ExactAcceleratedMemoryBackend",
    "HardwareTelemetry",
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
    "run_scan_feasibility_harness",
    "run_exact_acceleration_matrix",
    "sequential_linear_recurrence",
    "write_stage_b_artifacts",
    "write_scan_feasibility_artifact",
    "write_exact_acceleration_matrix",
]
