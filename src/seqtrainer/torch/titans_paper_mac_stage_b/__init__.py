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

__all__ = [
    "APPROXIMATE_WINDOWS",
    "ActivationDType",
    "AffineStateMap",
    "AttentionBackend",
    "BackendCapability",
    "BackendUnavailableError",
    "HardwareTelemetry",
    "MemoryBackend",
    "ModelGeometry",
    "ParityReport",
    "ScanFeasibilityResult",
    "StageBBackendConfig",
    "StageBBackendRegistry",
    "StageBBenchmarkResult",
    "TensorParity",
    "TimingTelemetry",
    "affine_scan_linear_recurrence",
    "benchmark_stage_b",
    "compare_backends",
    "compose_affine",
    "execute_stage_b",
    "run_scan_feasibility_harness",
    "sequential_linear_recurrence",
    "write_stage_b_artifacts",
    "write_scan_feasibility_artifact",
]
