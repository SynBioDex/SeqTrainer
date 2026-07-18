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
    "AttentionBackend",
    "BackendCapability",
    "BackendUnavailableError",
    "HardwareTelemetry",
    "MemoryBackend",
    "ModelGeometry",
    "ParityReport",
    "StageBBackendConfig",
    "StageBBackendRegistry",
    "StageBBenchmarkResult",
    "TensorParity",
    "TimingTelemetry",
    "benchmark_stage_b",
    "compare_backends",
    "execute_stage_b",
    "write_stage_b_artifacts",
]
