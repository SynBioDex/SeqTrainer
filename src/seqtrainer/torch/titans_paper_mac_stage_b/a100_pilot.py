"""One-command A100 evidence bundle and strict offline verifier for Stage B."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Mapping, Sequence

import torch

from .attention_benchmark import (
    run_attention_backend_evidence,
    write_attention_backend_evidence,
)
from .exact_acceleration_benchmark import (
    StageBScale,
    run_exact_acceleration_matrix,
    write_exact_acceleration_matrix,
)
from .long_context_benchmark import (
    DEFAULT_LONG_CONTEXT_VARIANTS,
    LongContextScale,
    run_long_context_study,
    write_long_context_study,
)
from .training_step_benchmark import (
    A100_TRAINING_STEP_SCALE,
    run_training_step_matrix,
    write_training_step_matrix,
)


A100_EXACT_SCALE = StageBScale(
    "a100_pilot", block_count=8, d_model=384, num_heads=8
)
A100_LONG_CONTEXT_SCALE = LongContextScale(
    "a100_pilot",
    block_count=8,
    d_model=384,
    num_heads=8,
    persistent_tokens=4,
    memory_depth=1,
    segment_count=4,
    requires_a100=True,
)
EVIDENCE_FILES = {
    "preflight": "a100_preflight.json",
    "exact": "b3_exact_acceleration_matrix.json",
    "attention": "b6_attention_backend_evidence.json",
    "long_context": "b7_long_context_study.json",
    "training": "a100_training_step_matrix.json",
}


class A100PilotUnavailableError(RuntimeError):
    """Raised before benchmarking when the selected device is not an A100."""


def inspect_a100(device: torch.device | str = "cuda") -> dict[str, object]:
    """Return an explicit, serializable A100 preflight report."""

    selected = torch.device(device)
    cuda_available = torch.cuda.is_available()
    report: dict[str, object] = {
        "format_version": 1,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "requested_device": str(selected),
        "cuda_available": cuda_available,
        "is_a100": False,
    }
    if selected.type != "cuda":
        report["reason"] = "the A100 pilot requires a CUDA device"
        return report
    if not cuda_available:
        report["reason"] = "CUDA is unavailable to this PyTorch installation"
        return report
    properties = torch.cuda.get_device_properties(selected)
    report.update(
        {
            "device_name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "is_a100": "A100" in properties.name.upper(),
        }
    )
    report["reason"] = (
        "named A100 attached"
        if report["is_a100"]
        else "device name does not identify an NVIDIA A100"
    )
    return report


def _strict_json(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _variant(result: Mapping[str, object], name: str) -> Mapping[str, object]:
    variants = result["variants"]
    assert isinstance(variants, list)
    return next(item for item in variants if item["variant"] == name)


def _check(
    checks: list[dict[str, object]], criterion: str, passed: bool, evidence: str
) -> None:
    checks.append({"criterion": criterion, "passed": bool(passed), "evidence": evidence})


def evaluate_a100_evidence(
    preflight: Mapping[str, object],
    exact: Mapping[str, object],
    attention: Mapping[str, object],
    long_context: Mapping[str, object],
    training: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate already-loaded evidence without trusting a prior manifest."""

    checks: list[dict[str, object]] = []
    device_name = str(preflight.get("device_name", ""))
    _check(
        checks,
        "Named A100 preflight",
        bool(preflight.get("is_a100")) and "A100" in device_name.upper(),
        device_name or str(preflight.get("reason", "missing device")),
    )

    exact_results = exact.get("results", [])
    exact_scale = next(
        (item for item in exact_results if item["scale"]["name"] == "a100_pilot"),
        None,
    )
    exact_ok = bool(
        exact_scale
        and exact_scale["reference"]["available"]
        and exact_scale["exact_accelerated"]["available"]
        and exact_scale["tensor_exact"]
        and int(exact_scale["repetitions"]) >= 3
        and len(exact_scale["reference"]["wall_times_seconds"]) >= 3
        and len(exact_scale["exact_accelerated"]["wall_times_seconds"]) >= 3
        and "A100" in str(exact_scale["device_name"]).upper()
    )
    _check(checks, "Repeated tensor-exact B3 run", exact_ok, str(exact_scale))

    parity = attention.get("parity", {})
    fp64 = parity.get("fp64_oracle", {})
    fp32 = parity.get("fp32_numerical", {})
    parity_keys = (
        "sequence",
        "retrieval",
        "fast_weights",
        "surprise",
        "input_gradient",
        "persistent_token_gradient",
        "attention_gradient",
    )
    fp64_ok = all(
        bool(fp64.get(key, {}).get("tensor_exact"))
        or float(fp64.get(key, {}).get("maximum_absolute_error", float("inf"))) <= 1e-10
        for key in parity_keys
    )
    fp32_ok = all(
        float(fp32.get(key, {}).get("maximum_absolute_error", float("inf"))) <= 1e-4
        for key in parity_keys
    )
    mask = attention.get("mask", {})
    causality = attention.get("causality", {})
    _check(
        checks,
        "Attention parity, exact mask, and causality",
        fp64_ok
        and fp32_ok
        and int(mask.get("boolean_complement_mismatches", -1)) == 0
        and bool(causality.get("passed")),
        f"fp64={fp64_ok}; fp32={fp32_ok}; mask={mask}; causality={causality}",
    )
    mixed = attention.get("mixed_precision_behavioral", {})
    mixed_ok = all(
        bool(mixed.get(dtype, {}).get("available"))
        and mixed.get(dtype, {}).get("memory_state_dtypes") == ["float32"]
        and mixed.get(dtype, {}).get("published_sequence_dtype") == "float32"
        for dtype in ("bfloat16", "float16")
    )
    _check(checks, "BF16/FP16 preserve the FP32 memory island", mixed_ok, str(mixed))
    flash = attention.get("flash_probe", {})
    flash_attempted = "A100" in str(flash.get("device", "")).upper() and (
        bool(flash.get("available"))
        or "forced Flash SDP rejected the exact mask" in str(flash.get("reason", ""))
    )
    _check(checks, "Forced exact-mask Flash probe attempted", flash_attempted, str(flash))

    hardware = long_context.get("hardware", {})
    long_scale = next(
        (
            item
            for item in long_context.get("long_stream_scales", [])
            if item["scale"]["name"] == "a100_pilot"
        ),
        None,
    )
    long_variants = [] if long_scale is None else long_scale.get("variants", [])
    long_ok = bool(
        hardware.get("a100_available")
        and long_scale
        and long_scale.get("available")
        and long_variants
        and all(
            item.get("available")
            and item.get("first_nonfinite_context_tokens") is None
            and item.get("cuda_peak_allocated_bytes") is not None
            for item in long_variants
        )
        and long_context.get("causality", {}).get("passed")
    )
    _check(checks, "A100 long-stream matrix is finite and causal", long_ok, str(hardware))
    recall = long_context.get("controlled_long_recall", {})
    adaptive_512 = float(
        recall.get("reference", {})
        .get("delay_accuracy_by_context_tokens", {})
        .get("512", -1.0)
    )
    controls_512 = max(
        float(
            recall.get(name, {})
            .get("delay_accuracy_by_context_tokens", {})
            .get("512", -1.0)
        )
        for name in ("frozen_memory", "no_memory")
    )
    _check(
        checks,
        "Adaptive memory beats controls at 512 tokens",
        adaptive_512 > controls_512,
        f"adaptive={adaptive_512}; controls={controls_512}",
    )

    required_training = ("reference_fp32", "exact_fp32", "exact_sdpa_fp32")
    measured = {name: _variant(training, name) for name in required_training}
    training_ok = all(
        item.get("available")
        and int(item.get("repetitions", 0)) >= 3
        and len(item.get("samples_seconds", [])) >= 3
        and item.get("all_gradients_finite")
        and item.get("output_and_state_finite")
        and item.get("memory_state_dtypes") == ["float32"]
        and int(item.get("cuda_peak_allocated_bytes") or 0) > 0
        for item in measured.values()
    ) and "A100" in str(training.get("hardware", {}).get("device_name", "")).upper()
    _check(
        checks,
        "Repeated outer-training steps cross the written state",
        training_ok,
        str({name: item.get("tokens_per_second") for name, item in measured.items()}),
    )

    reference_tps = float(measured["reference_fp32"].get("tokens_per_second") or 0.0)
    exact_candidates = {
        name: float(measured[name].get("tokens_per_second") or 0.0)
        for name in ("exact_fp32", "exact_sdpa_fp32")
    }
    fastest_name, fastest_tps = max(exact_candidates.items(), key=lambda item: item[1])
    selected_default = fastest_name if fastest_tps >= reference_tps * 1.05 else "reference_fp32"
    performance_ok = reference_tps > 0.0 and all(value > 0.0 for value in exact_candidates.values())
    _check(
        checks,
        "Target-hardware default uses a 5% exact-backend threshold",
        performance_ok,
        f"reference={reference_tps:.3f}; fastest={fastest_name}:{fastest_tps:.3f}; selected={selected_default}",
    )
    return {
        "passed": all(bool(item["passed"]) for item in checks),
        "checks": checks,
        "selected_default": selected_default,
        "selection_rule": (
            "Select the fastest tensor-exact FP32 backend only at >=1.05x reference "
            "throughput; otherwise retain reference_fp32. Approximate variants are ineligible."
        ),
    }


def verify_a100_pilot_directory(output_directory: Path | str) -> dict[str, object]:
    """Strictly reload and verify an A100 pilot directory and its checksums."""

    output = Path(output_directory)
    loaded = {name: _strict_json(output / filename) for name, filename in EVIDENCE_FILES.items()}
    verification = evaluate_a100_evidence(
        loaded["preflight"],
        loaded["exact"],
        loaded["attention"],
        loaded["long_context"],
        loaded["training"],
    )
    manifest_path = output / "a100_pilot_manifest.json"
    if manifest_path.exists():
        manifest = _strict_json(manifest_path)
        expected = manifest.get("artifact_sha256", {})
        checksums_ok = all(
            expected.get(filename) == _sha256(output / filename)
            for filename in EVIDENCE_FILES.values()
        )
        verification["manifest_checksums_passed"] = checksums_ok
        verification["passed"] = bool(verification["passed"]) and checksums_ok
    return verification


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def run_a100_pilot(
    output_directory: Path | str,
    *,
    device: torch.device | str = "cuda",
    warmup_runs: int = 1,
    repetitions: int = 3,
) -> dict[str, object]:
    """Run the complete isolated A100 evidence bundle and write its manifest."""

    if repetitions < 3:
        raise ValueError("the strict A100 pilot requires at least 3 repetitions")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    preflight = inspect_a100(device)
    preflight_path = output / EVIDENCE_FILES["preflight"]
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not preflight["is_a100"]:
        raise A100PilotUnavailableError(str(preflight["reason"]))
    selected = torch.device(device)
    exact = run_exact_acceleration_matrix(
        scales=(A100_EXACT_SCALE,),
        warmup_runs=warmup_runs,
        repetitions=repetitions,
        device=selected,
    )
    write_exact_acceleration_matrix(exact, output)
    attention = run_attention_backend_evidence(
        warmup_runs=max(2, warmup_runs),
        repetitions=max(10, repetitions),
        device=selected,
    )
    write_attention_backend_evidence(attention, output)
    long_context = run_long_context_study(
        scales=(A100_LONG_CONTEXT_SCALE,),
        variants=DEFAULT_LONG_CONTEXT_VARIANTS,
        device=selected,
    )
    write_long_context_study(long_context, output)
    training = run_training_step_matrix(
        scale=A100_TRAINING_STEP_SCALE,
        warmup_runs=warmup_runs,
        repetitions=repetitions,
        device=selected,
    )
    write_training_step_matrix(training, output)
    verification = evaluate_a100_evidence(
        preflight, exact.to_dict(), attention, long_context, training
    )
    checksums = {
        filename: _sha256(output / filename) for filename in EVIDENCE_FILES.values()
    }
    manifest = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "artifact_sha256": checksums,
        "verification": verification,
    }
    manifest_path = output / "a100_pilot_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    verified = verify_a100_pilot_directory(output)
    if not verified["passed"]:
        raise RuntimeError("A100 evidence was captured but failed strict verification")
    report_lines = [
        "# Stage B named-A100 pilot manifest",
        "",
        f"Commit: `{manifest['git_commit']}`; captured: `{manifest['created_utc']}`.",
        "",
        f"Strict verification: `{'PASS' if verified['passed'] else 'FAIL'}`.",
        f"Selected default: `{verified['selected_default']}`.",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    report_lines.extend(
        f"| {item['criterion']} | {'PASS' if item['passed'] else 'FAIL'} |"
        for item in verified["checks"]
    )
    report_lines.extend(("", "## SHA-256", ""))
    report_lines.extend(
        f"- `{filename}`: `{digest}`" for filename, digest in checksums.items()
    )
    (output / "a100_pilot_manifest.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="artifacts/titans_stage_b/a100", type=Path
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-runs", default=1, type=int)
    parser.add_argument("--repetitions", default=3, type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.preflight_only and args.verify_only:
        raise SystemExit("--preflight-only and --verify-only are mutually exclusive")
    if args.preflight_only:
        report = inspect_a100(args.device)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["is_a100"] else 2
    if args.verify_only:
        try:
            verification = verify_a100_pilot_directory(args.output_dir)
        except (
            AssertionError,
            FileNotFoundError,
            KeyError,
            OSError,
            StopIteration,
            TypeError,
            ValueError,
        ) as error:
            print(f"A100 evidence verification failed: {error}")
            return 1
        print(json.dumps(verification, indent=2, sort_keys=True))
        return 0 if verification["passed"] else 1
    try:
        manifest = run_a100_pilot(
            args.output_dir,
            device=args.device,
            warmup_runs=args.warmup_runs,
            repetitions=args.repetitions,
        )
    except A100PilotUnavailableError as error:
        print(f"A100 preflight failed: {error}")
        return 2
    print(json.dumps(manifest["verification"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
