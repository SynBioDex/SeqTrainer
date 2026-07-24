"""Write the bounded T4 handoff evidence from a successful GPU smoke report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def build_t4_evidence(smoke: Mapping[str, object]) -> dict[str, object]:
    if smoke.get("classification") != "stage_c_gpu_smoke" or smoke.get("passed") is not True:
        raise ValueError("T4 handoff requires a passed stage_c_gpu_smoke report")
    required = (
        "hardware",
        "geometry",
        "dataset",
        "fp32_training",
        "fp16_training",
        "fp32_cpu_gpu_parity",
        "fp16_causal_mask",
    )
    missing = [name for name in required if name not in smoke]
    if missing:
        raise ValueError(f"GPU smoke report is missing: {', '.join(missing)}")
    return {
        "format_version": 1,
        "classification": "stage_c_t4_bounded_evidence",
        "passed": True,
        "hardware": smoke["hardware"],
        "geometry": smoke["geometry"],
        "dataset": smoke["dataset"],
        "fp32_training": smoke["fp32_training"],
        "fp16_training": smoke["fp16_training"],
        "fp32_cpu_gpu_parity": smoke["fp32_cpu_gpu_parity"],
        "fp16_causal_mask": smoke["fp16_causal_mask"],
        "scope": {
            "validated": "one full-geometry horizon-3 optimizer step per FP32 and FP16 precision",
            "deferred_to_a100": "multi-step capacity, checkpoint, throughput, and extended validation matrix",
        },
    }


def write_t4_evidence(smoke_path: Path, output_dir: Path) -> dict[str, Path]:
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if not isinstance(smoke, Mapping):
        raise ValueError("GPU smoke report must be a JSON object")
    evidence = build_t4_evidence(smoke)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "t4_bounded_evidence.json"
    report_path = output_dir / "T4_BOUNDED_EVIDENCE.md"
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fp32 = evidence["fp32_training"]
    fp16 = evidence["fp16_training"]
    assert isinstance(fp32, Mapping) and isinstance(fp16, Mapping)
    report_path.write_text(
        "\n".join(
            (
                "# Stage C T4 bounded evidence",
                "",
                "The full multi-step capacity matrix is deferred to A100 because Colab T4 terminates "
                "a second long-running full-geometry subprocess. The GPU smoke already validated one "
                "horizon-3 optimizer step at both required precisions.",
                "",
                f"- FP32 optimizer steps: `{fp32['optimizer_steps']}`",
                f"- FP16 optimizer steps: `{fp16['optimizer_steps']}`",
                f"- FP32 valid bases: `{fp32['valid_bases']}`",
                f"- FP16 valid bases: `{fp16['valid_bases']}`",
                "- CPU/GPU FP32 parity and FP16 causal masking: passed.",
                "",
            )
        ),
        encoding="utf-8",
    )
    return {"json": json_path, "report": report_path}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = write_t4_evidence(args.smoke, args.output_dir)
    print(json.dumps({name: str(path) for name, path in paths.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
