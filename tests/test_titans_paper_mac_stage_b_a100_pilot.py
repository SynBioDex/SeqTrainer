from __future__ import annotations

import copy
import hashlib
import json

from seqtrainer.torch.titans_paper_mac_stage_b.a100_pilot import (
    EVIDENCE_FILES,
    evaluate_a100_evidence,
    inspect_a100,
    main,
    verify_a100_pilot_directory,
)


def _parity(error: float = 0.0) -> dict[str, dict[str, float | bool]]:
    return {
        name: {"maximum_absolute_error": error, "tensor_exact": error == 0.0}
        for name in (
            "sequence",
            "retrieval",
            "fast_weights",
            "surprise",
            "input_gradient",
            "persistent_token_gradient",
            "attention_gradient",
        )
    }


def _valid_evidence():
    preflight = {"is_a100": True, "device_name": "NVIDIA A100-SXM4-40GB"}
    timing = {
        "available": True,
        "wall_times_seconds": [1.0, 1.0, 1.0],
    }
    exact = {
        "results": [
            {
                "scale": {"name": "a100_pilot"},
                "device_name": "NVIDIA A100-SXM4-40GB",
                "repetitions": 3,
                "tensor_exact": True,
                "reference": timing,
                "exact_accelerated": timing,
            }
        ]
    }
    mixed_case = {
        "available": True,
        "memory_state_dtypes": ["float32"],
        "published_sequence_dtype": "float32",
    }
    attention = {
        "parity": {"fp64_oracle": _parity(), "fp32_numerical": _parity(1e-6)},
        "mask": {"boolean_complement_mismatches": 0},
        "causality": {"passed": True},
        "mixed_precision_behavioral": {
            "bfloat16": mixed_case,
            "float16": mixed_case,
        },
        "flash_probe": {
            "available": False,
            "device": "NVIDIA A100-SXM4-40GB",
            "reason": "forced Flash SDP rejected the exact mask: unsupported mask",
        },
    }
    long_variant = {
        "available": True,
        "first_nonfinite_context_tokens": None,
        "cuda_peak_allocated_bytes": 1024,
    }
    long_context = {
        "hardware": {"a100_available": True},
        "long_stream_scales": [
            {
                "scale": {"name": "a100_pilot"},
                "available": True,
                "variants": [long_variant],
            }
        ],
        "causality": {"passed": True},
        "controlled_long_recall": {
            "reference": {"delay_accuracy_by_context_tokens": {"512": 1.0}},
            "frozen_memory": {"delay_accuracy_by_context_tokens": {"512": 0.0}},
            "no_memory": {"delay_accuracy_by_context_tokens": {"512": 0.0}},
        },
    }

    def training_variant(name: str, throughput: float):
        return {
            "variant": name,
            "available": True,
            "repetitions": 3,
            "samples_seconds": [1.0, 1.0, 1.0],
            "all_gradients_finite": True,
            "output_and_state_finite": True,
            "memory_state_dtypes": ["float32"],
            "cuda_peak_allocated_bytes": 2048,
            "tokens_per_second": throughput,
        }

    training = {
        "hardware": {"device_name": "NVIDIA A100-SXM4-40GB"},
        "variants": [
            training_variant("reference_fp32", 100.0),
            training_variant("exact_fp32", 104.9),
            training_variant("exact_sdpa_fp32", 110.0),
        ],
    }
    return preflight, exact, attention, long_context, training


def test_a100_evidence_requires_all_contracts_and_uses_five_percent_threshold():
    evidence = _valid_evidence()
    result = evaluate_a100_evidence(*evidence)
    assert result["passed"]
    assert result["selected_default"] == "exact_sdpa_fp32"

    broken = copy.deepcopy(evidence)
    broken[4]["variants"][2]["tokens_per_second"] = 104.99
    result = evaluate_a100_evidence(*broken)
    assert result["passed"]
    assert result["selected_default"] == "reference_fp32"

    broken = copy.deepcopy(evidence)
    broken[2]["mixed_precision_behavioral"]["float16"]["memory_state_dtypes"] = [
        "float16"
    ]
    result = evaluate_a100_evidence(*broken)
    assert not result["passed"]


def test_a100_preflight_and_cli_fail_safely_on_cpu(tmp_path, capsys):
    report = inspect_a100("cpu")
    assert report["is_a100"] is False
    assert main(["--preflight-only", "--device", "cpu"]) == 2
    assert "requires a CUDA device" in capsys.readouterr().out
    assert main(["--verify-only", "--output-dir", str(tmp_path)]) == 1
    assert "verification failed" in capsys.readouterr().out


def test_directory_verifier_recomputes_manifest_checksums(tmp_path):
    values = dict(zip(EVIDENCE_FILES, _valid_evidence()))
    hashes = {}
    for name, filename in EVIDENCE_FILES.items():
        path = tmp_path / filename
        path.write_text(json.dumps(values[name]), encoding="utf-8")
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "a100_pilot_manifest.json").write_text(
        json.dumps({"artifact_sha256": hashes}), encoding="utf-8"
    )
    assert verify_a100_pilot_directory(tmp_path)["passed"]

    with (tmp_path / EVIDENCE_FILES["training"]).open("a", encoding="utf-8") as file:
        file.write("\n")
    verification = verify_a100_pilot_directory(tmp_path)
    assert verification["manifest_checksums_passed"] is False
    assert verification["passed"] is False
