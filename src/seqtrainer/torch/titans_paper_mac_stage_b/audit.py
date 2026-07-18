"""Evidence-backed Stage B audit and binary Stage C gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence


SOURCE_FILES = {
    "stage_a": "artifacts/titans_stage_a/stage_a_benchmark.json",
    "b1": "artifacts/titans_stage_b/b1_reference_macbook.json",
    "b2": "artifacts/titans_stage_b/b2_convolution_comparison.json",
    "b3": "artifacts/titans_stage_b/b3_exact_acceleration_matrix.json",
    "b4": "artifacts/titans_stage_b/b4_exact_scan_feasibility.json",
    "b5": "artifacts/titans_stage_b/b5_approximate_scan_study.json",
    "b6": "artifacts/titans_stage_b/b6_attention_backend_evidence.json",
    "b7": "artifacts/titans_stage_b/b7_long_context_study.json",
}


def _load_sources(repository_root: Path) -> dict[str, object]:
    sources: dict[str, object] = {}
    for name, relative in SOURCE_FILES.items():
        path = repository_root / relative
        if not path.exists():
            raise FileNotFoundError(f"required Stage B evidence is missing: {path}")
        sources[name] = json.loads(path.read_text(encoding="utf-8"))
    return sources


def _scale_variant(b7: Mapping[str, object], scale_name: str, variant_name: str):
    scale = next(
        item
        for item in b7["long_stream_scales"]
        if item["scale"]["name"] == scale_name
    )
    return next(item for item in scale["variants"] if item["variant"] == variant_name)


def build_stage_b_audit(
    repository_root: Path | str = Path("."),
    *,
    final_test_count: int,
    final_warning_count: int,
) -> dict[str, object]:
    """Load immutable artifacts and compute the binary Stage C decision."""

    root = Path(repository_root)
    raw = _load_sources(root)
    stage_a = raw["stage_a"]
    b1 = raw["b1"]
    b2 = raw["b2"]
    b3 = raw["b3"]
    b4 = raw["b4"]
    b5 = raw["b5"]
    b6 = raw["b6"]
    b7 = raw["b7"]
    a100_available = bool(b7["hardware"]["a100_available"])
    exact_scales = {
        item["scale"]["name"]: item for item in b3["results"]
    }
    b5_windows = b5["dense_mechanism"]["approximate_windows"]
    b6_fp64 = b6["parity"]["fp64_oracle"]
    b6_fp32 = b6["parity"]["fp32_numerical"]
    b7_recall = b7["controlled_long_recall"]
    requirements = [
        {
            "criterion": "Reference backend and all immutable Stage A gates pass",
            "passed": bool(stage_a["gates"]["passed"]) and bool(b1["parity"]["passed"]),
            "evidence": (
                f"Stage A gates={stage_a['gates']['passed']}; B1 reference parity={b1['parity']['passed']}; "
                f"final suite={final_test_count} passed/{final_warning_count} warnings"
            ),
        },
        {
            "criterion": "Declared exact backends preserve output/state/gradient semantics",
            "passed": (
                bool(exact_scales["debug"]["tensor_exact"])
                and bool(exact_scales["nimble"]["tensor_exact"])
                and bool(b6_fp64["sequence"]["tensor_exact"])
                and bool(b6_fp32["sequence"]["tensor_exact"])
            ),
            "evidence": (
                "B3 debug/nimble tensor-exact; B6 FP64/FP32 sequence, state, surprise, input, "
                "persistent-token, and attention-parameter gradients tensor-exact on CPU"
            ),
        },
        {
            "criterion": "Approximate backend has explicit staleness and speed–fidelity classification",
            "passed": (
                b5["classification"]
                == "experimental_approximation_not_parity_equivalent"
                and b5["decision"]["promotion_allowed"] is False
                and set(b5_windows) == {"2", "4", "8", "16", "32"}
            ),
            "evidence": (
                f"classification={b5['classification']}; windows={sorted(b5_windows)}; "
                f"promotion={b5['decision']['promotion_allowed']}"
            ),
        },
        {
            "criterion": "Convolution and every available attention backend are causal",
            "passed": bool(b2["causality"]["passed"]) and bool(b7["causality"]["passed"]),
            "evidence": (
                f"B2 gate/output prefix errors={b2['causality']['gate_prefix_maximum_error']}/"
                f"{b2['causality']['current_output_prefix_maximum_error']}; "
                f"B7 all variants={b7['causality']['passed']}"
            ),
        },
        {
            "criterion": "MacBook and named A100 measurements are reproducible",
            "passed": a100_available,
            "evidence": (
                "MacBook debug/nimble artifacts and installed rerun command exist; "
                + (
                    "named A100 pilot measured"
                    if a100_available
                    else "named A100 pilot, CUDA memory, FP16, and forced-Flash mask probe unavailable"
                )
            ),
        },
        {
            "criterion": "Adaptive memory retains synthetic advantage over frozen/no-memory",
            "passed": (
                bool(stage_a["gates"]["adaptive_beats_controls_beyond_32"])
                and b7_recall["reference"]["delay_accuracy_by_context_tokens"]["512"]
                > max(
                    b7_recall["frozen_memory"]["delay_accuracy_by_context_tokens"]["512"],
                    b7_recall["no_memory"]["delay_accuracy_by_context_tokens"]["512"],
                )
            ),
            "evidence": (
                "Stage A adaptive >32=1.0 vs frozen=0.125/no-memory=0.25; "
                "B7 512-token recall reference=1.0 vs controls=0.0"
            ),
        },
        {
            "criterion": "B8 publishes an unambiguous Stage C decision",
            "passed": True,
            "evidence": "This audit computes READY only when every preceding exit criterion passes.",
        },
    ]
    exit_criteria_passed = all(bool(item["passed"]) for item in requirements)
    decision = "READY" if exit_criteria_passed else "NO_GO"
    blockers: list[dict[str, object]] = []
    if not a100_available:
        blockers.append(
            {
                "blocker": "Named Colab Pro A100 evidence is absent",
                "why_it_blocks": (
                    "Stage C is a 15 Gbp foundation-training program; CPU evidence cannot establish "
                    "A100 throughput, memory headroom, FP16/BF16 behavior, or exact-mask Flash support."
                ),
                "unblock": (
                    "Run B3, B6, and B7 on an attached A100 with fixed seeds, raw repetitions, CUDA "
                    "peak memory, FP32 memory island, BF16/FP16 attention, and forced exact-mask Flash probe."
                ),
            }
        )
    nimble_speedup = exact_scales["nimble"]["speedup"]
    if nimble_speedup is None or float(nimble_speedup) <= 1.0:
        blockers.append(
            {
                "blocker": "No robust target-hardware performance case for changing the default",
                "why_it_blocks": (
                    f"The tensor-exact functional loop regressed to {nimble_speedup:.3f}x on the B3 "
                    "nimble CPU run; B7 single-stream timings are not a repeated A100 training case."
                ),
                "unblock": (
                    "Use repeated A100 end-to-end training-step measurements to choose reference or an "
                    "exact backend; approximate windows may not satisfy this blocker."
                ),
            }
        )

    backends = [
        {
            "name": "reference",
            "exactness_class": "FP64 oracle / FP32 reference",
            "feature_flags": "defaults: memory=reference, attention=multihead_attention, gates=token_wise, activation=float32",
            "paper_mapping": "Equations 11–15 and read-once/integrate/write-once MAC; authoritative [P,H,S] mask",
            "causal_result": "PASS: immutable Stage A and B7 future-perturbation tests",
            "parity": "B1 reference-to-reference output/retrieval/state/surprise/trainable-gradient error 0",
            "hardware_precision": "Intel MacBook CPU; FP64 oracle and FP32 execution; A100 unavailable",
            "geometry_timing": (
                f"B1 1x d=8: {b1['timing']['tokens_per_second']:.2f} tok/s, 1 warmup/3 reps; "
                f"B7 nimble 4x d=256: {_scale_variant(b7, 'nimble', 'reference')['tokens_per_second']:.2f} tok/s"
            ),
            "default": True,
            "limitations": "No target A100 training-step performance or memory-capacity evidence.",
        },
        {
            "name": "exact_accelerated",
            "exactness_class": "tensor-exact functional-loop refactor",
            "feature_flags": "memory=exact_accelerated",
            "paper_mapping": "Same nonlinear evolving-gradient recurrence; execution refactor only",
            "causal_result": "PASS: attention/read-write path unchanged; B7 prefix test",
            "parity": "B3 FP32 debug/nimble tensor-exact; B7 context drift 0",
            "hardware_precision": "Intel MacBook CPU FP32; A100 unavailable",
            "geometry_timing": (
                f"B3 debug speedup {exact_scales['debug']['speedup']:.3f}x; "
                f"nimble {exact_scales['nimble']['speedup']:.3f}x; 1 warmup/1 rep"
            ),
            "default": False,
            "limitations": "Not consistently faster; no A100 result.",
        },
        {
            "name": "exact_scan",
            "exactness_class": "restricted_only / unavailable for nonlinear MLP",
            "feature_flags": "memory=exact_scan (rejected by registry)",
            "paper_mapping": "Paper linear/LTI affine example only; no proven nonlinear associative composition",
            "causal_result": "Not selectable",
            "parity": (
                f"Linear state error {b4['linear_state_max_abs_error']:.3e}; nonlinear stale state "
                f"error {b4['nonlinear_stale_state_max_abs_error']:.3e}, gradient error "
                f"{b4['nonlinear_stale_gradient_max_abs_error']:.3e}"
            ),
            "hardware_precision": "FP64 CPU feasibility harness, seed 20260727",
            "geometry_timing": "Algebraic feasibility harness, not a performance backend",
            "default": False,
            "limitations": "Exact nonlinear scan remains unproven and unavailable.",
        },
        {
            "name": "approximate_scan",
            "exactness_class": "experimental stale-within-window approximation",
            "feature_flags": "memory=approximate_scan; approximate_window=2|4|8|16|32",
            "paper_mapping": "Engineering ablation motivated by chunked parallelism; not paper-exact",
            "causal_result": "PASS: post-core write and B7 prefix tests unchanged",
            "parity": (
                f"w2 dense state rel-L2={b5_windows['2']['fast_weights']['relative_l2_error']:.3f}, "
                f"gradient rel-L2={b5_windows['2']['gradient_relative_l2_error']:.3f}; all windows non-equivalent"
            ),
            "hardware_precision": "Intel MacBook CPU FP32; CUDA peak unavailable",
            "geometry_timing": (
                f"B5 w2 local speedup={b5_windows['2']['speedup_over_reference']:.3f}x; "
                f"B7 nimble w2={_scale_variant(b7, 'nimble', 'approximate_w2')['tokens_per_second']:.2f} tok/s"
            ),
            "default": False,
            "limitations": "Sparse recall task does not activate within-window staleness; promotion forbidden.",
        },
        {
            "name": "causal_convolution",
            "exactness_class": "repository-defined opt-in gate context; not paper-exact",
            "feature_flags": "gates=causal_convolution; convolution_kernel_size=3; memory=reference",
            "paper_mapping": "Minimal left-padded depthwise causal convolution before adaptive gate projection",
            "causal_result": (
                f"PASS: gate/output prefix max error {b2['causality']['gate_prefix_maximum_error']}/"
                f"{b2['causality']['current_output_prefix_maximum_error']}"
            ),
            "parity": "Disabled flag exactly recovers Stage A; enabled path intentionally changes gates/state and has nonzero gradients",
            "hardware_precision": "FP64 mechanism comparison; FP32 MacBook long-stream matrix",
            "geometry_timing": (
                f"B7 nimble 4x d=256: {_scale_variant(b7, 'nimble', 'causal_convolution')['tokens_per_second']:.2f} tok/s, single stream"
            ),
            "default": False,
            "limitations": "Paper does not specify Section 5.9 convolution placement or kernel; no A100 result.",
        },
        {
            "name": "sdpa",
            "exactness_class": "FP64 oracle / FP32 tensor-exact attention adapter on CPU",
            "feature_flags": "attention=sdpa; activation=float32|bfloat16|float16(device-gated)",
            "paper_mapping": "Execution substitution only; exact [P,H,S] edge set retained",
            "causal_result": (
                f"PASS: {b6['mask']['boolean_complement_mismatches']} mask mismatches; "
                f"prefix error {b6['causality']['prefix_maximum_error']}"
            ),
            "parity": (
                f"FP64 tensor-exact={b6_fp64['sequence']['tensor_exact']}; "
                f"FP32 tensor-exact={b6_fp32['sequence']['tensor_exact']}; BF16 behavioral only"
            ),
            "hardware_precision": "CPU FP64/FP32/BF16; CPU FP16 unavailable; A100/Flash unavailable",
            "geometry_timing": (
                f"B6 d=64 MHA {b6['timing']['multihead_attention']['tokens_per_second']:.2f} vs "
                f"SDPA {b6['timing']['sdpa']['tokens_per_second']:.2f} tok/s, 2 warmups/10 reps"
            ),
            "default": False,
            "limitations": "No A100 BF16/FP16 or forced-Flash exact-mask evidence.",
        },
        {
            "name": "flash",
            "exactness_class": "unavailable pending forced exact-mask A100 probe",
            "feature_flags": "attention=flash only after probe_and_enable_flash succeeds",
            "paper_mapping": "Kernel substitution only; mask may never be weakened",
            "causal_result": "Not selectable on current host",
            "parity": "No behavioral/parity result because no CUDA/A100 device is attached",
            "hardware_precision": "Required A100 FP16/BF16 evidence absent",
            "geometry_timing": "Unavailable",
            "default": False,
            "limitations": str(b6["flash_probe"]["reason"]),
        },
        {
            "name": "frozen_memory / no_memory controls",
            "exactness_class": "controls, not candidate backends",
            "feature_flags": "benchmark-only controls",
            "paper_mapping": "Ablate state update or retrieval to test adaptive-memory value",
            "causal_result": "PASS in B7 future-prefix matrix",
            "parity": "Expected behavioral difference",
            "hardware_precision": "MacBook CPU FP32",
            "geometry_timing": "B7 reports debug/nimble control latency and zero update norm",
            "default": False,
            "limitations": "Controls intentionally remove memory capability and are not deployment candidates.",
        },
    ]
    provenance = {
        "reference": {
            "artifact": SOURCE_FILES["b1"],
            "seed_timing": "seed 20260727; 1 warmup/3 repetitions at B1; B7 fixed stream seed 20260738",
            "command": (
                "/Users/gonzalovidal/opt/anaconda3/envs/seqtrainer/bin/python -m "
                "seqtrainer.torch.titans_paper_mac_stage_b.benchmark --d-model 8 "
                "--num-heads 2 --memory-depth 1 --segments 1 --warmup-runs 1 "
                "--repetitions 3 --stem b1_reference_macbook"
            ),
        },
        "exact_accelerated": {
            "artifact": SOURCE_FILES["b3"],
            "seed_timing": "seeds 20260727/20260728; 1 warmup/1 repetition per B3 scale",
            "command": "Python API: run_exact_acceleration_matrix(); write_exact_acceleration_matrix()",
        },
        "exact_scan": {
            "artifact": SOURCE_FILES["b4"],
            "seed_timing": "seed 20260727; FP64 algebraic feasibility harness (not timing)",
            "command": "Python API: run_scan_feasibility_harness(); write_scan_feasibility_artifact()",
        },
        "approximate_scan": {
            "artifact": SOURCE_FILES["b5"],
            "seed_timing": "seed 20260736; 1 warmup/3 repetitions; Stage A task seed protocol embedded",
            "command": "Python API: run_approximate_scan_study(); write_approximate_scan_study()",
        },
        "causal_convolution": {
            "artifact": SOURCE_FILES["b2"],
            "seed_timing": "seed 20260733; FP64 matched mechanism comparison; B7 seed 20260738",
            "command": "Python API: run_convolution_comparison(); write_convolution_comparison()",
        },
        "sdpa": {
            "artifact": SOURCE_FILES["b6"],
            "seed_timing": "seed 20260735; 2 warmups/10 repetitions for CPU attention timing",
            "command": "Python API: run_attention_backend_evidence(); write_attention_backend_evidence()",
        },
        "flash": {
            "artifact": SOURCE_FILES["b6"],
            "seed_timing": "B6 exact-mask hardware probe; unavailable without named A100",
            "command": "Python API on A100: StageBBackendRegistry().probe_and_enable_flash(block)",
        },
        "frozen_memory / no_memory controls": {
            "artifact": SOURCE_FILES["b7"],
            "seed_timing": "seed 20260738; one multi-segment stream with per-segment samples",
            "command": b7["protocol"]["rerun_command"],
        },
    }
    for backend in backends:
        backend.update(provenance[backend["name"]])

    return {
        "format_version": 1,
        "audit_scope": "Titans paper-MAC Stage B synthetic fidelity/performance platform",
        "stage_a_reference_commit": "46dd0158523377ef36cce0edc75743879c109387",
        "stage_b_evidence_commits": {
            "B0_B1": "490fc01",
            "B4": "52eb72a",
            "B3": "0716327",
            "B2": "8d7753f",
            "B6": "715e1a7",
            "B5": "7976540",
            "B7": "f8b54a5",
        },
        "selected_default": "reference",
        "requirements": requirements,
        "backends": backends,
        "source_artifacts": SOURCE_FILES,
        "validation": {
            "environment": "/Users/gonzalovidal/opt/anaconda3/envs/seqtrainer",
            "test_command": (
                "/Users/gonzalovidal/opt/anaconda3/envs/seqtrainer/bin/python -m pytest -q"
            ),
            "test_count": final_test_count,
            "warning_count": final_warning_count,
            "stage_a_command": (
                "/Users/gonzalovidal/opt/anaconda3/envs/seqtrainer/bin/python -m "
                "seqtrainer.torch.titans_paper_mac.benchmark --output-dir artifacts/titans_stage_a"
            ),
            "stage_a_gates_passed": bool(stage_a["gates"]["passed"]),
            "long_context_command": b7["protocol"]["rerun_command"],
            "audit_command": (
                "/Users/gonzalovidal/opt/anaconda3/envs/seqtrainer/bin/"
                f"seqtrainer-titans-stage-b-audit --test-count {final_test_count} "
                f"--warning-count {final_warning_count}"
            ),
        },
        "hardware_summary": {
            "macbook": {
                "available": True,
                "platform": b7["hardware"]["platform"],
                "device": b7["hardware"]["device_name"],
                "torch": b7["hardware"]["torch"],
                "scales": ["debug_d64", "debug_d128", "nimble"],
            },
            "a100": {
                "available": a100_available,
                "reason": next(
                    item["reason"]
                    for item in b7["long_stream_scales"]
                    if item["scale"]["name"] == "a100_pilot"
                ),
            },
        },
        "stage_c_gate": {
            "decision": decision,
            "ready": exit_criteria_passed,
            "definition": (
                "Genome/clade-separated 15 Gbp bacterial next-base foundation training."
            ),
            "blockers": blockers,
            "rule": (
                "READY requires every Stage B exit criterion, a reproducible selected default, "
                "causal/state contracts, target-hardware performance, and retained synthetic mechanism advantage."
            ),
        },
        "stage_d_definition": (
            "Promoter/CDS/phylum adaptation after Stage C; not implemented in Stage B."
        ),
        "scope_exclusions": [
            "No 15 Gbp corpus training was started.",
            "No promoter/CDS/phylum adaptation was implemented.",
            "No biology-performance claim is made from synthetic tasks.",
        ],
    }


def render_fidelity_performance_audit(audit: Mapping[str, object]) -> str:
    decision = audit["stage_c_gate"]
    lines = [
        "# Titans paper-MAC Stage B fidelity and performance audit",
        "",
        f"**Stage C gate: {decision['decision']}**",
        "",
        "This audit covers synthetic mechanism and systems validation only. Stage C is genome/clade-separated 15 Gbp bacterial next-base foundation training. Stage D is promoter/CDS/phylum adaptation. Neither is implemented or started here.",
        "",
        "## Executive decision",
        "",
        f"Selected default: `{audit['selected_default']}`. {decision['rule']}",
        "",
    ]
    if decision["blockers"]:
        lines.extend(("### Blocking conditions", ""))
        for blocker in decision["blockers"]:
            lines.extend(
                (
                    f"- **{blocker['blocker']}** — {blocker['why_it_blocks']}",
                    f"  Unblock: {blocker['unblock']}",
                )
            )
        lines.append("")
    lines.extend(
        (
            "## Stage B exit criteria",
            "",
            "| Criterion | Result | Evidence |",
            "| --- | --- | --- |",
        )
    )
    for item in audit["requirements"]:
        lines.append(
            f"| {item['criterion']} | {'PASS' if item['passed'] else 'FAIL'} | {item['evidence']} |"
        )
    lines.extend(
        (
            "",
            "## Backend classification",
            "",
            "Exact and approximate paths are intentionally separate. `exact_scan` is unavailable; `approximate_scan` is experimental-only and cannot satisfy an exact-backend or default-performance gate.",
            "",
            "| Backend | Exactness / availability | Flags | Causal result | Default |",
            "| --- | --- | --- | --- | --- |",
        )
    )
    for backend in audit["backends"]:
        lines.append(
            f"| {backend['name']} | {backend['exactness_class']} | `{backend['feature_flags']}` | "
            f"{backend['causal_result']} | {backend['default']} |"
        )
    lines.append("")
    for backend in audit["backends"]:
        lines.extend(
            (
                f"### {backend['name']}",
                "",
                f"- Paper mapping: {backend['paper_mapping']}",
                f"- Output/state/gradient evidence: {backend['parity']}",
                f"- Hardware/dtype/precision: {backend['hardware_precision']}",
                f"- Geometry/timing protocol: {backend['geometry_timing']}",
                f"- Evidence artifact: `{backend['artifact']}`",
                f"- Seed/timing: {backend['seed_timing']}",
                f"- Command: `{backend['command']}`",
                f"- Limitations: {backend['limitations']}",
                "",
            )
        )
    hardware = audit["hardware_summary"]
    validation = audit["validation"]
    lines.extend(
        (
            "## Hardware and reproducibility",
            "",
            f"MacBook: available on `{hardware['macbook']['platform']}`, `{hardware['macbook']['device']}`, PyTorch `{hardware['macbook']['torch']}`. Measured scales: {', '.join(hardware['macbook']['scales'])}.",
            "",
            f"A100: **{'available' if hardware['a100']['available'] else 'unavailable'}** — {hardware['a100']['reason']}",
            "",
            f"- Final tests: `{validation['test_command']}` → {validation['test_count']} passed, {validation['warning_count']} warnings.",
            f"- Stage A gate: `{validation['stage_a_command']}` → `{validation['stage_a_gates_passed']}`.",
            f"- Long context: `{validation['long_context_command']}`.",
            f"- Regenerate audit: `{validation['audit_command']}`.",
            "",
            "Seeds and raw timing protocols are stored per source artifact: B1/B3/B4 use seed 20260727 (B3 nimble 20260728); B2 20260733; B5 20260736; B6 20260735; B7 20260738. Warmups, repetitions, geometry, raw samples, state bytes, and unavailable hardware fields remain in JSON rather than being inferred here.",
            "",
            "## Evidence index",
            "",
        )
    )
    for name, path in audit["source_artifacts"].items():
        lines.append(f"- {name}: `{path}`")
    lines.extend(
        (
            "",
            "## Final gate",
            "",
            f"**{decision['decision']}: Stage C may {'begin' if decision['ready'] else 'not begin'}.**",
            "",
            "Stage B implementation and CPU evidence are complete, but a completed audit is not the same as a passed Stage C gate. Re-run this audit after the named A100 blockers are resolved; do not substitute approximate-scan speed for exactness or synthetic recall for genomic evidence.",
            "",
        )
    )
    return "\n".join(lines)


def write_stage_b_audit(
    audit: Mapping[str, object],
    *,
    artifact_path: Path | str = Path("artifacts/titans_stage_b/b8_stage_b_audit.json"),
    document_path: Path | str = Path(
        "docs/titans_stage_b/fidelity_performance_audit.md"
    ),
) -> dict[str, Path]:
    artifact = Path(artifact_path)
    document = Path(document_path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    document.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    document.write_text(
        render_fidelity_performance_audit(audit), encoding="utf-8"
    )
    return {"json": artifact, "document": document}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--test-count", type=int, required=True)
    parser.add_argument("--warning-count", type=int, required=True)
    args = parser.parse_args(argv)
    audit = build_stage_b_audit(
        args.repository_root,
        final_test_count=args.test_count,
        final_warning_count=args.warning_count,
    )
    paths = write_stage_b_audit(
        audit,
        artifact_path=args.repository_root
        / "artifacts/titans_stage_b/b8_stage_b_audit.json",
        document_path=args.repository_root
        / "docs/titans_stage_b/fidelity_performance_audit.md",
    )
    print(
        json.dumps(
            {
                "decision": audit["stage_c_gate"]["decision"],
                "artifacts": {name: str(path) for name, path in paths.items()},
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
