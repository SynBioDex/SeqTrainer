# Titans paper-MAC Stage B fidelity and performance audit

**Stage C gate: NO_GO**

This audit covers synthetic mechanism and systems validation only. Stage C is genome/clade-separated 15 Gbp bacterial next-base foundation training. Stage D is promoter/CDS/phylum adaptation. Neither is implemented or started here.

## Executive decision

Selected default: `reference`. READY requires every Stage B exit criterion, a reproducible selected default, causal/state contracts, target-hardware performance, and retained synthetic mechanism advantage.

### Blocking conditions

- **Named Colab Pro A100 evidence is absent** — Stage C is a 15 Gbp foundation-training program; CPU evidence cannot establish A100 throughput, memory headroom, FP16/BF16 behavior, or exact-mask Flash support.
  Unblock: Run B3, B6, and B7 on an attached A100 with fixed seeds, raw repetitions, CUDA peak memory, FP32 memory island, BF16/FP16 attention, and forced exact-mask Flash probe.
- **No robust target-hardware performance case for changing the default** — The tensor-exact functional loop regressed to 0.821x on the B3 nimble CPU run; B7 single-stream timings are not a repeated A100 training case.
  Unblock: Use the isolated A100 pilot to measure repeated end-to-end training steps and choose reference or an exact backend; approximate windows may not satisfy this blocker.

## Stage B exit criteria

| Criterion | Result | Evidence |
| --- | --- | --- |
| Reference backend and all immutable Stage A gates pass | PASS | Stage A gates=True; B1 reference parity=True; final suite=93 passed/7 warnings |
| Declared exact backends preserve output/state/gradient semantics | PASS | B3 debug/nimble tensor-exact; B6 FP64/FP32 sequence, state, surprise, input, persistent-token, and attention-parameter gradients tensor-exact on CPU |
| Approximate backend has explicit staleness and speed–fidelity classification | PASS | classification=experimental_approximation_not_parity_equivalent; windows=['16', '2', '32', '4', '8']; promotion=False |
| Convolution and every available attention backend are causal | PASS | B2 gate/output prefix errors=0.0/0.0; B7 all variants=True |
| MacBook and named A100 measurements are reproducible | FAIL | MacBook debug/nimble artifacts and installed rerun command exist; verified named A100 pilot, CUDA memory, FP16, and forced-Flash mask probe unavailable |
| Adaptive memory retains synthetic advantage over frozen/no-memory | PASS | Stage A adaptive >32=1.0 vs frozen=0.125/no-memory=0.25; B7 512-token recall reference=1.0 vs controls=0.0 |
| B8 publishes an unambiguous Stage C decision | PASS | This audit computes READY only when every preceding exit criterion passes. |

## Backend classification

Exact and approximate paths are intentionally separate. `exact_scan` is unavailable; `approximate_scan` is experimental-only and cannot satisfy an exact-backend or default-performance gate.

| Backend | Exactness / availability | Flags | Causal result | Default |
| --- | --- | --- | --- | --- |
| reference | FP64 oracle / FP32 reference | `defaults: memory=reference, attention=multihead_attention, gates=token_wise, activation=float32` | PASS: immutable Stage A and B7 future-perturbation tests | True |
| exact_accelerated | tensor-exact functional-loop refactor | `memory=exact_accelerated` | PASS: attention/read-write path unchanged; B7 prefix test | False |
| exact_scan | restricted_only / unavailable for nonlinear MLP | `memory=exact_scan (rejected by registry)` | Not selectable | False |
| approximate_scan | experimental stale-within-window approximation | `memory=approximate_scan; approximate_window=2|4|8|16|32` | PASS: post-core write and B7 prefix tests unchanged | False |
| causal_convolution | repository-defined opt-in gate context; not paper-exact | `gates=causal_convolution; convolution_kernel_size=3; memory=reference` | PASS: gate/output prefix max error 0.0/0.0 | False |
| sdpa | FP64 oracle / FP32 tensor-exact attention adapter on CPU | `attention=sdpa; activation=float32|bfloat16|float16(device-gated)` | PASS: 0 mask mismatches; prefix error 0.0 | False |
| flash | unavailable pending forced exact-mask A100 probe | `attention=flash only after probe_and_enable_flash succeeds` | Not selectable on current host | False |
| frozen_memory / no_memory controls | controls, not candidate backends | `benchmark-only controls` | PASS in B7 future-prefix matrix | False |

### reference

- Paper mapping: Equations 11–15 and read-once/integrate/write-once MAC; authoritative [P,H,S] mask
- Output/state/gradient evidence: B1 reference-to-reference output/retrieval/state/surprise/trainable-gradient error 0
- Hardware/dtype/precision: Intel MacBook CPU; FP64 oracle and FP32 execution; A100 unavailable
- Geometry/timing protocol: B1 1x d=8: 326.69 tok/s, 1 warmup/3 reps; B7 nimble 4x d=256: 53.11 tok/s
- Evidence artifact: `artifacts/titans_stage_b/b1_reference_macbook.json`
- Seed/timing: seed 20260727; 1 warmup/3 repetitions at B1; B7 fixed stream seed 20260738
- Command: `/Users/gonzalovidal/opt/anaconda3/envs/seqtrainer/bin/python -m seqtrainer.torch.titans_paper_mac_stage_b.benchmark --d-model 8 --num-heads 2 --memory-depth 1 --segments 1 --warmup-runs 1 --repetitions 3 --stem b1_reference_macbook`
- Limitations: No target A100 training-step performance or memory-capacity evidence.

### exact_accelerated

- Paper mapping: Same nonlinear evolving-gradient recurrence; execution refactor only
- Output/state/gradient evidence: B3 FP32 debug/nimble tensor-exact; B7 context drift 0
- Hardware/dtype/precision: Intel MacBook CPU FP32; A100 unavailable
- Geometry/timing protocol: B3 debug speedup 1.232x; nimble 0.821x; 1 warmup/1 rep
- Evidence artifact: `artifacts/titans_stage_b/b3_exact_acceleration_matrix.json`
- Seed/timing: seeds 20260727/20260728; 1 warmup/1 repetition per B3 scale
- Command: `Python API: run_exact_acceleration_matrix(); write_exact_acceleration_matrix()`
- Limitations: Not consistently faster; no A100 result.

### exact_scan

- Paper mapping: Paper linear/LTI affine example only; no proven nonlinear associative composition
- Output/state/gradient evidence: Linear state error 4.441e-16; nonlinear stale state error 1.494e+00, gradient error 1.216e+01
- Hardware/dtype/precision: FP64 CPU feasibility harness, seed 20260727
- Geometry/timing protocol: Algebraic feasibility harness, not a performance backend
- Evidence artifact: `artifacts/titans_stage_b/b4_exact_scan_feasibility.json`
- Seed/timing: seed 20260727; FP64 algebraic feasibility harness (not timing)
- Command: `Python API: run_scan_feasibility_harness(); write_scan_feasibility_artifact()`
- Limitations: Exact nonlinear scan remains unproven and unavailable.

### approximate_scan

- Paper mapping: Engineering ablation motivated by chunked parallelism; not paper-exact
- Output/state/gradient evidence: w2 dense state rel-L2=0.756, gradient rel-L2=1.167; all windows non-equivalent
- Hardware/dtype/precision: Intel MacBook CPU FP32; CUDA peak unavailable
- Geometry/timing protocol: B5 w2 local speedup=1.187x; B7 nimble w2=51.65 tok/s
- Evidence artifact: `artifacts/titans_stage_b/b5_approximate_scan_study.json`
- Seed/timing: seed 20260736; 1 warmup/3 repetitions; Stage A task seed protocol embedded
- Command: `Python API: run_approximate_scan_study(); write_approximate_scan_study()`
- Limitations: Sparse recall task does not activate within-window staleness; promotion forbidden.

### causal_convolution

- Paper mapping: Minimal left-padded depthwise causal convolution before adaptive gate projection
- Output/state/gradient evidence: Disabled flag exactly recovers Stage A; enabled path intentionally changes gates/state and has nonzero gradients
- Hardware/dtype/precision: FP64 mechanism comparison; FP32 MacBook long-stream matrix
- Geometry/timing protocol: B7 nimble 4x d=256: 73.79 tok/s, single stream
- Evidence artifact: `artifacts/titans_stage_b/b2_convolution_comparison.json`
- Seed/timing: seed 20260733; FP64 matched mechanism comparison; B7 seed 20260738
- Command: `Python API: run_convolution_comparison(); write_convolution_comparison()`
- Limitations: Paper does not specify Section 5.9 convolution placement or kernel; no A100 result.

### sdpa

- Paper mapping: Execution substitution only; exact [P,H,S] edge set retained
- Output/state/gradient evidence: FP64 tensor-exact=True; FP32 tensor-exact=True; BF16 behavioral only
- Hardware/dtype/precision: CPU FP64/FP32/BF16; CPU FP16 unavailable; A100/Flash unavailable
- Geometry/timing protocol: B6 d=64 MHA 10355.89 vs SDPA 11130.96 tok/s, 2 warmups/10 reps
- Evidence artifact: `artifacts/titans_stage_b/b6_attention_backend_evidence.json`
- Seed/timing: seed 20260735; 2 warmups/10 repetitions for CPU attention timing
- Command: `Python API: run_attention_backend_evidence(); write_attention_backend_evidence()`
- Limitations: No A100 BF16/FP16 or forced-Flash exact-mask evidence.

### flash

- Paper mapping: Kernel substitution only; mask may never be weakened
- Output/state/gradient evidence: No behavioral/parity result because no CUDA/A100 device is attached
- Hardware/dtype/precision: Required A100 FP16/BF16 evidence absent
- Geometry/timing protocol: Unavailable
- Evidence artifact: `artifacts/titans_stage_b/b6_attention_backend_evidence.json`
- Seed/timing: B6 exact-mask hardware probe; unavailable without named A100
- Command: `Python API on A100: StageBBackendRegistry().probe_and_enable_flash(block)`
- Limitations: unavailable: no CUDA device attached

### frozen_memory / no_memory controls

- Paper mapping: Ablate state update or retrieval to test adaptive-memory value
- Output/state/gradient evidence: Expected behavioral difference
- Hardware/dtype/precision: MacBook CPU FP32
- Geometry/timing protocol: B7 reports debug/nimble control latency and zero update norm
- Evidence artifact: `artifacts/titans_stage_b/b7_long_context_study.json`
- Seed/timing: seed 20260738; one multi-segment stream with per-segment samples
- Command: `/Users/gonzalovidal/opt/anaconda3/envs/seqtrainer/bin/seqtrainer-titans-stage-b-long-context`
- Limitations: Controls intentionally remove memory capability and are not deployment candidates.

## Hardware and reproducibility

MacBook: available on `macOS-10.16-x86_64-i386-64bit`, `CPU x86_64`, PyTorch `2.2.2`. Measured scales: debug_d64, debug_d128, nimble.

A100: **unavailable** — no strictly verified A100 pilot bundle is present

- Final tests: `/Users/gonzalovidal/opt/anaconda3/envs/seqtrainer/bin/python -m pytest -q` → 93 passed, 7 warnings.
- Stage A gate: `/Users/gonzalovidal/opt/anaconda3/envs/seqtrainer/bin/python -m seqtrainer.torch.titans_paper_mac.benchmark --output-dir artifacts/titans_stage_a` → `True`.
- Long context: `/Users/gonzalovidal/opt/anaconda3/envs/seqtrainer/bin/seqtrainer-titans-stage-b-long-context`.
- Regenerate audit: `/Users/gonzalovidal/opt/anaconda3/envs/seqtrainer/bin/seqtrainer-titans-stage-b-audit --test-count 93 --warning-count 7`.

Seeds and raw timing protocols are stored per source artifact: B1/B3/B4 use seed 20260727 (B3 nimble 20260728); B2 20260733; B5 20260736; B6 20260735; B7 20260738. Warmups, repetitions, geometry, raw samples, state bytes, and unavailable hardware fields remain in JSON rather than being inferred here.

## Evidence index

- stage_a: `artifacts/titans_stage_a/stage_a_benchmark.json`
- b1: `artifacts/titans_stage_b/b1_reference_macbook.json`
- b2: `artifacts/titans_stage_b/b2_convolution_comparison.json`
- b3: `artifacts/titans_stage_b/b3_exact_acceleration_matrix.json`
- b4: `artifacts/titans_stage_b/b4_exact_scan_feasibility.json`
- b5: `artifacts/titans_stage_b/b5_approximate_scan_study.json`
- b6: `artifacts/titans_stage_b/b6_attention_backend_evidence.json`
- b7: `artifacts/titans_stage_b/b7_long_context_study.json`

## Final gate

**NO_GO: Stage C may not begin.**

Stage B implementation and CPU evidence are complete, but a completed audit is not the same as a passed Stage C gate. Re-run this audit after the named A100 blockers are resolved; do not substitute approximate-scan speed for exactness or synthetic recall for genomic evidence.
