# Titans paper-MAC Stage A fidelity and reproducibility audit

> **STAGE B DECISION: READY**
>
> Stage A contains a correct, differentiable neural-memory reference, a
> causality-tested minimal MAC block whose post-core output drives the single
> memory write, and a matched three-way benchmark using the real
> `FunctionalNeuralMemory` path. All correctness and benchmark gates pass. The
> Stage A worktree is captured in the commit containing this audit and its
> `uv.lock`; Stage B may proceed while preserving these reference tests.

Audit target: Behrouz, Zhong, and Mirrokni, [*Titans: Learning to Memorize at
Test Time*, arXiv:2501.00663v1](https://arxiv.org/abs/2501.00663). Equation
numbers below refer to v1. The repository's fixed Stage A contract additionally
requires segment length `C=32`, full per-position retrieval, a block-causal
interpretation, and explicit stream isolation.

## Mechanism-by-mechanism evidence

| Mechanism | Status | Code evidence | Test evidence | Audit finding |
| --- | --- | --- | --- | --- |
| Associative KV objective, equations 11–12 | Compliant | `memory.py:89-91` defines trainable K/V projections; `memory.py:131-152` computes `0.5 * ||M(k)-v||²` and its exact higher-order gradient. | `test_titans_paper_mac_memory.py:26-54` checks an FP64 hand calculation. | The factor `0.5` differs from the paper's displayed norm but only rescales the gradient; `theta` absorbs that conventional choice. |
| MLP neural-memory fast weights | Compliant | `memory.py:84-97` builds the MLP and exposes its full parameter pytree; `memory.py:121-129` evaluates it with `torch.func.functional_call`. | `test_titans_paper_mac_memory.py:57-74` proves state replacement and distinct stream states. | Stage A uses a complete MLP parameter pytree rather than a slot matrix. |
| Momentary surprise | Compliant | `memory.py:139-152` differentiates the associative loss with `create_graph=True`. | `test_titans_paper_mac_memory.py:26-52` compares surprise to the hand gradient. | Runtime update tensors are not detached. Serialization detaches intentionally because autograd graphs are process-local. |
| Momentum / past surprise, equations 9–10 and 14 | Compliant | `state.py:43-45` stores surprise alongside fast weights; `memory.py:154-162` applies `eta*S_(t-1) - theta*grad`. | `test_titans_paper_mac_memory.py:26-52` checks the one-step equation; lines 77–94 prove all 32 updates remain differentiable. | Surprise has the same ordered pytree and shapes as the memory weights. |
| Adaptive `alpha`, `eta`, `theta` | Compliant in the neural-memory reference | `memory.py:42-61` produces token-dependent sigmoid gates; `memory.py:164-197` applies forgetting and momentum; `memory.py:217-231` uses all 32 gate triplets. | `test_titans_paper_mac_memory.py:12-54` fixes known gates for the hand reference; lines 77–94 prove gradients reach the gate projection. | All three gates are restricted to `(0,1)` by the Stage A contract. The paper clearly makes `alpha` and `eta` data-dependent; the exact preferred parameterization remains underspecified. |
| Persistent memory | Compliant for the minimal core | `mac.py:103-119` owns learned input-independent persistent tokens; `mac.py:154-168` prepends them to `[H,S]`. | `test_titans_paper_mac_block.py:10-27` checks persistent visibility; lines 87–100 prove gradients reach persistent tokens. | Persistent tokens are outer-loop parameters and are never part of stream fast weights. |
| Full `h_t` retrieval | Compliant | `memory.py:124-129` retrieves by inference from a supplied state; `memory.py:238-255` projects and retrieves all 32 queries before producing the replacement state. | `test_titans_paper_mac_memory.py:57-69` and `test_titans_paper_mac_block.py:30-43` require shape `(32,d)` and equality with the incoming-state snapshot. | There is one retrieval vector per segment position, not top-k slots. |
| Read/write timing | Compliant | `memory.py:258-268` provides a read-only full-segment boundary. `mac.py:170-183` retrieves first, causally integrates second, and passes only the core sequence to `update_segment` for one returned replacement state. | `test_titans_paper_mac_block.py:30-51` proves pre-write retrieval, equality with a core-output update, inequality with a raw-input update, and one state transition. | Equations 24–25 are now represented directly: the causal core controls what is written to the next memory state. |
| Block-causal MAC mask | Compliant with the locked Stage A safety interpretation | `mac.py:28-63` explicitly allows persistent tokens plus `H_1:i` and `S_1:i`; `mac.py:159-168` passes it to standard multi-head attention. | `test_titans_paper_mac_block.py:10-27` checks every allowed edge; lines 50–84 perturb future retrievals and tokens. | Persistent-token queries are restricted to persistent keys. This is stricter than the paper's graphical mask and prevents them becoming a future-information side channel. |
| Stream isolation and lifecycle | Compliant at the harness/state layer | `state.py:33-107` makes state immutable and stream-named; `lifecycle.py:24-90` keys state by `stream_id`, resets before a segment, and ends after it. | `test_titans_paper_mac_streams.py:123-179` checks interleaving, stream-local reset, and save/resume equivalence. | The paper specifies a sequence recurrence, not multi-request ownership; explicit isolation is a repository safety contract. |
| State serialization | Compliant for stream state | `state.py:109-170` uses a versioned CPU payload preserving dtype, key order, and metadata; `lifecycle.py:97-143` serializes the stream map and ordering. | `test_titans_paper_mac_memory.py:97-128` checks tensor-exact restoration; `test_titans_paper_mac_streams.py:164-179` checks resumed transitions. | Model/optimizer/RNG checkpoint packaging is not implemented in this minimal Stage A package. Benchmark reproduction relies on fixed seeds and `uv.lock`. |
| Full differentiability | Compliant for the implemented path | `memory.py:145-151` retains the higher-order graph; state updates build replacement tensors without `.data`, in-place parameter mutation, or runtime detach. `mac.py:159-168` uses ordinary PyTorch attention. | `test_titans_paper_mac_memory.py:77-94` and `test_titans_paper_mac_block.py:87-100` prove gradients reach inputs, gates, attention, persistent tokens, and updated fast weights. | Detach appears only in explicit serialization and diagnostic metric collection, outside the differentiable forward/update path. |

## Benchmark evidence and limitation

The deterministic A4 command compares three conditions with the same
`FunctionalNeuralMemory`, 7,383 total parameters, 820 outer-optimized readout
parameters, AdamW optimizer, seeds, token budgets, and held-out fixtures. Its
default smoke result is:

| Variant | Delayed recall `>32` | Overwrite | Reset |
| --- | ---: | ---: | ---: |
| adaptive | 1.000 | 1.000 | 1.000 |
| frozen memory | 0.125 | 0.125 | 1.000 |
| no memory | 0.250 | 0.250 | 1.000 |

The benchmark gate therefore passes: adaptive exceeds the stronger control by
`0.750`, above the configured `0.50` margin. Its determinism and artifact
schema are tested in `tests/test_titans_paper_mac_benchmark.py:16-46`.

The benchmark now exercises `FunctionalNeuralMemory` directly. Controlled
synthetic encodings and K/V/Q projections make the expected association
auditable, while candidate writes use the same MLP fast weights, associative
loss, surprise, momentum, and sigmoid gate implementation as the MAC block.
Adaptive commits returned states; frozen memory discards candidates; no-memory
uses only the current query. All conditions retain identical module capacity.
Only the shared readout is outer-optimized, so the result is evidence for the
Stage A neural-memory mechanism under a controlled synthetic protocol—not a
general capacity, DNA, or biology-performance claim.

## Deliberate deviations

1. **Fixed `C=32`.** The paper permits general chunk sizes; Stage A hard-codes
   32 to keep every reference test and mask exact.
2. **Sequential reference recurrence.** All 32 inner updates run as a Python
   loop. This is the correctness reference before associative scans or
   convolution.
3. **Locked block-causal interpretation.** The paper's MAC attention map is
   graphical. Stage A makes both the retrieval and sequence prefixes causal and
   prevents persistent queries from reading data-dependent positions.
4. **Minimal core.** `PaperMACBlock` is one standard multi-head-attention layer,
   residual, and layer norm. It has no complete language-model head, feedforward
   stack, equation-25 fusion head, or DNA pipeline.
5. **Single-stream tensor API.** A block consumes `(32,d_model)` plus one named
   state. Multi-stream execution is represented by independent calls managed by
   `StreamLifecycleHarness`, not a batched fast-weight transform.
6. **Serialization detach.** Saved state is detached and restored as leaf
   tensors because autograd graphs cannot be portably serialized. Runtime
   forward/update operations remain differentiable.
7. **Controlled benchmark projections.** A4 uses deterministic disjoint key and
   value subspaces and fixed interpretable gate initialization. The actual
   functional neural-memory update runs, but only the readout is outer-trained;
   this remains a mechanism gate rather than a general learning benchmark.

## Unresolved ambiguities and gaps

- The exact tensor layout in paper equations 21–25 is not fully textual. Stage
  A's `[persistent, H_1..H_32, S_1..S_32]` block-causal layout is a documented
  safety interpretation.
- `valid_mask` suppresses associative writes for padded tail positions, but it
  is not passed as an attention key-padding mask. Fixed tail padding cannot
  affect earlier valid positions under the causal mask; arbitrary holes are not
  supported or tested.
- The minimal package has state serialization but not a combined
  model/optimizer/RNG checkpoint API.
- The present test corpus establishes equation/reference correctness, not
  numerical parity with an official implementation; no official implementation
  is identified by the paper record used for this audit.
- The benchmark proves the neural-memory mechanism under controlled
  projections. A later learned-projection experiment must not replace this
  deterministic reference gate; it should be additive.

## Explicitly excluded Stage B features

The following were absent by design in Stage A. Stage B may introduce them one
at a time, but each change must retain this sequential, unfused reference path
and all Stage A gates:

- global convolution or chunk-constant gate approximation;
- associative/parallel scans for the 32 update recurrence;
- fused or flash-attention substitution;
- kernel fusion, compilation, throughput tuning, or scaling experiments;
- long-context/DNA training and any biological performance claim.

## Reproduction contract

Audited environment on 2026-07-17:

| Component | Value |
| --- | --- |
| Branch | `feat/titans-paper-mac-stage-a` |
| Revision | the commit containing this audit and the complete Stage A patch |
| Platform | macOS 15.7.4, Intel x86_64 |
| uv | 0.11.29 |
| Python | 3.12.13 from `.python-version` |
| PyTorch | 2.2.2 |
| NumPy | 1.26.4 |
| Dependency lock | repository `uv.lock` |

From a clean checkout containing the Stage A changes:

```bash
uv sync --extra torch --extra dev
uv lock --check
uv run pytest -q tests/test_titans_paper_mac_memory.py \
  tests/test_titans_paper_mac_streams.py \
  tests/test_titans_paper_mac_block.py \
  tests/test_titans_paper_mac_benchmark.py \
  tests/test_titans_paper_mac_fidelity.py
uv run pytest -q
uv run seqtrainer-titans-stage-a-benchmark \
  --output-dir artifacts/titans_stage_a
```

The benchmark writes:

- `artifacts/titans_stage_a/stage_a_benchmark.json`;
- `artifacts/titans_stage_a/stage_a_benchmark.md`; and
- `artifacts/titans_stage_a/stage_a_delayed_recall.svg`.

Reproduction seeds:

- default benchmark protocol: global `20260727`; training `20260727`,
  `20260737`, `20260747`; evaluation `20260827`;
- memory references: `7`, `11`, `13`, `17`;
- MAC/mask references: `101`, `103`, `107`, `109`;
- stream fixtures: `123`, `31`, `41`, `51`, `61`, `62`, `71`.

## Binary exit checklist

- [x] Associative KV loss and exact FP64 update have code/test evidence.
- [x] MLP fast weights, surprise, momentum, and adaptive gates have code/test evidence.
- [x] Persistent tokens and full 32-vector `h_t` retrieval have code/test evidence.
- [x] Block-causal masking has exhaustive edge and perturbation tests.
- [x] Stream isolation, reset, serialization, and resume behavior pass tests.
- [x] Full autograd reaches memory gates and MAC parameters.
- [x] The deterministic synthetic acceptance runner beats both controls beyond 32.
- [x] Reproduction environment, commands, seeds, and artifacts are recorded.
- [x] Equation-24 memory writes consume the causal-core output rather than raw segment embeddings.
- [x] The matched adaptive/frozen/no-memory benchmark exercises the paper neural-memory path.
- [x] Stage A is captured in a reproducible commit rather than only an uncommitted worktree.

**Binary result: READY for Stage B.** Preserve the sequential reference, matched
controls, causal perturbation tests, and stream-lifecycle tests as non-regression
gates when adding convolution, scans, fused attention, throughput optimization,
or scaling.
