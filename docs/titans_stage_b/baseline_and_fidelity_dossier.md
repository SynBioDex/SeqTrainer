# Titans paper-MAC Stage B baseline and fidelity dossier

## Scope and authority

This dossier is the B0 contract for Stage B. It separates statements supported
by *Titans: Learning to Memorize at Test Time* (arXiv:2501.00663v1) from
SeqTrainer engineering decisions. Stage B extends the paper-MAC package without
changing the Stage A reference implementation or the legacy `titans_mac`
package.

Authoritative sources:

- Paper Sections 3.1--3.3 and 4.1, especially equations 11--18 and 21--25.
- Stage B project specification and GitHub epic #30.
- Stage A fidelity audit and executable tests at the pinned commit below.

## Pinned Stage A reference

| Item | Value |
| --- | --- |
| Reference branch | `feat/titans-paper-mac-stage-a` |
| Reference commit | `46dd0158523377ef36cce0edc75743879c109387` |
| Commit subject | `Complete paper MAC Stage A reference` |
| Stage B branch point | exactly the reference commit; no rebase or merge |
| Merge base with local `main` | `5e9701d2db6ab51a498763fbe8469523e155c32f` |
| Divergence from local `main` | 8 main-only commits, 24 Stage-A-only commits |
| Diff at branch point | 97 files, 17,295 insertions, 673 deletions |

The large divergence makes an opportunistic merge or rebase high risk: Stage A
contains both the paper-MAC package and broad repository modernization, while
`main` advanced independently. Stage B therefore develops on its own branch.
Integration must later use a dedicated review that reruns every immutable gate
listed below and resolves unrelated repository changes separately.

## Executed baseline

Environment captured on 2026-07-17 (America/Denver):

- macOS 15.7.4, build 24G517, Intel `x86_64` (2019 MacBook Pro class).
- Python 3.12.13 from the project `.venv`.
- PyTorch 2.2.2 CPU; MPS unavailable on this Intel host.
- Test command: `PYTHONPATH=src .venv/bin/python -m pytest -q`.
- Result: **45 passed**, 7 third-party warnings, 34.35 seconds.
- Mechanism command:
  `PYTHONPATH=src .venv/bin/python -m seqtrainer.torch.titans_paper_mac.benchmark`.
- Result: `gates_passed: true`; JSON, Markdown, and SVG artifacts were emitted
  under `artifacts/titans_stage_a/`.

The explicit `PYTHONPATH=src` is required for the already-created local virtual
environment because it did not contain the repository as an editable install.
Stage B telemetry must record the actual invocation rather than hide this
environment fact.

## Paper-supported mechanisms

### Exact neural-memory recurrence

The paper defines associative keys and values (equation 11), reconstruction
loss (12), adaptive forgetting and momentum (13--14), and inference-style
retrieval (15). Section 3.2 then unrolls chunk updates (16), gives a linear
memory matrix-product example (17), and observes that the momentum recurrence
(18) can use an associative scan once its gradient inputs are available.

For the nonlinear MLP used by SeqTrainer, each surprise gradient depends on the
currently evolving fast weights. The paper says the MLP treatment is analogous
but does not provide a complete exact associative composition for those
evolving nonlinear gradients. Stage B must therefore prove exactness rather
than infer it from the linear example.

### Convolution and gating: what the paper actually says

There are two distinct convolution references, and the paper does not connect
them with a complete implementation recipe:

1. At the end of Section 3.2, the paper says that making `alpha`, `eta`, and
   `theta` constant within a chunk makes the update linear time-invariant and
   permits a global convolution. It immediately says the reported experiments
   instead make these parameters functions of tokens. Thus global convolution
   is a described efficiency simplification, not the reported token-dependent
   experimental path.
2. The Section 5.9 ablation reports a model "w/o Convolution" and shows a
   degradation, but the paper text does not specify that component's placement,
   kernel width, grouping, padding, nonlinearity, or relationship to the three
   adaptive update gates.

Consequently, no single "paper-exact convolution" can be recovered from the
paper text. B2 may implement only a minimal, causal, opt-in interpretation and
must label all choices as repository decisions. It may not silently replace
the Stage A token-wise gate path or claim that the Section 5.9 component is the
Section 3.2 global convolution.

## Repository decisions for Stage B

1. `reference` remains the default backend and directly invokes the unchanged
   Stage A recurrence.
2. `exact_accelerated` may change execution only. It preserves token order,
   evolving-gradient semantics, read-once/write-once timing, and declared
   precision. Unsupported execution falls back to `reference`.
3. `exact_scan` remains unavailable unless B4 proves an associative composition
   for the actual nonlinear update and verifies outputs, state, surprise, and
   gradients.
4. `approximate_scan` is opt-in research behavior. Its window, stale snapshot
   semantics, error, speed, and synthetic-task effect are mandatory metadata.
5. Convolution is opt-in and causal. The smallest admissible interpretation is
   a left-padded depthwise temporal convolution applied before adaptive gate
   projection; kernel size and placement must be exposed and documented.
6. The FP32 memory island includes fast weights, surprise, gates, and state.
   FP64 remains the oracle; reduced precision applies only to model/attention
   activations behind an explicit configuration.
7. Every backend is selected through typed configuration, emits reproducibility
   metadata, and is compared with reference, frozen-memory, and no-memory
   controls where the task is meaningful.

## Code seams

| Stage B concern | Existing seam at the pinned commit | Constraint |
| --- | --- | --- |
| Fast-weight update | `memory.py:199-236`, `FunctionalNeuralMemory.update_segment` | Keep runnable and unchanged |
| One-token recurrence | `memory.py:172-197`, `update_one` | Oracle for state/surprise math |
| Gate production | `memory.py:42-61`, `AdaptiveUpdateGates` | Existing token-wise path remains default |
| Functional MLP gradient | `memory.py:121-152` | Preserve `create_graph=True` and evolving weights |
| Read/write timing | `mac.py:171-183`, `PaperMACBlock.forward` | All reads use `M_(t-1)`; one returned `M_t` |
| Causal layout | `mac.py:28-63`, `block_causal_attention_mask` | `[P,H,S]` edge set is authoritative |
| Core attention | `mac.py:147-169`, `integrate` | Adapter must preserve exact mask semantics |
| Stream lifecycle | `lifecycle.py` and `state.py` | No cross-stream state or anonymous global memory |
| Synthetic controls | `benchmark.py` and `synthetic.py` | Preserve adaptive/frozen/no-memory comparison |

Stage B implementation should add adapters and dispatch beside these seams,
not rewrite them in place.

## Immutable regression gates

The following Stage A evidence is immutable for Stage B:

- `tests/test_titans_paper_mac_memory.py`: FP64 hand update, full 32-step
  differentiability, state replacement, serialization.
- `tests/test_titans_paper_mac_block.py`: exact mask edges, full pre-write read,
  one post-core write, no future-token influence, end-to-end gradients.
- `tests/test_titans_paper_mac_streams.py`: stream isolation, reset/end behavior,
  interleaving, save/resume equivalence.
- `tests/test_titans_paper_mac_benchmark.py`: adaptive/no/frozen controls and
  overwrite/forgetting gates.
- `tests/test_titans_paper_mac_fidelity.py`: documented public contract and
  excluded-feature boundary.
- Full repository test suite and the Stage A deterministic benchmark must pass
  before any Stage B issue is closed.

Each new backend additionally needs declared output, retrieval, fast-weight,
surprise, and trainable-gradient comparison evidence; causal paths require
future-perturbation tests.

## Branch and publication policy

- Stage B work lives on `feat/titans-paper-mac-stage-b`, branched directly from
  the pinned reference.
- Do not modify legacy `src/seqtrainer/torch/titans_mac`.
- Do not change the bodies or public behavior of
  `FunctionalNeuralMemory.update_segment` or `PaperMACBlock`.
- Exact and approximate results must be stored and discussed separately.
- Hardware claims must name device, dtype, geometry, seed, warmup, repetitions,
  and unavailable features.
- Stage C and Stage D remain out of scope.

## B0 acceptance decision

- [x] Exact Stage A reference commit and integration risk recorded.
- [x] Paper-supported acceleration and convolution statements separated from
  repository decisions.
- [x] Convolution ambiguity and the global-gating conflict are explicit.
- [x] Immutable tests and branch/backend policy are named.
- [x] Stage A tests and deterministic mechanism gates pass on the baseline host.

**B0 status: complete. Stage B may proceed to B1.**
