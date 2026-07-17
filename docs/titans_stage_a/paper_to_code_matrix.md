# Titans paper-to-code matrix (Stage A)

This is an implementation plan, not an implementation of the paper model.  It
maps [Behrouz, Zhong, and Mirrokni, *Titans: Learning to Memorize at Test
Time* (arXiv:2501.00663v1)](https://arxiv.org/pdf/2501.00663) to a new,
isolated `titans_paper_mac` package.  Equation numbers below refer to that
paper.  The locked Stage A contract overrides a few paper-level choices where
it is more specific: segment length `C = 32`, `h_t` has 32 vectors, the entire
segment reads the same `M_(t-1)`, and the segment writes exactly once after the
causal core pass.

## Locked execution order

For each independently identified stream `r` and segment `t`, with
`S_t.shape == (B, 32, d_model)`:

1. Read the immutable snapshot `M_(t-1)` once for all 32 segment positions.
2. Produce `q_t = S_t W_Q` and `h_t = M_(t-1)^*(q_t)`, so
   `h_t.shape == (B, 32, d_model)`.
3. Run causal MAC on `[persistent_tokens, h_t, S_t]`.  At position `i`, the
   usable keys/values are persistent tokens, `h_t[:, :i+1]`, and `S_t[:, :i+1]`.
   **Future-token leakage is forbidden:** neither `S_t[:, i+1:]` nor the
   corresponding future `h_t` may affect output `i`.
4. Form associative keys and values from causal core outputs and run the 32
   functional inner recurrence steps.  Intermediate candidate states are never
   published for retrieval; commit only the final `M_t` and `surprise_t` once.
5. Persist the resulting state only under the owning stream identifier.  A
   reset replaces that stream's state; it does not clear another stream.

The symbol `S_t` is overloaded in the paper for a sequence segment and for
surprise.  This plan uses `segment_t` for the former and `surprise_t` for the
latter in code and tests.

## Matrix

| Locked mechanism / paper equation | B-stream and `C=32` shape | Proposed module, function, or state field | Test evidence required | Fidelity ambiguity / recorded decision |
| --- | --- | --- | --- | --- |
| Neural memory `M_t`; the paper uses an MLP long-term memory (§3.1). | `B` logical streams are represented as `B` independent states; each state is a functional collection of MLP fast-weight tensors (no shared mutable batch buffer). | `titans_paper_mac/state.py`: frozen `PaperMACStreamState(memory_params, surprise, segment_index)`; `memory.py`: `FunctionalNeuralMemory`. | FP64 one-step update equals a hand calculation; reset produces initial params without mutating the initial parameter module. | The paper denotes `M_t` as a neural function/weights rather than a single tensor. Store the complete MLP parameter pytree, not a slot matrix. |
| Associative key/value objective (11–12): `k=xW_K`, `v=xW_V`, `L=1/2 ||M_(t-1)(k)-v||^2`. | `x`, `k`, and `v`: `(B, 32, d_model)`; scalar loss may retain `(B, 32)` before reduction. | `FunctionalNeuralMemory.associative_loss(memory_params, keys, values)` and trainable `key_projection`, `value_projection`. | Hand-reference gradient test and an input/key/value shape test. | The paper fixes the outer-loop projection parameters while updating memory. Stage A keeps them ordinary model parameters but makes the inner update functional. |
| Momentary and past surprise (9–10); momentum state `S_t^surprise`. | `surprise` has exactly the same MLP pytree structure as one stream's fast weights; it is calculated across all 32 positions before the stream state is replaced. | `PaperMACStreamState.surprise`; `FunctionalNeuralMemory.segment_update`. | Fixed-seed FP64 test proves `surprise_t = eta_t * surprise_(t-1) - theta_t * grad` for a one-position reference and checks all 32 positions participate. | The paper's notation uses `S_t` both for a segment and surprise. Code must not reuse the same name. |
| Adaptive forgetting (13–14): `M_t=(1-alpha_t)M_(t-1)+surprise_t`. | `alpha_t`, `eta_t`, `theta_t`: `(B, 32, 1)` (or broadcastable to every fast-weight tensor); apply them to the local candidate state in 32 sequential steps, then commit one final segment state. | `AdaptiveUpdateGates` in `memory.py`; state remains `memory_params` plus `surprise`. | Gate-range test; exact FP64 update test; overwrite/forgetting task. | The paper makes all gates token-dependent. Stage A must expose token-dependent values; a constant or a silent mean-over-segment gate would be a fidelity deviation. |
| Full differentiation through the update sequence. | Autograd graph spans all 32 sequential inner updates; no `torch.no_grad()`, `.data`, `detach()`, or in-place mutation in the paper path. | Functional `torch.func.functional_call` / `autograd.grad(create_graph=True)` implementation inside `segment_update`. | A meta-gradient from post-update loss reaches memory initializer and gate/projection parameters; test fails if a detach is inserted. | Paper discusses chunk-parallel evaluation; Stage A is correctness-first and uses the sequential 32-step reference before scan/convolution work. |
| Retrieval (15) and MAC read (21): `h_t=M_(t-1)^*(S_t W_Q)`. | One retrieved vector per segment position: `h_t.shape == (B, 32, d_model)`.  All reads use the same `M_(t-1)` snapshot. | `FunctionalNeuralMemory.retrieve(memory_params, queries)` called once before core attention. | Assert length 32; perturbing the update-only path cannot change `h_t`; snapshot/read-once test. | Equation (21) gives a segment query matrix, consistent with 32 retrieved vectors. The Stage A contract fixes this rather than using top-k slots. |
| Segment boundary and MAC write (21–25). | A segment is `(B, 32, d_model)` even for a tail: pad/valid-mask it, preserve `valid_length`, and perform no write for padding. | `TitansPaperMACForCausalLM.forward_segment`; a stream runner splits token streams into 32-token segments. | Boundary-reset task, tail-mask test, and sequential vs save/resume equivalence. | Equation (24) says the segment output updates the memory. Stage A locks one write *after* the segment; it deliberately does not write after each token. |
| Persistent memory (19, 22). | `persistent_memory`: `(num_persistent_tokens, d_model)`, expanded to `(B, P, d_model)`; it is input-independent and outside stream state. | `TitansPaperMACForCausalLM.persistent_memory`. | Prefix shape and causal-mask tests; optimizer/stream update test proves it is not replaced by stream-state writes. | Paper says persistent parameters are fixed at test time. Stage A treats them as outer-loop trainable and never changes them in the inner stream update. |
| MAC causal integration (22–23 and Figure 3a). | Attention sequence length `P + 32 + 32`: `[persistent, h, segment]`.  Position `i` sees `P + (i+1) + (i+1)` entries. | `model.py`: `_paper_mac_causal_mask(segment_length=32, persistent_tokens=P)`. | No-future-token leakage: changing `S[:, i+1:]` must not change logits through `i`; independently mutate future `h` and assert the same. | The paper's Figure 3a is graphical; the Stage A mask is explicitly block-causal for both the `h` and token blocks. This is a locked safety interpretation. |
| State serialization and resumability. | Serialize a mapping `stream_id -> PaperMACStreamState`, including all fast weights, surprise tensors, initial state metadata, and segment count.  Do not serialize a mixed batch as anonymous global state. | `checkpoints.py`: `paper_mac_stream_states`; `state.py`: `to_state_dict` / `from_state_dict`. | Midstream save/resume must exactly match uninterrupted execution under fixed seed/FP64. | Existing legacy checkpoints only preserve module buffers. Paper does not prescribe a file format; Stage A defines one versioned payload. |
| Stream isolation. | For `B` independently named streams, state is logically `dict[StreamId, PaperMACStreamState]`; no mean/reduction across stream IDs. | `streaming.py`: `PaperMACStreamStore`, `step(stream_ids, segment)`, `reset(stream_ids)`. | Interleaving two streams equals processing each alone; reset A leaves B bitwise unchanged. | The paper writes a sequence recurrence, not a multi-request server protocol. Isolation is an application correctness requirement and must be explicit. |

## Evidence in the current repository

The only current paper-adjacent implementation is deliberately a different
baseline:

- `src/seqtrainer/torch/titans_mac/model.py::_LongTermMemory` retrieves top-k
  vectors from learned `memory_slots`, uses the first input token as a
  window-global query, and updates a `memory_state` buffer using a detached,
  `@torch.no_grad()` EMA.  It has no associative-loss gradient, momentum
  surprise, adaptive `alpha`/`eta`/`theta`, functional fast weights, or
  stream-keyed state.
- `src/seqtrainer/torch/titans.py::NeuralLongTermMemory` is an educational
  classifier prototype with the same EMA-style update; it is not a paper
  reproduction.
- `tests/test_titans_mac_lm.py` proves legacy causal logits and a conventional
  module checkpoint round trip.  It does not prove the paper update or
  stateful streaming requirements.
- `src/seqtrainer/data/bacteria_titan/token_shards.py::TokenShardDataset` and
  `scripts/train_titan_mac_dna_lm_colab.py` shuffle independent windows.
  They do not retain a stream identity or segment-continuation order, so they
  cannot drive stateful paper-MAC training unchanged.

## Unresolved questions to settle before Stage A implementation

1. **Segment-local update chronology:** the paper gives token recurrences and
   segment MAC equations separately.  Stage A resolves this by keeping the
   read/retrieval snapshot fixed for the whole segment while executing the 32
   local functional update steps after the core pass, then committing only the
   final state.  The test names and public docstrings must preserve this
   distinction.
2. **Tail policy:** whether a short final segment is padded and masked or
   dropped must be a public streaming contract.  Padding must never generate
   associative updates.
3. **Outer-loop loss:** the paper's `o_t = y_t ⊗ M_t^*(y_t)` (25) leaves the
   particular nonlinear fusion open.  Stage A needs a minimal, documented
   fusion choice for causal next-base logits.
4. **Batching:** logical stream isolation requires a stream ID for every batch
   row.  Shuffling plain token windows is incompatible with carrying state.
5. **State dtype/device:** FP64 is required for the hand-reference test; the
   runtime policy for mixed precision and serialized cross-device state must
   be specified before a training runner is added.
