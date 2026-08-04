# Titans paper MAC repository interface map (Stage A)

## Boundary: preserve the legacy baseline

`src/seqtrainer/torch/titans_mac/` is the existing **legacy slot/EMA Titan
MAC baseline** and is out of scope for this epic.  In particular, do not alter:

- `TitansMACLMConfig` in `configuration.py`;
- `_LongTermMemory` and `TitansMACForCausalLM` in `model.py`;
- its `memory_slots`, `memory_state`, `retention_gate`, top-k retrieval, or
  `@torch.no_grad()` EMA update;
- its checkpoint format, generation helper, tests, or the
  `train_titan_mac_dna_lm_colab.py` script.

The educational `src/seqtrainer/torch/titans.py` path is also a separate
MIRAS-inspired classifier.  Neither baseline is an extension target for paper
state or equations: reusing a name or importing a legacy memory class would
silently inherit detached, shared state and violate Stage A.

## New extension seam

Create a sibling package, `src/seqtrainer/torch/titans_paper_mac/`, whose
public names are distinct from the legacy API:

| New file | Public / internal interface | Responsibility |
| --- | --- | --- |
| `configuration.py` | `TitansPaperMACConfig` | Validates `segment_length == 32`, model widths, persistent-token count, gate parameterization, and state format version. |
| `state.py` | `PaperMACStreamState`, `PaperMACStateDict` | Immutable functional fast-weight/surprise state; cloning, reset, CPU serialization, and device restoration. |
| `memory.py` | `FunctionalNeuralMemory`, `AdaptiveUpdateGates` | MLP fast-weight functional call; associative key/value loss; 32-step differentiable surprise/forgetting update; read-only retrieval. |
| `model.py` | `TitansPaperMACForCausalLM`, `_paper_mac_causal_mask` | Pre-segment retrieval, `[persistent, h, segment]` causal core, post-segment update, next-base head. |
| `streaming.py` | `PaperMACStreamStore`, `PaperMACStreamRunner` | Maps explicit stream IDs to states; enforces one write per segment and isolation across interleaved requests. |
| `checkpoints.py` | `save_paper_mac_checkpoint`, `load_paper_mac_checkpoint` | Versioned model/optimizer/RNG plus keyed stream states; must not reuse legacy checkpoint assumptions. |
| `__init__.py` | Limited explicit exports | Keeps `titans_paper_mac` opt-in and prevents a name collision with `titans_mac`. |

Only after the isolated package works should
`src/seqtrainer/torch/__init__.py` receive optional imports for its public
names.  That export is a later, additive integration step; it must not replace
the `titans_mac` exports.

## Existing integration points and why they are not reused directly

| Existing file / class | Current behavior | Stage A action |
| --- | --- | --- |
| `src/seqtrainer/torch/titans_mac/model.py::_LongTermMemory` | Learned slot bank plus detached EMA `memory_state`; one first-token query and top-k context. | Preserve unchanged; document as the negative compatibility boundary. |
| `src/seqtrainer/torch/titans_mac/model.py::TitansMACForCausalLM._causal_mask` | Gives every DNA position the same retrieved context prefix. | Do not extend: paper MAC needs `h_1:i` block causality. Add `_paper_mac_causal_mask` in the new package. |
| `src/seqtrainer/torch/titans_mac/checkpoints.py` | Serializes a normal module state dict, optimizer, scheduler, and RNG. | Use as a serialization-shape reference only. New code must add keyed functional stream states and a format version. |
| `tests/test_titans_mac_lm.py` | Tests legacy model, conventional causality, and checkpoint loading. | Leave untouched. Add paper-specific tests instead of broadening legacy assertions. |
| `src/seqtrainer/data/bacteria_titan/token_shards.py::TokenShardDataset` | Serves independent next-token windows; no source/stream ID is returned. | Do not change in A0. A2 should add a distinct synthetic stream dataset/adapter returning ordered segments and stream IDs. |
| `scripts/train_titan_mac_dna_lm_colab.py` | Builds the legacy config, shuffles training windows, resets a single model buffer once per epoch. | Preserve unchanged. A4 may add a separate `train_titans_paper_mac_*.py` runner after stream lifecycle semantics exist. |

## Test placement by child issue

| Test file | Child issue | Required proof |
| --- | --- | --- |
| `tests/test_titans_paper_mac_memory.py` | A1 | Associative loss; `M`, surprise, and adaptive gates; FP64 hand reference; full differentiation through 32 updates. |
| `tests/test_titans_paper_mac_streams.py` | A2 | Explicit stream IDs, reset, interleaving isolation, segment boundary behavior, and save/resume equivalence. |
| `tests/test_titans_paper_mac_causal.py` | A3 | `h_t` length 32; a pre-segment snapshot read; one post-segment write; persistent prefix; block-causal no-future leakage. |
| `tests/test_titans_paper_mac_benchmarks.py` | A4 | Adaptive vs no-memory vs frozen-memory controls past 32 tokens, plus overwrite/forgetting tasks. |
| `tests/test_titans_paper_mac_fidelity.py` | A5 | Traceability audit against this matrix, public API, checkpoint format, and documented deviations. |

## Required invariants for every future patch

1. A paper-MAC segment is always 32 positions before tail masking.
2. `M_(t-1)` is a single read-only snapshot for every retrieval in that
   segment; the new state is committed once after the core outputs exist.
3. `h_t` has one retrieved vector per segment position, hence length 32.
4. The causal mask allows position `i` to see only persistent tokens,
   `h_1:i`, and segment tokens `S_1:i`.
5. State belongs to explicit stream IDs and is preserved in checkpoints;
   execution of one stream must never affect another.
6. New paper code must never import the legacy private `_LongTermMemory` or
   mutate `titans_mac` buffers.
