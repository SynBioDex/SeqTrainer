# B3 exact recurrent acceleration

## Exactness class

`exact_accelerated` is a **tensor-exact functional-loop refactor** of the Stage
A memory write. It is not an associative scan and does not precompute or stale
any gradient.

The reference creates a complete replacement `PaperMACStreamState` during each
of the 32 inner updates. `ExactAcceleratedMemoryBackend` instead carries the
ordered `fast_weights` and `surprise` mappings through the same loop, calls the
same projection, gate, `surprise_gradient`, `momentum_update`, and
`forgetting_update` operations in the same order, and creates the public state
once at the end.

Preserved invariants:

- all retrievals still use the incoming `M_(t-1)`;
- the core output, not raw input, is written once;
- token order and evolving `u_i(M_(i-1))` gradients are unchanged;
- no detach, stale snapshot, mixed precision, compilation, or scan is used;
- valid-tail masking and segment index behavior match reference;
- the FP32 memory island remains FP32.

## Parity evidence

Fixed-seed FP64 and FP32 tests compare reference and `exact_accelerated` for:

- sequence and retrieval outputs;
- every final fast-weight and surprise tensor;
- segment input gradient;
- every trainable parameter gradient, including attention, persistent tokens,
  projections, gates, and MLP initialization.

All are tensor-exact (`atol=0`, `rtol=0`) for the tested one- and two-layer
memory configurations. A two-block stack test is tensor-exact as well.

## Reference fallback

The backend uses a conservative support predicate. Unsupported memory types or
geometries call the unchanged `FunctionalNeuralMemory.update_segment` and emit
`last_execution=reference_fallback`. The fallback has a dedicated tensor-exact
test. Ordinary validation errors are not swallowed.

## MacBook scale matrix

Artifact files:

- `artifacts/titans_stage_b/b3_exact_acceleration_matrix.json`
- `artifacts/titans_stage_b/b3_exact_acceleration_matrix.md`

The measurement host was macOS 15.7.4, Intel `x86_64`, PyTorch 2.2.2 CPU,
FP32. Seed 20260727 (nimble seed 20260728), one 32-token segment, one warmup,
and one timed repetition were used. These are basal observations, not stable
throughput claims:

| Scale | Geometry | Parameters | Reference | Exact functional loop | Observed ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| debug | 2 blocks, d=64, 4 heads, depth-1 memory | 67,334 | 1.0393 s | 0.8436 s | 1.232x |
| nimble | 4 blocks, d=256, 8 heads, depth-1 memory | 2,111,500 | 1.7726 s | 2.1602 s | 0.821x |
| A100 pilot | 8 blocks, d=384, 8 heads | not instantiated | unavailable | unavailable | unavailable |

The debug run observed less Python object overhead. The nimble run regressed;
there is no general CPU acceleration claim. The named Colab Pro A100 was not
attached to this execution, so the artifact records it as unavailable rather
than substituting CPU or making a GPU claim. B7 must repeat with multiple
repetitions on the named hardware before selecting a performance backend.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python -c '
from seqtrainer.torch.titans_paper_mac_stage_b import (
    run_exact_acceleration_matrix,
    write_exact_acceleration_matrix,
)
matrix = run_exact_acceleration_matrix(warmup_runs=1, repetitions=1, device="cpu")
write_exact_acceleration_matrix(matrix, "artifacts/titans_stage_b")
'
```

The reference backend remains the default because exactness alone does not
establish a consistent performance case.
