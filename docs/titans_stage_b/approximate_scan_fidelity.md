# B5 approximate-scan fidelity and decision

## Classification

`MemoryBackend.APPROXIMATE_SCAN` is an opt-in scientific ablation. It is not an
exact scan and is never described as parity-equivalent. The default remains the
unchanged Stage A reference. `approximate_window` is mandatory and limited to
`2, 4, 8, 16, 32`; configuration, runtime metadata, JSON, Markdown, and plot all
name it explicitly.

## Exact stale-window semantics

For a window beginning with fast weights `M_w`:

1. Project all 32 integrated segment positions to keys, values, and token-wise
   `alpha`, `eta`, and `theta` gates exactly as the reference does.
2. For every valid position in `[w, w+window)`, evaluate its associative loss
   gradient using the same incoming snapshot `M_w`.
3. Consume those gradients in original token order. Surprise/momentum and
   forgetting remain sequential and use the position's actual gates.
4. Snapshot the resulting fast weights only at the next window boundary.
5. Publish exactly one replacement state after the complete segment.

Invalid tail-padding positions are skipped. Window boundaries still follow
physical segment positions. Stream ID, reset/end lifecycle, incoming-state
immutability, and read-before-write timing are unchanged.

The approximation is the stale gradient. Even though the subsequent recurrence
is sequential, a nonlinear MLP's exact gradient normally depends on every
immediately preceding fast-weight update. B4 demonstrated that replacing those
evolving gradients is not associative or exact.

## Study design

The B5 artifact combines two deliberately separate comparisons:

- A dense 32-write random MAC segment exposes state, surprise, and outer-gradient
  drift for every window. It reports relative error, cosine, gate statistics,
  update norms, tokens/s, raw timing samples, and peak-memory telemetry against
  both reference and the B3 tensor-exact backend.
- The unchanged Stage A delayed-recall/overwrite/reset protocol compares
  reference, exact-accelerated, every approximate window, frozen memory, and no
  memory on loss/BPB and task correctness.

The Stage A fixture has at most one valid memory write in each segment. That is
excellent for isolating delayed cross-segment memory, but it means within-window
staleness can be inactive. Equal task scores on that fixture therefore do not
cancel dense-update state or gradient error. B7 must use longer/dense streams.

On CPU, `tracemalloc` reports Python allocation peaks rather than native PyTorch
tensor storage; the artifact labels that limitation. CUDA runs additionally
record `max_memory_allocated` when available.

## Decision rule

Approximate scan remains `experimental_only` regardless of incidental local
timing unless a later task shows a meaningful end-to-end benefit at an accepted
error window. No result may promote it to the default or call it equivalent to
the exact recurrence. The study's JSON and Markdown compute and store the
fastest observed window, but the decision is based on both speed and fidelity.

Artifacts:

- `artifacts/titans_stage_b/b5_approximate_scan_study.json`
- `artifacts/titans_stage_b/b5_approximate_scan_study.md`
- `artifacts/titans_stage_b/b5_approximate_scan_speed_fidelity.svg`

