# Stage A adaptive-memory benchmark

`seqtrainer-titans-stage-a-benchmark` is a deterministic synthetic acceptance
runner. It is not a DNA training run and does not support biological
performance claims.

```bash
uv run seqtrainer-titans-stage-a-benchmark --output-dir artifacts/titans_stage_a
```

The command writes three portable artifacts:

- `stage_a_benchmark.json`: machine-readable protocol, per-variant metrics, and gates.
- `stage_a_benchmark.md`: a concise human report.
- `stage_a_delayed_recall.svg`: a dependency-free delayed-recall plot.

## Matched comparison

Every condition uses the same `FunctionalNeuralMemory` and vocabulary readout
(7,383 total parameters, including 820 outer-optimized readout parameters),
AdamW optimizer, learning rate, three deterministic training seeds, token
budget, and held-out evaluation fixture. The only experimental difference is
whether the returned functional fast-memory state is used:

| Variant | State behavior |
| --- | --- |
| `adaptive` | Commits the associative MLP fast-weight update and reads it later. |
| `frozen_memory` | Reads the identical initialized MLP but never commits candidate updates. |
| `no_memory` | Uses the current query representation and ignores memory retrieval. |

Synthetic observations are deterministically encoded as disjoint key and
value subspaces, and controlled K/V/Q projections make the associative
gradient independently interpretable. The update itself is the real Stage A
MLP fast-weight, surprise, momentum, forgetting, and gate path. The shared
readout is trained and evaluated for all three variants; memory meta-parameters
are held fixed in this short correctness probe. Results include cross-entropy
and BPB, train/evaluation gap, gradient norms, actual `alpha`/`eta`/`theta`
statistics, committed update norms, task correctness, and delay-stratified
delayed recall (including the `>32` bucket).

## Executable gates

The runner exits nonzero unless all gates pass:

1. adaptive delayed recall at `>32` exceeds both controls by the configured
   substantial margin (default `0.50`);
2. adaptive overwrite and reset tasks are exactly correct; and
3. compact lifecycle, block-causal mask, and perturbation-based leakage
   reference probes pass.

The default smoke protocol contains eight independent streams and reports the
complete seed and token-budget provenance in the JSON artifact.
