# Stage A synthetic streams and lifecycle harness

`seqtrainer.torch.titans_paper_mac.synthetic` provides deterministic fixtures
for the Stage A correctness harness.  They are deliberately token-level
fixtures, not DNA training data and not a MAC-attention implementation.

## Fixed vocabulary and reproducibility

Every fixture consists of 32-token segments and uses this fixed vocabulary:

| Token range | Meaning |
| --- | --- |
| `0` | Padding (unused by the generated fixtures) |
| `1` | Query marker; the next token is its key |
| `2` | `no_memory`, the expected answer immediately after a reset |
| `3..10` | Keys |
| `11..18` | Values |
| `19` | Filler |

The default seed is `20260717`; each fixture builder accepts `seed` and
`num_streams`. `build_stage_a_fixtures()` uses consecutive seeds for delayed
recall, overwrite, and boundary-reset fixtures.  Repeating the same arguments
produces equal immutable dataclass values.  Each fixture gives active streams
distinct key/value assignments (up to eight streams), so accidental ownership
sharing is observable rather than hidden by duplicate examples.

## Tasks and scoring

- **Delayed key-value recall:** segment 0 writes `[key, value]`, segment 1 is
  filler, and segment 2 asks `[query, key]`. The query is at least 63 tokens
  after the written value. Its target is absent from its 32-token query segment.
- **Overwrite/forgetting:** segments 0 and 1 write the same key with distinct
  old and new values. Segment 2 must retrieve the new value; the old and new
  values are both absent from its query segment.
- **Context-boundary reset:** segment 0 writes a key/value pair. Segment 1 has
  `reset=True`, which the lifecycle harness applies before its query; its target
  is `no_memory`, not the value from the previous context.

`score_query_predictions(fixture, predictions)` scores exact matches keyed by
`(stream_id, segment_index)` and reports correct, total, missing, and accuracy.

## Ownership, order, and resume

Each `SyntheticSegment` carries `stream_id`, `segment_index`, `reset`, and
`end_of_stream`. `StreamLifecycleHarness` owns a separate state object for
each stream ID, rejects out-of-order segments, resets only the named stream
before a reset segment, and marks it ended after an end segment. It serializes
both state values and ordering/end bookkeeping so resumed execution has the
same transitions as uninterrupted execution.

`stream_level_shuffle(seed)` shuffles complete stream blocks only. It never
shuffles individual segments. `round_robin_interleave()` is available to build
mixed-stream batches while retaining each individual stream's segment order;
this makes state leaks and incorrect global-state ownership visible in tests.
