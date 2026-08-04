# B7 long-context synthetic validation protocol

## Scope

This benchmark is a pre-genomic systems validation. It does not load or train
the 15 Gbp corpus and makes no biological or DNA-model quality claim. It tests
whether accepted Stage B mechanisms survive multiple segments, stream-state
serialization/resume, longer context, and the target scale shapes without
silently changing their classifications.

## Fixed matrix

| Scale | Blocks | d_model | Heads | Segments/tokens | Environment |
| --- | ---: | ---: | ---: | ---: | --- |
| debug_d64 | 2 | 64 | 4 | 4 / 128 | 2019 Intel MacBook CPU |
| debug_d128 | 2 | 128 | 4 | 3 / 96 | 2019 Intel MacBook CPU |
| nimble | 4 | 256 | 8 | 2 / 64 | 2019 Intel MacBook CPU |
| a100_pilot | 8 | 384 | 8 | 4 / 128 | Colab Pro A100 only |

Each available scale runs reference, exact-accelerated, approximate windows
2/4/8/16/32, causal convolution, SDPA, frozen-memory, and no-memory. Seed,
dtype, geometry, segment values, backend metadata, and raw timings are stored.

An untrained random memory with sigmoid gate logits near zero starts all three
gates near 0.5 and becomes non-finite over only a few dense segments. That is a
property of an unsafe random initialization, not useful backend evidence. The
stress protocol therefore declares a stable common initialization: gate
projection weights are multiplied by 0.01 and biases set initial
`alpha=1e-4`, `eta=1e-2`, and `theta=1e-3`. The convolutional projection is
copied from the same initialized reference gate. Every segment records a
finite-state flag and first failure context; JSON writing rejects NaN values.

## Multi-segment state protocol

After every segment, each block state is converted to its serializable state
dictionary and restored as fresh differentiable leaf tensors before the next
segment. This provides three things at once:

- an explicit state/resume latency and payload measurement;
- bounded autograd history for long inference-style stress; and
- a value-preserving checkpoint boundary that exercises stream identity,
  segment index, fast weights, surprise, and dtype.

The benchmark reports per-segment latency, end-to-end input tokens/s, Python
allocation peak, CUDA peak when available, state/resume latency, state bytes,
gate statistics, update norms, and fast-weight/surprise drift by context length
against the reference. CPU `tracemalloc` does not count native PyTorch tensor
storage and is labeled accordingly.

Frozen-memory retains incoming fast weights/surprise while incrementing the
segment index. No-memory also substitutes zero retrieval. Both use the same
causal core layout but skip neural-memory write cost.

## Controlled long recall

A separate deterministic linear-memory task writes one held-out key/value pair,
then queries it after 64, 128, 256, and 512 tokens. It reports delay-stratified
accuracy/loss, mean BPB, overwrite, and reset for every backend/control. This
isolates persistence; the random multi-block matrix supplies the dense-update
state-drift evidence. SDPA uses reference memory semantics in this controlled
task because attention is intentionally absent; causal convolution uses its
actual gate adapter.

The future-perturbation matrix additionally requires zero prefix error for every
variant, including controls.

## Hardware honesty and rerun

MacBook and A100 sections are distinct. The A100 pilot runs only when the
attached CUDA device name contains `A100`; otherwise every A100 row is recorded
as unavailable. Results are never extrapolated from CPU.

From the repository root:

```bash
/Users/gonzalovidal/opt/anaconda3/envs/seqtrainer/bin/seqtrainer-titans-stage-b-long-context
```

Artifacts are written to `artifacts/titans_stage_b/` as JSON, Markdown, and SVG.
