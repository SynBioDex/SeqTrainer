# Stage B backend interface and telemetry

B1 adds a sibling `seqtrainer.torch.titans_paper_mac_stage_b` package around
the immutable Stage A reference. No Stage A source or legacy `titans_mac`
source is changed.

## Selection contract

`StageBBackendConfig` records memory backend, attention backend, activation
dtype, and an optional approximate window. At B1 the only available tuple is:

```text
memory_backend=reference
attention_backend=multihead_attention
activation_dtype=float32
approximate_window=null
```

The names `exact_accelerated`, `exact_scan`, `approximate_scan`, `sdpa`, and
`flash` are stable configuration values but deliberately unavailable. Selecting
one raises `BackendUnavailableError` with the issue that must prove it. Reduced
precision is likewise unavailable until B6.

`StageBBackendRegistry.execute` spells out the unchanged transition:

1. read all retrievals with `FunctionalNeuralMemory.read_segment`;
2. integrate through `PaperMACBlock.integrate`;
3. write once with `FunctionalNeuralMemory.update_segment`;
4. return `PaperMACBlockOutput`.

Later issues register a separate implementation instead of editing the
reference methods.

## Parity contract

`compare_backends` runs matched blocks and reports, for every tensor:

- exact equality and declared `atol`/`rtol` closeness;
- maximum absolute and relative error;
- cosine similarity;
- sequence and retrieval outputs;
- every fast-weight and surprise tensor;
- every trainable parameter gradient, including persistent tokens, attention,
  neural-memory projections, MLP initialization, and gates.

The B1 reference-to-reference artifact passes with zero absolute/relative error
for all compared tensors.

## Hardware telemetry

`benchmark_stage_b` records warmed per-repetition wall time, median tokens/s,
state payload bytes, seed, segment count, complete model geometry, dtype,
device, platform, torch version, and CUDA peak allocated/reserved bytes. CUDA
fields are explicit `null` values on a CPU execution rather than fabricated
zeros.

Reproduce a small MacBook measurement:

```bash
PYTHONPATH=src .venv/bin/python -m \
  seqtrainer.torch.titans_paper_mac_stage_b.benchmark \
  --d-model 8 --num-heads 2 --memory-depth 1 --segments 1 \
  --warmup-runs 1 --repetitions 3 --device cpu \
  --stem b1_reference_macbook
```

The command writes JSON and Markdown from the same payload under
`artifacts/titans_stage_b/`. The measured B1 host was macOS 15.7.4 on Intel
`x86_64`, PyTorch 2.2.2 CPU. Its `d=8`, one-block, one-segment measurement used
seed 20260727, one warmup and three repetitions; median time was approximately
0.279 seconds (114.7 tokens/s). This is a basal reproducibility measurement,
not an acceleration claim.
