# Handoff 02 — A100 capacity and horizon selection

Notebook: `notebooks/titans_stage_c/02_stage_c_a100_capacity.ipynb`

## Runtime and protocol

Select an A100 runtime. Edit only the reviewed commit, Drive root, locked
dataset, and run name. The notebook compares horizons 2, 3, and 4 on the same
model, stream sample, batch geometry, and A100. It records reference FP32,
exact-SDPA FP32, and the BF16 activation candidate with an FP32 memory island.

The provisional budget is 35 compute units. Capture starting and ending Colab
balances separately from wall time.

## Gate and returned directory

Every promoted configuration must be finite, show a nonzero gradient through
written state, retain FP32 functional memory, and restore its checkpoint. The
longest stable horizon that improves or matches bounded validation BPB remains
the candidate; this capacity notebook alone does not choose it. Return the
complete printed directory, including checkpoints and failure reasons for any
unavailable matrix entries, the throughput/memory SVGs, and the Colab manifest
and logs. `capacity_validation_bpb.svg` is a matched small validation comparison,
not authorization for the full corpus.
