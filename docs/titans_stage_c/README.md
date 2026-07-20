# Titans paper-MAC Stage C

Stage C is the isolated causal bacterial sequence-modeling path built on the
reviewed Stage B paper-MAC stack. The legacy `titans_mac` slot/EMA model and
training script remain preserved baselines.

## Locked contracts

- A MAC segment is 32 tokens; reports also record its exact raw-base span.
- Dense *E. coli* genomes split by deterministic 99% ANI single-linkage group;
  other taxa split by GTDB species cluster. Groups, accessions, and contigs may
  not cross train/validation/test.
- A contig or replicon is one stream. Memory resets at every stream, split, and
  replacement boundary.
- The correctness configuration is exact accelerated memory, exact-mask SDPA,
  token-wise gates, and FP32. Flash and approximate scans are ineligible.
- Horizons 2, 3, and 4 are compared; horizon 1 is diagnostic. State persists
  numerically across a complete stream and is detached only after each horizon.
- The bounded biological gate is at least 0.01 held-out BPB improvement over
  both separately trained frozen-memory and no-memory controls.

## Workflow

1. Run the tokenizer and local CPU gate with
   `seqtrainer-titans-stage-c-tokenizers` and
   `seqtrainer-titans-stage-c-cpu-pilot`. The latter runs both equal-step and
   equal-raw-base regimes and emits the frozen `tokenizer_selection.json`.
2. Generate skani extended pair evidence, build hybrid clade splits, and run
   `scripts/build_bacteria_titan_stage_c_streams.py --tokenizer auto
   --tokenizer-selection ...` with that locked artifact. Handoff 00b provides
   the thin Drive-backed notebook for this step.
3. Use the T4 horizon-3 notebook, then the same-hardware A100 horizon matrix.
4. Run the four bounded-pilot conditions with explicit valid-base budgets.
5. Run `seqtrainer-titans-stage-c-evaluate` on validation. Touch test only
   after the configuration is frozen.

Production datasets store memory-mapped token and token-base-length arrays with
a contig-level JSONL index. Segments are generated lazily, avoiding hundreds
of millions of materialized 32-token rows for a 15 Gbp corpus.

## Compute budget

The Colab cap is 200 units: 20 for T4, 35 for A100 capacity, 105 for the bounded
pilot, and 40 held in reserve. Stop exploratory work after 160 units. The first
full-corpus proposal is at most one 15 Gbp pass and requires a separate review
of observed learning curves and cost.

Notebook-specific operating instructions are under `docs/titans_stage_c/handoffs/`.
Every substantive notebook command runs through the Drive-backed Colab wrapper.
It continuously writes `logs/*.log`, records commit/dirty/hardware metadata in
`colab_run_manifest.json`, and leaves `FAILED.txt` when a command fails.
