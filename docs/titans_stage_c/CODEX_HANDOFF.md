# Titans paper-MAC Stage C — Codex implementation handoff

## Purpose

This document is the starting context for a new Codex CLI session that will
plan and implement Stage C. It is a draft planning brief, not authorization to
launch the full 15 Gbp training run.

Stage C is defined by the Stage B audit as **genome/clade-separated 15 Gbp
bacterial next-base foundation training**. Stage D covers promoter, CDS, and
phylum adaptation and remains out of scope.

## Starting status

- Work from `feat/titans-paper-mac-stage-b` or its reviewed successor.
- Stage A reference commit:
  `46dd0158523377ef36cce0edc75743879c109387`.
- Successful named-A100 capture commit:
  `68262e4e4c8e6207c208fc5e7e7b31baa2ab191f`.
- The Stage B audit is `READY` with 94 tests passing and 7 known third-party
  warnings.
- The checksum-verified A100 evidence is under
  `artifacts/titans_stage_b/a100/`.
- The selected Stage B target-hardware configuration is:

  ```text
  memory=exact_accelerated
  attention=sdpa
  activation=float32
  gates=token_wise
  ```

- Approximate scan is experimental and ineligible as a default.
- Exact nonlinear associative scan is unproven and unavailable.
- Flash attention rejected the authoritative exact additive mask and must
  remain disabled unless a future implementation proves the identical edge
  set and passes all causality/parity gates.

Before changing code, read:

1. `docs/titans_stage_b/fidelity_performance_audit.md`
2. `docs/titans_stage_b/baseline_and_fidelity_dossier.md`
3. `docs/titans_stage_b/a100_pilot.md`
4. `docs/titans_stage_a/paper_to_code_matrix.md`
5. `docs/titans_stage_a/repository_interface_map.md`
6. `docs/bacteria_titan.md`

Then run:

```bash
/Users/gonzalovidal/opt/anaconda3/envs/seqtrainer/bin/python -m pytest -q
/Users/gonzalovidal/opt/anaconda3/envs/seqtrainer/bin/python -m \
  seqtrainer.torch.titans_paper_mac_stage_b.a100_pilot \
  --output-dir artifacts/titans_stage_b/a100 --verify-only
```

## What Stage B established

Stage B provides a correctness-first paper-MAC block and a minimal multi-block
stack with typed execution adapters. It proved:

- one immutable `M_(t-1)` read for each 32-token segment;
- causal integration over `[persistent, retrieval, sequence]`;
- one post-core memory write per segment;
- differentiable fast-weight, surprise, forgetting, and momentum updates;
- explicit stream-local functional state;
- exact reference and exact-functional output/state/gradient parity;
- mask-preserving functional SDPA;
- an FP32 neural-memory island around optional BF16/FP16 activations;
- long-stream causality, state/resume behavior, and adaptive-memory advantage;
- complete two-segment forward/backward/AdamW execution on a named A100.

The A100 pilot used an 8-block, `d_model=384`, 8-head stack with approximately
9.48 million parameters. End-to-end two-segment measurements were 60.65
tokens/s for reference FP32, 63.54 for exact-memory FP32, and 65.07 for exact
memory plus SDPA FP32. These are backend-selection measurements, not a capacity
estimate for 2,048-token batches or the 15 Gbp corpus.

## Critical integration boundary

The existing `src/seqtrainer/torch/titans_mac/` model and
`scripts/train_titan_mac_dna_lm_colab.py` implement the legacy slot/EMA memory
model. They are preserved baselines, not the Stage C extension target. In
particular, the legacy model uses detached `@torch.no_grad()` memory updates
and does not implement the paper's functional fast-weight recurrence.

Stage C must add a distinct paper-MAC language-model and training path. Do not
silently replace, rename, or mutate the legacy implementation.

The current paper path also is not yet a production language model:

- `StageBMACStack.forward` accepts one `(32, d_model)` embedded segment;
- it does not own token embeddings, an LM head, or next-base loss;
- it does not batch independently named stream states;
- it has no corpus training checkpoint format;
- it does not define truncated backpropagation across long streams;
- it has only a small two-segment outer-training harness.

## Stage C design decisions that must be explicit

### 1. Model contract

Create a separately named paper-MAC causal LM around `StageBMACStack`. Define:

- token embedding, positional treatment, final normalization, and tied/untied
  next-base head;
- conversion of a token stream into ordered 32-token segments;
- tail padding and valid-mask behavior;
- whether persistent tokens live independently in every block, as they do in
  Stage B;
- how logits align with labels across segment boundaries;
- the exact Stage B backend configuration stored in every checkpoint;
- a matched parameter-count baseline, preferably reference paper-MAC plus the
  legacy model as a clearly labeled external baseline.

Do not add a second feed-forward/Transformer stack merely to reproduce the
legacy architecture. The primary experiment is the paper-MAC stack itself.

### 2. Stream and memory lifecycle

Every batch row must have an explicit stream ID and independent state for every
MAC block. The plan must decide:

- whether a stream is a contig, replicon, or complete assembly;
- whether memory resets at contig, plasmid, assembly, batch, epoch, and split
  boundaries;
- how streams are shuffled without shuffling segments within a stream;
- how finished streams are replaced in a batch without leaking state;
- whether validation begins from reset state or a documented prefix;
- which stream states, RNG state, optimizer state, and data cursor are included
  in checkpoints.

A safe initial default is to reset at contig boundaries because arbitrary
contig ordering should not become an undocumented source of context. If Stage
C wants assembly-level memory, it must define and preserve an ordering.

### 3. Gradient-horizon policy

The exact neural-memory update uses higher-order gradients. Retaining the graph
through an entire genome or 2,048-token window may be impractical. Stage B only
proved a two-segment, 64-token outer gradient.

Stage C must define truncated backpropagation independently from state
persistence. A candidate starting policy is:

- retain autograd through 2–4 consecutive 32-token segments;
- optimize the accumulated loss;
- detach the resulting functional state at the boundary while preserving its
  numerical values;
- continue the same logical stream with the detached state;
- record the gradient horizon in configs, checkpoints, metrics, and artifacts.

This is a fidelity/optimization decision, not a hidden implementation detail.
The plan should compare at least two horizons on a small pilot.

### 4. Dataset and leakage contract

The current bacterial pipeline provides deterministic accession-level 90/5/5
splits and `uint8` token shards, but Stage C is described as genome/clade
separated. Accession separation alone does not guarantee clade separation.

The C0 specification must choose a grouping boundary such as ANI cluster,
species, genus, or family and prove that no group crosses train/validation/test.
For an E. coli-related corpus, an ANI/species-cluster holdout may preserve a
usable target distribution better than a complete genus holdout, but the
scientific question must drive the choice.

The new dataset interface must also expose enough metadata to reconstruct
ordered streams:

- accession and contig/replicon identifier;
- segment offset and valid length;
- split and clade group;
- an explicit start/end-of-stream marker;
- source FASTA and manifest checksum provenance.

Do not carry memory across unrelated accessions or across train/evaluation
splits. Do not fall back to random token-window splits.

### 5. Precision and attention policy

Start correctness work with the selected FP32 configuration. BF16/FP16 Stage B
runs demonstrated behavioral execution with an FP32 memory island, not
numerical parity. A Stage C capacity pilot may promote BF16 activations only
after it passes:

- finite-loss and finite-gradient checks;
- state dtype checks;
- short-run learning-curve comparison;
- held-out BPB tolerance;
- checkpoint/resume and causal tests.

Flash remains disabled. Differentiable training currently uses a portable math
SDPA path because the Colab PyTorch build selected efficient kernels whose
backward implementation was unavailable.

### 6. Capacity and cost gate

Do not extrapolate the unbatched 65 tokens/s Stage B result into a training
schedule. Before authorizing the full corpus, benchmark the actual LM with:

- context represented as 64 ordered 32-token segments for a nominal 2,048
  token example;
- at least two batch sizes and gradient-accumulation values;
- at least two gradient horizons;
- FP32 and any candidate BF16 activation mode;
- optimizer step time, tokens/s, allocated/reserved peak memory, state bytes,
  checkpoint size, and save/resume time;
- an estimated A100-hours and storage budget for one corpus pass.

The existing script's default of ten epochs over 15 Gbp must not be inherited
without a budget review. Prefer an explicit token budget and number of corpus
passes over an ambiguous epoch count.

### 7. Evaluation contract

At minimum, report:

- next-base loss, perplexity, and bits per base;
- token and top-2 accuracy;
- GC-stratified losses;
- per-clade and per-accession metrics;
- adaptive/reference, frozen-memory, and no-memory ablations;
- exact-memory plus SDPA versus reference paper-MAC at matched initialization;
- memory update norms, surprise norms, gate distributions, and state drift;
- throughput, memory, and checkpoint telemetry;
- uninterrupted versus resumed learning equivalence under a documented
  tolerance.

Synthetic recall remains a regression gate, not evidence of biological
quality. Stage C's final decision must use held-out genomic results.

## Provisional Stage C work packages

The next agent should turn these into an evidence-gated plan rather than treat
them as preapproved implementation tickets:

| Package | Draft outcome |
| --- | --- |
| C0 | Fidelity, data-split, stream-lifecycle, gradient-horizon, compute-budget, and acceptance contract. |
| C1 | Paper-MAC DNA causal-LM wrapper with 32-token segmentation and next-base loss. |
| C2 | Ordered, clade-grouped stream dataset and immutable manifest/checksum evidence. |
| C3 | Stream-aware trainer, truncated-gradient policy, checkpoint/resume, and telemetry. |
| C4 | CPU/small-CUDA correctness pilot covering causality, leakage, reset, resume, and learning smoke tests. |
| C5 | Representative A100 capacity matrix and FP32/BF16/default decision. |
| C6 | Bounded genomic pilot with reference/frozen/no-memory comparisons and a full-run cost decision. |
| C7 | Resumable 15 Gbp foundation-training run, only after C0–C6 gates pass. |
| C8 | Held-out genomic evaluation, reproducibility package, and binary Stage D gate. |

## Suggested acceptance gates before the 15 Gbp run

- All Stage A and Stage B tests and audits remain green.
- No clade group, accession, or stream crosses splits.
- Every tail/padding position is excluded from loss and memory updates.
- Future-token perturbations do not alter earlier logits.
- Interleaved batched streams equal isolated execution within the declared
  precision tolerance.
- Reset/end/replacement behavior cannot leak state between streams.
- Checkpoint resume restores model, optimizer, RNG, data cursor, stream IDs,
  and functional states.
- The selected gradient horizon has finite, nonzero gradients through written
  memory state.
- A representative A100 pilot supplies a credible token budget, runtime, and
  storage estimate.
- A bounded genomic pilot learns beyond frozen/no-memory controls before the
  expensive run is authorized.

## Questions for the project owner

The new planning session should obtain or explicitly default these answers:

1. What is the maximum A100-hours and Drive-storage budget?
2. Is 15 Gbp the unique-bases corpus size, the training token budget, or both?
3. How many corpus passes are intended?
4. Which clade boundary defines held-out generalization?
5. Should memory persist across contigs from the same assembly?
6. Is the legacy slot/EMA model required as a matched baseline?
7. What smallest genomic pilot is acceptable before full-run authorization?
8. Which result authorizes Stage D: held-out BPB, memory-ablation advantage,
   downstream transfer, or a combination?

## Recommended first prompt for a new Codex CLI session

```text
Read docs/titans_stage_c/CODEX_HANDOFF.md and the Stage B READY audit. Inspect
the current paper-MAC stack, bacterial dataset pipeline, and legacy training
script without modifying the legacy model. Draft an evidence-gated Stage C C0
specification and implementation plan for genome/clade-separated bacterial
next-base foundation training. Resolve or clearly surface the model wrapper,
ordered stream dataset, clade leakage boundary, memory reset lifecycle,
truncated-gradient horizon, A100 capacity gate, bounded genomic pilot, and
15 Gbp authorization criteria. Do not start full-corpus training or Stage D.
```
