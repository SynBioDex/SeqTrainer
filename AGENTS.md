# SeqTrainer agent instructions

All coding agents working in this repository MUST follow this file. Nested
`AGENTS.md` files add rules for their directory. They cannot relax this file.
System, developer, and user instructions take precedence.

## Working rules

- Treat the checked-out source, tests, `pyproject.toml`, and `uv.lock` as the
  implementation contract. Documentation supplies context, not proof that a
  feature exists.
- Read the affected source, direct tests, and applicable research protocol
  before editing. Keep changes narrow and avoid unrelated refactors.
- Use concise English prose. Keep comments only for non-obvious constraints,
  deliberate deviations, or cross-file contracts. Do not narrate the code or
  the change.
- Do not edit `artifacts/`, `outputs/`, published manifests, checkpoints,
  notebook outputs, or reported results unless the request explicitly includes
  regeneration and review.
- Do not push, force-push, merge, or alter commit history without explicit
  user permission. Keep one logical change per commit when commits are asked
  for.

## Repository structure

```text
src/seqtrainer/                         Package source
  data/bacteria_titan/                  Dataset and manifest materialisation
  torch/titans_mac/                     Legacy slot/EMA baseline
  torch/titans_paper_mac*/              Separate paper-MAC research paths
docs/titans_stage_{a,b,c}/              Stage protocols and evidence
notebooks/tutorials/                    Public tutorial notebooks
tests/                                  Regression contracts
```

- Keep framework-neutral behavior in `data`, `transforms`, `models`, `clients`,
  `sparql`, and `applications`.
- Keep PyTorch and Keras optional. Preserve lazy imports so a base install
  remains usable.
- Preserve legacy `titans_mac` and paper-MAC paths as separately named
  baselines. Never replace one with the other silently.
- This branch has no `.github/workflows/` directory. Do not claim CI exists or
  add CI configuration unless the task requires it.

## Environment and validation

The repository pins Python 3.12 in `.python-version`. Package metadata supports
Python 3.9 and later. Prefer the locked development environment:

```bash
uv sync --extra torch --extra bacteria-titan --extra dev
uv lock --check
uv run pytest -q tests/test_bacteria_titan_dataset.py
uv run pytest -q tests/test_titans_paper_mac_stage_c_model.py \
  tests/test_titans_paper_mac_stage_c_workflows.py
uv run ruff check src tests
```

- Start with the smallest relevant tests, then broaden coverage for shared
  contracts.
- Do not launch data downloads, full-corpus training, or GPU capacity studies
  merely to validate an edit.
- Use `seqtrainer --help` and `seqtrainer-titans-stage-* --help` before adding
  commands to documentation. Do not invent commands from a roadmap.
- `torch`, `tensorflow`, `torch-geometric`, and bacterial Titan tooling are
  optional extras. Do not add or upgrade binary packages as incidental cleanup.
- In Colab, retain the coordinated runtime stack and editable `--no-deps`
  installation. Do not independently reinstall NumPy, Pandas, PyArrow, or
  PyTorch.

## Scientific integrity

- Record source, dataset version, accession or assembly selection, manifest
  hash, tokenizer checksum, split fingerprint, seeds, code revision, complete
  configuration, runtime, device, dtype, and backend. Do not backfill these
  values by hand.
- Treat datasets and manifests used for a reported run as immutable. A changed
  split, deduplication rule, or grouping boundary defines a new experiment.
- Split biological data before sharding, tokenization, or windowing. For
  bacterial Titan, accession and clade/ANI groups MUST NOT cross train,
  validation, and test. Do not use random token-window fallback splits.
- Check other sequence tasks for normalized duplicates, homologous sequences,
  and group leakage across splits.
- Choose thresholds, checkpoints, model variants, and hyperparameters using
  training and validation only. Inspect the test set only after freezing the
  configuration. Never select a winner with test metrics.
- Fair baselines use the same dataset contract, split boundary, label
  definition, preprocessing scope, seed policy, parameter or compute budget,
  and metrics. State every deliberate difference.
- Report failed, interrupted, negative, and inconclusive results directly.
  Synthetic gates, smoke tests, and unlabelled scans are not biological
  performance evidence.

## Titans research contract

Stage A is a deterministic synthetic mechanism gate, not a DNA benchmark.
Stage B selected `exact_accelerated + SDPA` in FP32 as the Stage C correctness
path. Approximate scan and Flash attention remain experimental or ineligible
until their causality, mask, parity, and target-hardware gates pass again.

- Preserve strict causality. Future-token perturbations MUST NOT alter earlier
  logits, reads, or valid memory state.
- Keep memory state explicit and stream-local. Reset it at stream, split, and
  replacement boundaries. Never carry it between unrelated accessions or from
  training into evaluation.
- Keep numerical state persistence separate from autograd retention. Record the
  gradient horizon, preserve numerical state at detach boundaries, and do not
  hide truncation behavior.
- Checkpoints MUST bind model configuration, optimizer, scheduler/data cursor,
  RNG, stream states, dataset fingerprint, and code provenance. Resume only
  when the contract matches. Preserve read-only resume verification.
- Report held-out BPB/loss, per-accession and GC-stratified metrics, matched
  controls, memory telemetry, throughput, and peak memory. Do not describe a
  partial evaluation as a full-corpus result.
- Hardware results are configuration-specific. Record hardware, dtype,
  attention and memory backend, valid-base budget, batch/accumulation settings,
  and measured resource use. Do not extrapolate a small probe into a full run.

## Colab and notebooks

Colab is ephemeral. Stage C notebook commands MUST remain revision-pinned,
Drive-backed handoffs that write logs, a run manifest, and a failure marker.
Keep resume paths working. Mark interrupted attempts as interrupted. Keep
credentials, tokens, Drive paths, caches, and generated artifacts out of Git.

## Documentation and references

- Update user-facing documentation when a public command or run artifact
  changes.
- Use structured parsers for JSON, TOML, CSV, manifests, and checkpoints.
- When a protocol and implementation disagree, preserve scientific behavior,
  identify the discrepancy, and request a decision or update the documentation
  in the same reviewed change.
