# Stage C paper-MAC instructions

This directory implements the Stage C causal bacterial language-model path.
The root `AGENTS.md` applies. These rules add the local research contract.

Read `docs/titans_stage_c/README.md`, the relevant handoff, and the Stage B
fidelity audit before changing the model, trainer, checkpoint, evaluation, or
CLI. The legacy `titans_mac` model is a preserved baseline, not a substitute.

- Keep the locked correctness path: exact accelerated memory, exact-mask SDPA,
  token-wise gates, and FP32. Do not silently promote Flash, approximate scan,
  or reduced precision.
- Treat a contig or replicon as an ordered stream. Key state by stream ID.
  Reset it at stream, split, and replacement boundaries. Require batched and
  isolated execution to agree within the declared tolerance.
- Preserve write-after-read ordering. Test future-token perturbation,
  valid-mask/tail handling, and state isolation after segment or attention
  changes.
- Preserve the configured truncated-backpropagation horizon. State persists
  numerically across its boundary, then detaches from autograd. Do not detach
  functional memory within an active horizon.
- Preserve checkpoint completeness: model, optimizer, scheduler/data cursor,
  Python/NumPy/Torch RNG, stream states, dataset/panel fingerprint, and model
  contract. Keep `resume_verify_cli` read-only for its source checkpoint.
- Record failed and interrupted Colab work through the existing wrapper. Do not
  label partial evidence as a completed run.

Start with the relevant `tests/test_titans_paper_mac_stage_c_*.py` test. Run
the workflow and model tests when shared state, checkpoints, notebook
contracts, or CLI behavior changes. Do not download the corpus or run the full
15 Gbp workflow for a code edit.
