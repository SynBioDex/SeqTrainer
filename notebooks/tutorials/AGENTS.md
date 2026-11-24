# Tutorial notebook instructions

The root `AGENTS.md` applies. Tutorials are public learning materials, not
benchmark evidence or published biological claims.

- Keep notebooks self-contained, restartable, and explicit about optional
  dependencies and hardware.
- Use small fixtures or clearly user-supplied inputs by default. Do not add a
  large download, long training run, or dependency on prior interactive state
  without explicit approval.
- Never commit credentials, tokens, personal Drive paths, model caches, or
  generated checkpoints.
- Preserve train, validation, and test separation. Do not choose a model,
  threshold, or hyperparameter from a held-out test result.
- Pin code and model revisions when making a reproducibility claim. Label demo
  or smoke output clearly and remove stale output that resembles a benchmark.
- Update the tutorial list in `README.md` when adding or renaming a notebook.
  Add a lightweight structural or functional test when a notebook establishes a
  repository contract.
