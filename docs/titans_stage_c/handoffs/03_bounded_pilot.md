# Handoff 03 — bounded genomic pilot

Notebook: `notebooks/titans_stage_c/03_stage_c_bounded_genomic_pilot.ipynb`

## Runtime, budget, and resume

Use an A100 after handoffs 00–02 are green. Edit the reviewed commit, locked
dataset, chosen horizon, base budget, and checkpoint frequency. The default
budget is 100 million valid bases for each separately trained adaptive,
reference, frozen-memory, and no-memory run. The combined pilot allocation is
at most 105 Colab compute units.

Each run writes atomically to its own Drive directory. Rerunning the same cells
restores `latest.pt`, optimizer/RNG state, stream permutation, active stream
cursors, and functional states. Do not use `--no-resume` after a disconnect.

## Success and return

Success means four `latest.pt` checkpoints, histories, validation JSON, and run
manifests with identical dataset fingerprints and compatible geometry. A
single run failure does not authorize changing its data order or architecture;
return the Drive-persisted `logs/`, `FAILED.txt`, Colab manifest, and completed
sibling runs. Return the printed pilot root.
