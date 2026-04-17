# SeqTrainer

SeqTrainer is a synthetic biology ML domain toolkit focused on connecting **SBOL / SynBioHub data** to modern model workflows.

It is designed to be complementary to Keras and PyTorch rather than replacing them.

## What this refactor introduces

- Clear package layering under `seqtrainer/`
- Framework-neutral core (`clients`, `sparql`, `data`, `transforms`, `models`)
- Optional framework adapters (`seqtrainer.keras`, `seqtrainer.torch`)
- Graph-focused utilities in `seqtrainer.graph`
- Application-level API entrypoints in `seqtrainer.applications`
- CLI foundation (`seqtrainer` command)

## Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e '.[torch]'
pip install -e '.[keras]'
pip install -e '.[gnn]'
pip install -e '.[dev]'
```

## Package layout

- `seqtrainer/clients`: SynBioHub and remote clients
- `seqtrainer/sparql`: prefixes, builders, and query recipes
- `seqtrainer/data`: SBOL loaders, recipes, materialized datasets
- `seqtrainer/transforms`: DNA transforms and feature extraction
- `seqtrainer/models`: framework-neutral backbone/head registry stubs
- `seqtrainer/keras`: Keras adapters/factories (optional dependency)
- `seqtrainer/torch`: PyTorch adapters/fine-tune helpers (optional dependency)
- `seqtrainer/graph`: RDF/SBOL graph conversion utilities
- `seqtrainer/applications`: task-oriented blueprints
- `seqtrainer/cli`: command-line entrypoints

## CLI examples

```bash
seqtrainer sparql prefixes
seqtrainer inspect-sbol data/sbol_data/sample_design_0.xml
seqtrainer build-dataset data/sbol_data/sample_design_0.xml
```

## Status

This is the first architecture-focused cleanup. Some framework integrations are intentionally placeholders with TODOs to keep a stable, minimal public surface.
