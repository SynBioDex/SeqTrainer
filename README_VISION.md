# SeqTrainer Vision

SeqTrainer aims to become a **"Keras for synthetic biology"** in spirit: easy domain APIs, composable defaults, and clear extension points.

## Core principle

SeqTrainer should own the **domain layer**:

- SynBioHub/SBOL retrieval and interoperability
- SPARQL recipes for common synbio data patterns
- Dataset recipe abstractions and provenance
- DNA-aware transforms and feature extraction
- Task-focused applications (e.g., promoter regression)

SeqTrainer should **not** replace model ecosystems:

- Keras remains a high-level training interface
- PyTorch remains native for HF/DNABERT and graph ecosystems

## Architecture direction

1. Framework-neutral core first
2. Optional framework adapters
3. Backbone + head composition
4. Pluggable application recipes
5. Strong provenance and reproducibility story over time

## Near-term roadmap

- Expand SynBioHub client auth + pagination support
- Curate reusable SPARQL recipe library for SBOL entities
- Add robust dataset caching/snapshot materialization
- Add production-grade torch/keras model factories
- Promote graph prototype scripts into stable graph APIs
