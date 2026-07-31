# c16 scientific communication package

This directory contains a reproducible, exploratory research article and slide
deck describing the c16 compact paper-deep Titans DNA model, its held-out
behavior, conditional-generation study, and decoding ablation.

Build:

```bash
.venv/bin/python docs/stage_c_c16_article/build_package.py
```

Outputs are written to `artifacts/stage_c_c16_scientific_package/`.

The numerical inputs are frozen in `build_package.py` from the Drive-backed
ledger events `adaptive_exploration_5m`, `adaptive_exploration_5m_generation`,
and `adaptive_exploration_5m_decoding_ablation`. The checkpoint SHA-256 is
`21898362291f4fd1e6aafcfbe47e8b05dbe69e5c8036e6ae7927a6ac24ac4541`.

This is an exploratory preprint-style communication artifact, not a
peer-reviewed publication and not evidence of functional genes, adaptive-memory
superiority, viability, or safety.
