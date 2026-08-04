# c16 scientific communication package

This directory contains a reproducible, exploratory research article and slide
deck describing the c16 compact paper-deep Titans DNA model, its held-out
behavior, conditional-generation study, and decoding ablation.

Build the existing figures and presentation, then compile the article from
LaTeX:

```bash
.venv/bin/python docs/stage_c_c16_article/build_package.py
.venv/bin/python docs/stage_c_c16_article/build_latex_article.py
```

The second command requires Tectonic on `PATH`, in `TECTONIC`, or supplied as
`--tectonic /path/to/tectonic`. The LaTeX source is
`C16_Titans_DNA_Generation_Research_Article.tex`. The compiler validates the
page count and required title/methods text and records source, figure, Drive
PCA-input, compiler, and PDF hashes in `BUILD_MANIFEST.txt`.

Outputs are written to `artifacts/stage_c_c16_scientific_package/`.

The numerical inputs are frozen in `build_package.py` from the Drive-backed
ledger events `adaptive_exploration_5m`, `adaptive_exploration_5m_generation`,
and `adaptive_exploration_5m_decoding_ablation`. The checkpoint SHA-256 is
`21898362291f4fd1e6aafcfbe47e8b05dbe69e5c8036e6ae7927a6ac24ac4541`.
The PCA panels are rebuilt from the Drive-exported JSON files retained in
`drive_artifacts/`; they cover one accession and are explicitly not interpreted
as evidence of taxonomic separation.

The reviewed Figure 1 source is retained at
`assets/figure_1_model_architecture.png`. The LaTeX builder reinstalls this
asset into the output package before compilation so running the older
ReportLab figure builder cannot silently restore the superseded architecture
graphic.

This is an exploratory preprint-style communication artifact, not a
peer-reviewed publication and not evidence of functional genes, adaptive-memory
superiority, viability, or safety.
