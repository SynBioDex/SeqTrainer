# Handoff 04 — held-out assessment

Notebook: `notebooks/titans_stage_c/04_stage_c_evaluation.ipynb`

## Protocol

Evaluation is read-only with respect to model weights. Edit the reviewed
commit, locked dataset, pilot root, split, and optional stream cap. Use `val`
while making decisions; run `test` once only after the configuration is frozen.
The notebook verifies dataset/checkpoint fingerprints before scoring.

## Authorization evidence

The output contains overall, per-clade, and per-accession BPB; token/top-2
accuracy and perplexity; whole-contig GC-bin BPB; memory retrieval, update,
surprise, gate, and state-drift telemetry; paired accession bootstrap intervals;
SVG comparisons; and
`EVALUATION_REPORT.md`. Full-corpus planning is green only when the separately
trained, distinct adaptive run beats both frozen-memory and no-memory by at least 0.01
BPB and each paired bootstrap interval supports the direction. Return the
complete printed evaluation directory. A red gate is a valid result and must
not be bypassed by editing the JSON or testing repeatedly on the test split.
