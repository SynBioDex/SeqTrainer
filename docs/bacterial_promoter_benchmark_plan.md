# Bacterial Promoter Benchmark and Annotation Plan

This document defines the scientific plan for reproducible bacterial promoter
prediction benchmarks and plasmid promoter annotation in SeqTrainer. It is meant
for researchers who want to understand what is being evaluated, what evidence is
needed to compare models, and what outputs should be reproducible.

## Summary

SeqTrainer will benchmark promoter prediction models on shared DNA sequence
examples, using shared splits, shared metrics, and recorded provenance. The
starting reference is the existing CNN tutorial baseline. Candidate models such
as DNABERT2 and iPro-MP should be adopted for annotation only when they improve
relevant benchmark metrics or provide a clearly justified scientific or
operational tradeoff.

The first annotation target is plasmid promoter prediction. The expected output
is a structured table of predicted promoter calls and an SBOL annotation draft
that preserves enough provenance for another researcher to inspect and rerun the
workflow.

The near-term priority is benchmark discipline rather than model novelty. Before
SeqTrainer uses a stronger model to make annotation claims, it should make the
data, labels, splits, thresholds, metrics, and run conditions identical across
model families.

## Current Implementation Status

The plan is being implemented in staged benchmark PRs rather than one large
annotation change.

- PR #14 added the TOML benchmark configuration contract for shared dataset,
  split, seed, metric, threshold, and output settings.
- PR #15 merged the CNN benchmark work into upstream `dev`. It reproduces the
  reference CNN path and adds CNN-v2 with AdamW, scheduler support, dropout,
  validation-MCC checkpoint selection, early stopping, threshold-consistent
  predictions, row-aligned artifacts, and Colab-ready CNN notebooks.
- PR #20 is the active combined model-baselines branch for DNABERT2 and iPro-MP.
  It builds on the CNN benchmark contract and keeps DNABERT2 and iPro-MP on the
  same predefined train/validation/test CSV splits, shared metrics, validation-
  only threshold selection, manifests, prediction tables, and comparison
  artifacts.
- DNABERT2 now has package-facing frozen and fine-tuning paths, plus Colab and
  Alpine/SLURM execution profiles. Colab T4 notebooks are intended as lightweight
  reproducibility paths; Alpine/HPC runs are the preferred route for claim-bearing
  fine-tuning scores.
- iPro-MP is treated first as an external pretrained E. coli inference baseline,
  not a retraining task. SeqTrainer converts shared CSV splits to iPro-MP inputs,
  records model/dependency provenance, and converts predictions back into the
  shared benchmark metrics.

The next scientific step is not more benchmark scaffolding. It is to run the
HPC/Colab model paths, compare CNN-v2, DNABERT2, and iPro-MP with the same test
split, and improve the DNABERT2/iPro-MP scores only through controlled changes
that preserve the shared evaluation contract.

## Existing SeqTrainer Capabilities

SeqTrainer already provides several building blocks for this work:

- SBOL-to-table extraction in `seqtrainer.data.sbol`.
- DNA preprocessing in `seqtrainer.transforms.dna`, including sequence
  normalization, padding/trimming, one-hot encoding, GC content, and k-mer
  features.
- Seeded train/validation/test splitting through
  `seqtrainer.data.materialized.MaterializedDataset`.
- A framework-neutral model registry stub in `seqtrainer.models.registry`.
- Optional PyTorch and Keras adapter namespaces.
- A command-line foundation with `inspect-sbol`, `build-dataset`, and SPARQL
  helper commands.
- Tutorial notebooks in `notebooks/tutorials/` for SBOL extraction, DNA
  features, dataset splits, CNN classification, and CNN regression.

The current CNN classification tutorial is a demonstration baseline. It uses 40
sample SBOL XML files from `data/sbol_data/`, derives binary labels by
thresholding the numeric `target` at the median, pads/trims sequences to a fixed
length, one-hot encodes the sequences, splits with seed `42`, and trains a small
PyTorch Conv1D model for 10 cycles with unweighted cross-entropy loss.

## Scientific Question

The benchmark asks:

> Given bacterial DNA sequence records or candidate sequence windows, how well
> can a model predict promoter-positive versus promoter-negative examples in a
> reproducible way?

Each benchmark example should contain:

- a DNA sequence or sequence window,
- a binary promoter label,
- source and provenance metadata,
- split assignment,
- stable identifiers that connect predictions back to the original sequence or
  SBOL source.

All model families must be evaluated on the same examples and split definitions.
The model selected for annotation should then be applied to plasmid candidate
windows to generate predicted promoter calls.

## Benchmark Decision Rules

Every benchmark comparison should separate three decisions:

1. how examples are labeled,
2. how examples are split,
3. how model scores are converted into predicted labels.

The label rule should be defined once for a dataset version. Curated binary
promoter/non-promoter labels are preferred when available. When binary labels are
derived from a numeric `target`, the thresholding rule must be recorded and the
original numeric target should be preserved for later calibration, regression, or
error analysis.

The split rule should also be defined once for a dataset version. The tutorial
baseline may keep its original seeded split for exact reproduction. Claim-bearing
benchmarks should use a saved split snapshot shared by CNN, DNABERT2, iPro-MP,
and any later model. When multiple examples can come from the same source record,
plasmid, genome region, or sequence family, the split strategy should prevent
related records from appearing in both training and test data.

The decision threshold should be selected on validation data and then frozen
before test evaluation. A default `0.5` threshold or raw `argmax` is acceptable
for reproducing the tutorial only, but not for model selection. For imbalanced
classification, threshold selection should emphasize MCC, AUPRC, balanced
accuracy, sensitivity, and specificity rather than accuracy alone.

No candidate model should be selected for annotation only because it has higher
training accuracy. Selection should be based on held-out test behavior under the
same data, split, metric, threshold-selection, and output rules.

## Data and Labels

Labels may come from either:

1. curated binary promoter/non-promoter annotations, when available; or
2. a documented threshold over a numeric `target` value, when using the current
   tutorial-style SBOL data.

When numeric targets are thresholded, the threshold must be saved with the
results. The tutorial median threshold is acceptable for reproducing the
demonstration CNN baseline. However, benchmark results intended to support
scientific claims should explain why the chosen threshold is appropriate for the
dataset.

Every dataset row should preserve:

- source file or source record identifier,
- sequence identifier when available,
- numeric target when used,
- binary label,
- threshold or label rule,
- parsing or validation warnings where applicable.

## Initial Benchmark Scope

The initial benchmark should include:

- reproduction of the existing CNN classification baseline,
- a shared benchmark interface for CNN, DNABERT2, and iPro-MP,
- shared seeded train/validation/test splits,
- shared metric computation,
- JSON/CSV metrics output,
- prediction tables,
- run manifests with environment, seed, split, model version, threshold, and
  hyperparameters,
- class distribution reporting and an explicit imbalance-handling policy,
- DNABERT2 frozen-embedding baseline, with classifier-head fine-tuning only if
  the package path is stable,
- iPro-MP setup, smoke testing, wrapper or adapter evaluation, and conversion
  between SeqTrainer split tables and iPro-MP-compatible FASTA/prediction
  tables,
- unified model comparison on the same held-out test split,
- a model decision record explaining which model path should be used for
  plasmid annotation.

## Benchmark Development Roadmap

The benchmark should be developed in the following order. Items marked complete
or active reflect the current project state as of July 2026.

1. **Reproduce the CNN tutorial baseline.** Completed in PR #15. This establishes a reference result
   for the current notebook behavior, including the median-derived labels,
   existing preprocessing, seed `42`, and unweighted loss.
2. **Create a shared experiment contract.** Completed in PR #14 and extended in
   PR #15/#20. A single configuration schema should
   describe datasets, labels, splits, model identifiers, thresholds,
   hyperparameters, metrics, seeds, environment details, and output paths for all
   model families.
3. **Persist shared splits and metrics.** Completed for CNN and active for
   DNABERT2/iPro-MP in PR #20. Split files, metric computation, JSON
   and CSV output, and prediction tables should be produced by shared code so
   CNN, DNABERT2, and iPro-MP are evaluated under identical rules.
4. **Improve the CNN baseline after reproduction.** Completed in PR #15. CNN improvements should be
   treated as controlled ablations, such as weighted loss, early stopping,
   validation-threshold selection, or reverse-complement augmentation. These
   improved CNNs should be compared against the exact reproduced CNN baseline,
   not replace it silently.
5. **Add a frozen DNABERT2 baseline.** Active in PR #20. DNABERT2 should first be used as a frozen
   sequence encoder with a lightweight classifier head. This is lower risk than
   full fine-tuning and gives a strong transfer-learning comparison.
6. **Evaluate iPro-MP through an adapter.** Active in PR #20. iPro-MP should first be isolated
   behind a file-format adapter and smoke-tested on the shared split. It should
   not be deeply coupled to SeqTrainer internals until it proves useful on the
   same benchmark outputs.
7. **Select a model for annotation.** Pending real model-comparison outputs. The selected model should improve relevant
   held-out metrics or provide a justified tradeoff such as simpler operation,
   better reproducibility, or more useful annotation behavior.
8. **Build the plasmid annotation MVP.** Pending model decision. Only after model selection should the
   annotation workflow generate candidate windows, score them, merge positive
   windows, and export promoter call tables and SBOL drafts.

This sequence keeps the project scientifically interpretable. It avoids using a
larger model to compensate for unclear labels, leaking splits, or inconsistent
metrics.

## Future Model Strategy

The model roadmap should remain evidence-based and staged.

- **CNN baseline:** the current Conv1D tutorial model is the reference point.
  The first goal is exact reproduction, followed by controlled CNN ablations.
- **Improved CNN:** scientifically relevant improvements include class-weighted
  or positive-weighted loss, early stopping, validation-selected thresholds,
  reproducibility controls, and reverse-complement or small positional
  perturbation tests when biologically justified.
- **DNABERT2 frozen baseline:** this is the first recommended non-CNN model path.
  DNABERT2 is designed as an efficient genomic foundation model, and a frozen
  encoder plus classifier head is practical when labeled promoter data are
  limited.
- **DNABERT2 fine-tuning or LoRA:** fine-tuning should come after stable data,
  splits, and frozen-embedding results. Parameter-efficient fine-tuning is a
  better early option than full fine-tuning if data or GPU memory are limited.
- **iPro-MP:** iPro-MP is directly relevant because it targets multiple
  prokaryotic promoters and reports promoter metrics such as AUC, AUPRC, and
  MCC. In SeqTrainer it should first be benchmarked through an adapter using the
  same split and metric code as every other model.

### iPro-MP Reference Assumptions

The iPro-MP paper and repository make it a useful comparison model, but also an
external-system dependency rather than a simple in-package model. For SeqTrainer,
the first iPro-MP milestone should therefore be a wrapper and smoke test, not
retraining.

The integration assumptions are:

- Treat iPro-MP as a DNABERT-based external predictor with its own environment.
- Use the documented FASTA input and CSV prediction output boundary.
- Start with the E. coli species model when benchmarking the current promoter
  dataset, because the current benchmark target is bacterial promoter prediction.
- Preserve SeqTrainer split IDs when converting CSV splits to FASTA, so iPro-MP
  scores can be joined back to the same metrics and prediction tables.
- Record the iPro-MP species ID, downloaded model archive/version, local model
  path, command, and output CSV path in the benchmark manifest.
- Keep any iPro-MP cross-validation or retraining separate from the shared
  held-out test split. The held-out test split should remain untouched until a
  candidate model or threshold has been selected using training/validation data.
- Do not commit pretrained iPro-MP model files. Document how to download them
  and where they should live locally.

The working principle is: make the benchmark trustworthy first, then increase
model complexity only when the extra complexity answers a scientific or
operational question.

## Initial Annotation Scope

The initial annotation workflow should:

1. generate sliding windows or candidate promoter regions from plasmid
   sequences,
2. score those windows with the selected model,
3. merge overlapping positive windows into promoter feature calls,
4. export predicted promoters to a structured table,
5. export an SBOL annotation draft,
6. evaluate predictions against a curated held-out plasmid set when such a set
   is available.

The first version may use simple sliding windows and basic overlap merging.
Strand-aware post-processing, promoter-CDS association, confidence bands, and
genome-scale annotation are follow-on improvements unless they are required for
the curated plasmid evaluation.

## Initial Non-Goals

The initial benchmark and plasmid annotation workflow do not require:

- full genome-level E. coli annotation,
- retraining iPro-MP from scratch,
- cross-species iPro-MP transfer experiments,
- long-context model integration,
- production-grade model serving,
- exhaustive hyperparameter search,
- replacing PyTorch, Keras, Hugging Face, or iPro-MP with SeqTrainer-native model
  implementations,
- treating tutorial-only median-threshold results as a final biological claim.

## Required Metrics

Every benchmarked model path should report:

- accuracy,
- balanced accuracy,
- AUROC,
- AUPRC,
- F1,
- MCC,
- sensitivity/recall for the positive class,
- specificity for the negative class,
- confusion matrix.

For imbalanced datasets, AUPRC, MCC, balanced accuracy, sensitivity, and
specificity should be emphasized over plain accuracy. If a split contains only
one observed class, metrics that require both classes should be reported as
undefined or skipped with an explicit warning.

Candidate models should be compared against the reproduced CNN baseline.
DNABERT2, iPro-MP, or any improved model should be selected for annotation only
if it improves relevant metrics or provides a clearly justified tradeoff, such
as better reproducibility, lower operational complexity, or more useful
annotation behavior. Accuracy alone is not sufficient evidence of improvement.

When reporting model comparisons, the exact CNN tutorial baseline should remain
visible even if an improved CNN is added. This prevents the improved CNN from
moving the baseline target and makes it clear whether gains come from model
architecture, loss weighting, threshold choice, split strategy, or data changes.

## Reproducibility Requirements

Each benchmark run should write:

- metrics JSON,
- metrics CSV or an appendable summary table,
- prediction table with example IDs and scores,
- split metadata,
- run manifest,
- configuration file or resolved configuration snapshot.

The run manifest should include:

- git commit and branch,
- package version or local editable install marker,
- Python version and relevant package versions,
- model family and model/checkpoint identifier,
- seed,
- split identifier,
- label rule and threshold,
- sequence preprocessing parameters,
- class distribution per split,
- hyperparameters,
- output directory.

The resolved configuration and manifest should make it possible to answer:

- which data version and label rule were used,
- whether the split was tutorial reproduction, random seeded, stratified, or
  grouped,
- how the decision threshold was chosen,
- which metrics were used for model selection,
- whether imbalance handling was applied,
- which model checkpoint or external model version was used,
- whether the run is an exact baseline reproduction or an improved ablation.

A researcher should be able to reproduce a reported result by installing
SeqTrainer, obtaining the documented data and model files, and running the
recorded configuration against the recorded split.

## Annotation Outputs

The plasmid annotation workflow should produce two primary artifacts.

### Promoter Call Table

The structured promoter table should include:

- plasmid or sequence ID,
- start and end coordinates,
- strand when supported,
- model score or probability,
- threshold,
- predicted label,
- supporting window IDs,
- run or model identifier.

### SBOL Annotation Draft

The SBOL annotation draft should contain predicted promoter features and
provenance sufficient to distinguish model predictions from existing curated
features.

## Reproduction Target

Another researcher should be able to:

1. build the same dataset or load the same materialized dataset snapshot,
2. inspect the label rule and class distribution,
3. reuse the same train/validation/test split,
4. rerun the CNN baseline,
5. rerun the DNABERT2 baseline path,
6. rerun the iPro-MP baseline path or documented iPro-MP adapter path,
7. compare models with the same metric code,
8. inspect the model decision record,
9. run the selected model on plasmid candidate windows,
10. regenerate promoter call tables and SBOL annotation drafts.

## Scope Checklist

This scope is satisfied when it:

- identifies the current CNN reference baseline,
- defines the promoter prediction and plasmid annotation tasks,
- states label and source assumptions,
- separates initial scope from follow-on work,
- lists required benchmark metrics,
- defines reproducibility outputs,
- defines annotation artifacts,
- defines benchmark decision rules for labels, splits, thresholds, and metrics,
- gives future implementation work a clear scientific target.

## Evidence Notes

This plan is guided by the current SeqTrainer package surface and by recent
genomic sequence-modeling and evaluation literature. DNABERT2 motivates an
efficient frozen-embedding baseline before heavier fine-tuning. iPro-MP motivates
an external adapter path for prokaryotic promoter prediction. MCC and AUPRC are
emphasized because plain accuracy can be misleading for binary or imbalanced
classification. Long-context model integration is left as future work because it
is most relevant when SeqTrainer moves beyond short promoter windows into larger
plasmid or genome-context modeling.

Useful references for the next implementation issues:

- iPro-MP article: Su et al., "iPro-MP: a BERT-based model to predict multiple
  prokaryotic promoters", Genome Biology, 2025,
  https://doi.org/10.1186/s13059-025-03819-9.
- iPro-MP source repository: https://github.com/Jackie-Suv/iPro-MP.
- iPro-MP model/data archive: https://doi.org/10.5281/zenodo.15180139.
