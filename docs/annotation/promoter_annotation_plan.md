# Promoter Annotation Plan

This document describes the next annotation work for SeqTrainer after the
benchmark-model phase. The goal is to turn trained promoter classifiers into a
reproducible plasmid annotation workflow for GenBank files.

## Current Status

The annotation MVP can now:

- read a GenBank plasmid file with Biopython;
- preserve existing GenBank features;
- generate sliding sequence windows over circular or linear DNA;
- score windows with either a dummy smoke-test predictor or a trained DNABERT2
  benchmark checkpoint;
- use the validation-selected benchmark threshold from a manifest;
- merge passing windows into promoter feature calls;
- write an annotated GenBank file, a predictions CSV, and an annotation
  manifest.

The real DNABERT2 checkpoint path has been verified locally on the sample
plasmid `pAN1717_cyan.gb` in a lightweight CPU scan.

## Original GenBank Versus DNABERT2 Annotated Output

Input file:

```text
C:\Users\Sgoff\Downloads\pAN1717_cyan.gb
```

Real DNABERT2 verification output:

```text
outputs\annotations\pAN1717_kaggle_dnabert2\annotated.gb
outputs\annotations\pAN1717_kaggle_dnabert2\predictions.csv
outputs\annotations\pAN1717_kaggle_dnabert2\manifest.json
outputs\annotations\pAN1717_kaggle_dnabert2\annotated.nt
outputs\annotations\pAN1717_kaggle_dnabert2\annotated.rdf
```

Comparison:

| Item | Original GenBank | DNABERT2 annotated GenBank |
| --- | ---: | ---: |
| Sequence ID | `pAN1717_cyan` | `pAN1717_cyan` |
| Sequence length | 5,969 bp | 5,969 bp |
| Topology | circular | circular |
| Existing features | 20 | 20 preserved |
| Original feature types | 19 `misc_feature`, 1 `CDS` | unchanged |
| New predicted promoter features | 0 | 5 |

The quick-check run used:

| Setting | Value |
| --- | --- |
| Model | DNABERT2 full fine-tuned Kaggle checkpoint |
| Checkpoint | `outputs\models\dnabert2_kaggle_best\checkpoints\best_model.pt` |
| Benchmark manifest | `outputs\models\dnabert2_kaggle_best\manifest.json` |
| Threshold | `0.677001953125` |
| Threshold source | validation MCC from benchmark manifest |
| Window size | 300 bp |
| Step size | 300 bp |
| Strand scan | plus strand only |
| Device used locally | CPU |
| Windows scanned | 20 |
| Windows above threshold | 6 |
| Promoter features added | 5 |

Predicted promoter features:

| Region ID | Location | Strand | Score | Threshold |
| --- | ---: | ---: | ---: | ---: |
| `predicted_promoter_0` | `901-1200` | `+` | `0.877948` | `0.677002` |
| `predicted_promoter_1` | `2701-3300` | `+` | `0.906066` | `0.677002` |
| `predicted_promoter_2` | `3901-4200` | `+` | `0.702515` | `0.677002` |
| `predicted_promoter_3` | `5101-5400` | `+` | `0.858647` | `0.677002` |
| `predicted_promoter_4` | `join(5701-5969,1-31)` | `+` | `0.848701` | `0.677002` |

The input contains no labelled promoter gold features, so the run verifies
model loading, window scoring, GenBank writing, and SBOL export only. It does
not establish biological precision or recall.

Important interpretation note: this is an actual DNABERT2 checkpoint run, but it
is still a coarse local scan. It verifies model loading and GenBank writing. A
scientific annotation pass should use a denser stride and both strands.

## How To Run The Verified Real-Model Quick Check

From the SeqTrainer repo root:

```powershell
cd C:\Users\Sgoff\MYfile\Desktop\PYThh\SeqTrainer
```

Install dependencies:

```powershell
C:\Users\Sgoff\anaconda3\python.exe -m pip install -e ".[annotation,torch]"
```

Prepare the Kaggle checkpoint bundle if needed:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\prepare_dnabert2_annotation_bundle.ps1 -Archive "C:\Users\Sgoff\Downloads\dnabert2_final_training_t4_seed42.zip"
```

Run the quick check:

```powershell
seqtrainer annotate promoters C:\Users\Sgoff\Downloads\pAN1717_cyan.gb `
  --model-family dnabert2 `
  --checkpoint outputs\models\dnabert2_kaggle_best\checkpoints\best_model.pt `
  --benchmark-manifest outputs\models\dnabert2_kaggle_best\manifest.json `
  --step-size 300 `
  --no-scan-both-strands `
  --output outputs\annotations\pAN1717_cyan_dnabert2_smoke_annotated.gb `
  --predictions-csv outputs\annotations\pAN1717_cyan_dnabert2_smoke_predictions.csv `
  --manifest outputs\annotations\pAN1717_cyan_dnabert2_smoke_manifest.json
```

Windows Command Prompt one-line form:

```bat
seqtrainer annotate promoters C:\Users\Sgoff\Downloads\pAN1717_cyan.gb --model-family dnabert2 --checkpoint outputs\models\dnabert2_kaggle_best\checkpoints\best_model.pt --benchmark-manifest outputs\models\dnabert2_kaggle_best\manifest.json --step-size 300 --no-scan-both-strands --output outputs\annotations\pAN1717_cyan_dnabert2_smoke_annotated.gb --predictions-csv outputs\annotations\pAN1717_cyan_dnabert2_smoke_predictions.csv --manifest outputs\annotations\pAN1717_cyan_dnabert2_smoke_manifest.json
```

Open the outputs:

```powershell
explorer outputs\annotations
Invoke-Item outputs\annotations\pAN1717_cyan_dnabert2_smoke_predictions.csv
notepad outputs\annotations\pAN1717_cyan_dnabert2_smoke_manifest.json
```

## How To Run A Denser Real Annotation

Use a smaller stride and both strands:

```powershell
seqtrainer annotate promoters C:\Users\Sgoff\Downloads\pAN1717_cyan.gb `
  --model-family dnabert2 `
  --checkpoint outputs\models\dnabert2_kaggle_best\checkpoints\best_model.pt `
  --benchmark-manifest outputs\models\dnabert2_kaggle_best\manifest.json `
  --step-size 25 `
  --scan-both-strands `
  --output outputs\annotations\pAN1717_cyan_dnabert2_full_annotated.gb `
  --predictions-csv outputs\annotations\pAN1717_cyan_dnabert2_full_predictions.csv `
  --manifest outputs\annotations\pAN1717_cyan_dnabert2_full_manifest.json
```

This is the more realistic annotation mode, but it is slow on native Windows
CPU. For routine full-plasmid scans, prefer Colab GPU, Alpine HPC, or another
CUDA Linux environment.

## Can CNN Be Used Here?

Yes, CNN can be used in the same annotation workflow conceptually:

```text
GenBank plasmid -> sliding windows -> CNN-v2 probability -> threshold -> merged promoter features
```

What is already shared:

- the annotation CLI accepts `--model-family cnn_v2`;
- the same GenBank window generator can feed CNN-v2;
- the same output files can be written;
- the same benchmark-manifest threshold policy can be used.

What still needs implementation:

- a CNN-v2 checkpoint loader for annotation;
- one-hot encoding for annotation windows using the same 300 bp preprocessing as
  the CNN benchmark;
- checkpoint metadata validation so the CNN annotation loader knows the expected
  sequence length and model variant;
- a smoke test proving CNN-v2 annotations reproduce the same CSV/GenBank output
  shape as DNABERT2.

Recommended order:

1. Keep DNABERT2 as the first real model in the annotation MVP, because its
   checkpoint path has now been proven.
2. Add CNN-v2 annotation loading next so scientists can compare a small local
   model against DNABERT2 on the same plasmid windows.
3. Later add iPro-MP as an external predictor if its input/output format can be
   adapted cleanly to plasmid windows.

## Development Plan

### Phase 1: Stabilize Real DNABERT2 Annotation

- Add a progress indicator for long scans.
- Add `--batch-size` override so CPU/GPU users can tune throughput.
- Add optional `--limit-windows` for debugging.
- Add resume/cache support for predictions CSV.
- Add a clear warning when a full dense scan is running on CPU.
- Keep threshold selection fixed from the validation benchmark manifest.

Success criteria:

- quick check finishes locally;
- dense scan finishes on GPU/HPC;
- output GenBank can be opened in standard plasmid tools;
- predictions CSV and manifest fully explain every feature added.

### Phase 2: Improve Post-Processing

- Merge nearby positive windows more carefully across circular boundaries.
- Add confidence bands, for example `low`, `medium`, `high`.
- Add promoter-to-CDS association:
  - nearest downstream CDS;
  - distance to CDS start;
  - strand compatibility;
  - circular wraparound support.
- Report overlaps with existing promoter/library annotations separately from
  overlaps with generic features.

Success criteria:

- each predicted promoter has a likely target gene/CDS when possible;
- circular plasmid boundary predictions are represented cleanly;
- CSV output is easy for wet-lab users to audit.

### Phase 3: Add CNN-v2 Annotation

- Load the CNN-v2 checkpoint from `checkpoints/best_model.pt`.
- Reuse the same 300 bp DNA normalization and one-hot encoding as the benchmark.
- Use the benchmark manifest threshold.
- Add a CNN-v2 smoke test on the synthetic GenBank fixture.
- Run CNN-v2 and DNABERT2 on the same plasmid windows and compare predicted
  regions.

Success criteria:

- `--model-family cnn_v2` works end to end;
- CNN and DNABERT2 predictions are comparable on identical windows;
- output manifests record model family, checkpoint, threshold, and preprocessing.

### Phase 4: Evaluation Against Curated Plasmids

- Collect plasmids with known promoter annotations.
- Run DNABERT2 and CNN-v2 on the same candidate windows.
- Compare region-level precision/recall, not only window-level metrics.
- Track whether predictions overlap expected promoter regions within a tolerance
  window, for example +/- 50 bp.

Success criteria:

- annotation quality can be measured against real curated plasmid labels;
- model choice is based on promoter-region performance, not only benchmark CSV
  classification metrics.

### Phase 5: User-Facing Workflow

Possible interfaces:

- beginner notebook for uploading `.gb` and downloading annotated `.gb`;
- command-line workflow for reproducible local/HPC runs;
- later FastAPI or web app where scientists upload GenBank/FASTA and select a
  model checkpoint.

For now, the CLI should remain the source of truth because it is easiest to
test and reproduce.

## Follow-Up Questions

Use these questions with mentors/scientists before locking the annotation design:

1. Should predicted promoter features be allowed to overlap existing annotated
   promoters, or should overlaps be reported but not added?
2. What genomic distance should count as promoter-to-CDS association in bacterial
   plasmids?
3. Should promoter calls be strand-specific by default?
4. Should reverse-strand promoter predictions be shown in the GenBank output, or
   only in the CSV until validated?
5. What confidence bands are useful for wet-lab users: low/medium/high, or exact
   probability only?
6. Should the threshold always come from validation MCC, or should users be able
   to choose a more sensitive threshold for discovery workflows?
7. What is the acceptable false-positive rate for plasmid promoter discovery?
8. Should dense scans use 25 bp stride, 10 bp stride, or motif/candidate-region
   prefiltering?
9. Should FASTA input be supported directly, with GenBank output generated from
   scratch?
10. Should SBOL import/export be part of the first annotation workflow or a
    separate second phase?
11. Which model should be treated as the default annotation model if CNN-v2 is
    faster but DNABERT2 is more biologically expressive?
12. Do users need a Colab notebook first, or is a local CLI plus documented
    outputs enough for the next milestone?
