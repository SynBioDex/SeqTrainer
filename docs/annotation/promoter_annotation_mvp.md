# Promoter Annotation MVP

SeqTrainer's first promoter annotation MVP adds model-predicted promoter
features to GenBank plasmid records while preserving existing annotations.

## Exact Hits Versus Predicted Promoters

SeqImprove-style exact-hit annotation compares an input DNA sequence against a
library of known parts. It can confidently annotate known sequences that match a
library entry, but it cannot find a novel promoter-like region that is absent
from the library.

SeqTrainer model annotation is different. It scans the sequence with fixed-size
windows and assigns each window a promoter score from a selected model. These
features are probabilistic computational predictions. They must be labeled as
`predicted_promoter`, not as confirmed biological part names such as
`BBa_J23101`.

## How Benchmark Outputs Connect

The annotation command can read a benchmark `manifest.json` to reuse the same
model sequence length and validation-selected threshold. This keeps annotation
connected to the Issue 3 benchmark policy:

- do not tune on the test set
- use validation-selected thresholds when available
- preserve model family, checkpoint, and threshold provenance in the manifest

If no threshold is provided and no benchmark threshold is found, the MVP uses a
conservative default threshold of `0.80` and records that fallback.

## Model Families

- `dummy`: deterministic smoke-test predictor. It does not make biological
  claims and should only be used to verify file writing and pipeline behavior.
- `dnabert2`: primary real predictor interface. It requires a compatible
  checkpoint and optional torch/transformers dependencies. The MVP fails clearly
  if those are unavailable.
- `cnn_v2`: lightweight fallback predictor interface. It also requires a
  compatible benchmark checkpoint.

The predictor interface is model-agnostic so iPro-MP or later models can be
added without changing the GenBank/window/manifest logic.

## Basic Smoke Run

Install the annotation extra first:

```bash
pip install -e ".[annotation]"
```

On Shreeya's local Anaconda setup, the matching install command is:

```powershell
C:\Users\Sgoff\anaconda3\python.exe -m pip install -e ".[annotation]"
```

For step-by-step local viewing commands, see `docs/annotation/README.md`.

Then run the dummy smoke test:

```bash
seqtrainer annotate promoters pAN1717_cyan.gb \
  --model-family dummy \
  --threshold 0.80 \
  --window-size 300 \
  --step-size 25 \
  --scan-both-strands \
  --output outputs/annotations/pAN1717_cyan_dummy_annotated.gb \
  --predictions-csv outputs/annotations/pAN1717_cyan_dummy_predictions.csv \
  --manifest outputs/annotations/pAN1717_cyan_dummy_manifest.json
```

## Real Model Runs

DNABERT2 example:

```bash
seqtrainer annotate promoters pAN1717_cyan.gb \
  --model-family dnabert2 \
  --checkpoint outputs/benchmarks/dnabert2_finetune/checkpoints/best.pt \
  --benchmark-manifest outputs/benchmarks/dnabert2_finetune/manifest.json \
  --output outputs/annotations/pAN1717_cyan_dnabert2_annotated.gb \
  --predictions-csv outputs/annotations/pAN1717_cyan_dnabert2_predictions.csv \
  --manifest outputs/annotations/pAN1717_cyan_dnabert2_manifest.json \
  --clean-output \
  --open-output-folder
```

`--clean-output` removes only the three target artifacts for this run before writing fresh outputs. `--open-output-folder` opens the output directory after success so users can immediately inspect or copy/download the annotated GenBank, predictions CSV, and manifest JSON.

CNN-v2 example:

```bash
seqtrainer annotate promoters pAN1717_cyan.gb \
  --model-family cnn_v2 \
  --checkpoint outputs/benchmarks/cnn_v2/checkpoints/best.pt \
  --benchmark-manifest outputs/benchmarks/cnn_v2/manifest.json \
  --output outputs/annotations/pAN1717_cyan_cnn_v2_annotated.gb \
  --predictions-csv outputs/annotations/pAN1717_cyan_cnn_v2_predictions.csv \
  --manifest outputs/annotations/pAN1717_cyan_cnn_v2_manifest.json
```

## Outputs

The annotated GenBank file preserves existing features and appends additional
`promoter` features with qualifiers including:

- `label=predicted_promoter`
- model family
- score
- threshold
- window and step size
- note saying this is a computational prediction only

The predictions CSV includes every scanned window, its score, threshold result,
strand, coordinates, circular-boundary flag, overlap status, merged region ID,
and window sequence.

The manifest records input/output paths, model metadata, threshold source,
window settings, circular topology, number of scanned windows, number of
predicted promoters added, overlap counts, warnings, timestamp, and git SHA.

## Current Limitations

- Dummy mode is not biologically meaningful.
- DNABERT2 and CNN-v2 annotation require compatible benchmark checkpoints.
- The MVP adds promoter-like regions by sliding windows, not by experimentally
  validating promoter activity.
- Predicted promoters remain separate from exact library-hit annotations.
