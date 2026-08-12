# Labelled Promoter Evaluation

This workflow evaluates an already-trained SeqTrainer model against locally downloaded Addgene GenBank records. It is external evaluation: the model is not retrained and the threshold is read from the benchmark manifest. Addgene sequences are never downloaded by SeqTrainer or committed to Git.

## Single plasmid

```powershell
seqtrainer annotate promoter data/addgene_18115/raw/pAN1717.gb `
  --model-family dnabert2 `
  --checkpoint outputs/models/dnabert2_kaggle_best/checkpoints/best_model.pt `
  --benchmark-manifest outputs/models/dnabert2_kaggle_best/manifest.json `
  --evaluation-dir outputs/addgene_18115/plasmids/pAN1717 `
  --sbol-output outputs/addgene_18115/plasmids/pAN1717/annotated.nt `
  --sbol2-output outputs/addgene_18115/plasmids/pAN1717/annotated_sbol2.rdf `
  --annotation-completeness partial `
  --promoter-label-mode strict
```

PowerShell uses a backtick for continuation. On Windows Command Prompt, use one line or replace the backticks with `^`.

The run preserves the input GenBank features and writes an annotated GenBank file, `predictions.csv`, `gold_promoters.csv`, `window_predictions.csv`, `merged_predictions.csv`, `promoter_matches.csv`, `metrics.csv`, `metrics.json`, `annotation_manifest.json`, `sbol_validation.json`, optional SBOL3 `annotated.nt`, and SBOL2 RDF/XML `annotated_sbol2.rdf` for SBOLCanvas. Add `--clean-output` to clear the named evaluation folder and replace old primary outputs before a repeat run.

## Collection

```powershell
seqtrainer annotate promoter-collection `
  --manifest data-manifests/addgene_article_18115.csv `
  --input-dir data/addgene_18115/raw `
  --output-dir outputs/addgene_18115 `
  --predictor dnabert2 `
  --model-path outputs/models/dnabert2_kaggle_best/checkpoints/best_model.pt `
  --benchmark-manifest outputs/models/dnabert2_kaggle_best/manifest.json `
  --promoter-label-mode strict `
  --annotation-completeness unknown `
  --write-sbol3 `
  --continue-on-error
```

`included_plasmids.csv` contains records with at least one explicit promoter annotation. `excluded_plasmids.csv` records unavailable files, missing labels, and processing errors. Use `verified_complete` only when the depositor annotations are known to cover the whole sequence; otherwise the window-level report is partial-label recovery, not a definitive negative-class benchmark.

## Evidence policy

Tier A accepts `promoter`, regulatory features with `regulatory_class=promoter`, or an explicit SO:0000167 cross-reference. Labelled mode additionally accepts a standalone `promoter` term in `label`, `name`, or `note`, excluding `promoterless` and `no promoter`. A promoter is never inferred from a plasmid name, motif, nearby CDS, gate name, or expected circuit architecture.

## Metrics and coordinate policy

Window labels use the centre of each strand-specific window and require the centre to lie inside a same-strand labelled promoter. Thresholds are fixed from the training benchmark and never tuned on Addgene. Merged predictions are matched one-to-one to gold promoters by strand-aware IoU, reported at 0.10, 0.25, and 0.50. Biopython uses 0-based/end-exclusive coordinates; SBOL3 uses 1-based/inclusive `Range` coordinates. Circular-origin features become ordered, bounded SBOL ranges.
