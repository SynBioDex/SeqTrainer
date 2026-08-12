# Annotation Quickstart

This is the beginner/local-system guide for running the SeqTrainer promoter
annotation MVP and opening its output files.

For the detailed roadmap, original-vs-annotated GenBank comparison, CNN notes,
and follow-up questions, see
[`promoter_annotation_plan.md`](promoter_annotation_plan.md).

## 1. Install The Annotation Dependency

If you are using the Anaconda Python that provides `seqtrainer.exe`, run this
from PowerShell:

```powershell
cd C:\Users\Sgoff\MYfile\Desktop\PYThh\SeqTrainer
C:\Users\Sgoff\anaconda3\python.exe -m pip install -e ".[annotation]"
```

This installs Biopython into the same Python environment that runs
`seqtrainer`.

You can quickly verify it with:

```powershell
C:\Users\Sgoff\anaconda3\python.exe -c "from Bio import SeqIO; print('Biopython OK')"
```

## 2. Run A Dummy Smoke Annotation

Dummy mode only checks that the GenBank read/write workflow works. It does not
make biological claims.

```powershell
cd C:\Users\Sgoff\MYfile\Desktop\PYThh\SeqTrainer

seqtrainer annotate promoters C:\Users\Sgoff\Downloads\pAN1717_cyan.gb `
  --model-family dummy `
  --threshold 0.80 `
  --window-size 300 `
  --step-size 25 `
  --scan-both-strands `
  --output outputs\annotations\pAN1717_cyan_dummy_annotated.gb `
  --predictions-csv outputs\annotations\pAN1717_cyan_dummy_predictions.csv `
  --manifest outputs\annotations\pAN1717_cyan_dummy_manifest.json
```

Expected outputs:

```text
outputs\annotations\pAN1717_cyan_dummy_annotated.gb
outputs\annotations\pAN1717_cyan_dummy_predictions.csv
outputs\annotations\pAN1717_cyan_dummy_manifest.json
```

For repeat runs, add `--clean-output`. It removes the explicitly named primary
outputs and clears the explicitly named `--evaluation-dir` before inference,
so stale metrics, plots, validation reports, and prediction tables are not
mixed with the new run. Cleanup is opt-in and does not touch neighboring run
folders.

### Automatic model bundle loading

For DNABERT2, keep the trained checkpoint and matching benchmark manifest in
one folder. The folder may contain `manifest.json` and either
`checkpoints/best_model.pt` or `checkpoints/best.pt`. This avoids copying or
typing two paths for every annotation run:

```powershell
seqtrainer annotate promoters C:\\Users\\Sgoff\\Downloads\\pAN1717_cyan.gb --model-family dnabert2 --model-bundle outputs\\models\\dnabert2_kaggle_best --output outputs\\annotations\\pAN1717_dnabert2\\annotated.gb --predictions-csv outputs\\annotations\\pAN1717_dnabert2\\predictions.csv --manifest outputs\\annotations\\pAN1717_dnabert2\\manifest.json --sbol-output outputs\\annotations\\pAN1717_dnabert2\\annotated.nt --sbol2-output outputs\\annotations\\pAN1717_dnabert2\\annotated.rdf --clean-output --open-output-folder
```

The bundle is only a convenience for locating files. The checkpoint still
supplies the trained model weights, and `manifest.json` still supplies the
validation-selected threshold and preprocessing settings. Both files must
come from the same benchmark run.

## 3. Open The Output Files Locally

Open the output folder in File Explorer:

```powershell
explorer outputs\annotations
```

Open the annotated GenBank file in the default local viewer:

```powershell
Invoke-Item outputs\annotations\pAN1717_cyan_dummy_annotated.gb
```

Open the predictions CSV in Excel or your default spreadsheet app:

```powershell
Invoke-Item outputs\annotations\pAN1717_cyan_dummy_predictions.csv
```

Open the manifest JSON in VS Code:

```powershell
code outputs\annotations\pAN1717_cyan_dummy_manifest.json
```

If `code` is not available, open the manifest with Notepad:

```powershell
notepad outputs\annotations\pAN1717_cyan_dummy_manifest.json
```

## 4. Inspect The Output From PowerShell

Preview the manifest summary:

```powershell
Get-Content outputs\annotations\pAN1717_cyan_dummy_manifest.json
```

Preview the first few prediction rows:

```powershell
Import-Csv outputs\annotations\pAN1717_cyan_dummy_predictions.csv |
  Select-Object -First 10 |
  Format-Table -AutoSize
```

Count predicted promoter features in the annotated GenBank file:

```powershell
C:\Users\Sgoff\anaconda3\python.exe -c "from Bio import SeqIO; r=SeqIO.read('outputs/annotations/pAN1717_cyan_dummy_annotated.gb','genbank'); labels=[f.qualifiers.get('label',[''])[0] for f in r.features]; print('total features:', len(r.features)); print('predicted promoters:', labels.count('predicted_promoter'))"
```

## 5. Prepare A Real DNABERT2 Checkpoint

Real DNABERT2 annotation needs both files from a completed benchmark run:

```text
outputs\models\dnabert2_kaggle_best\manifest.json
outputs\models\dnabert2_kaggle_best\checkpoints\best_model.pt
```

The manifest stores the model settings and validation-selected threshold. The
checkpoint stores the trained weights.

The combined annotation branch includes this bundle through Git LFS. After
cloning it, download the large-file content with:

```powershell
git lfs install
git lfs pull
```

If Git LFS is unavailable, use the repository helper with the supplied Kaggle
archive instead:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\prepare_dnabert2_annotation_bundle.ps1 -Archive "C:\Users\Scientist\Downloads\dnabert2_final_training_t4_seed42.zip"
```

Install the DNABERT2 runtime dependencies:

```powershell
C:\Users\Sgoff\anaconda3\python.exe -m pip install -e ".[annotation,torch]"
```

On native Windows, DNABERT2 may warn that Triton is unavailable. That is okay
for local CPU annotation because SeqTrainer disables FlashAttention and uses
DNABERT2's PyTorch attention fallback. It is slower than GPU/Colab/HPC.

## 6. Run A Real DNABERT2 Quick Check

This command uses the actual trained DNABERT2 checkpoint, but scans coarsely so
it finishes on a local CPU. This is why it is called a smoke check: the model is
real, but the scan is intentionally lightweight.

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

Equivalent one-line command for Windows Command Prompt:

```bat
seqtrainer annotate promoters C:\Users\Sgoff\Downloads\pAN1717_cyan.gb --model-family dnabert2 --checkpoint outputs\models\dnabert2_kaggle_best\checkpoints\best_model.pt --benchmark-manifest outputs\models\dnabert2_kaggle_best\manifest.json --output outputs\annotations\pAN1717_cyan_dnabert2_annotated.gb --predictions-csv outputs\annotations\pAN1717_cyan_dnabert2_predictions.csv --manifest outputs\annotations\pAN1717_cyan_dnabert2_manifest.json --clean-output --open-output-folder
```

`--clean-output` removes only this run's three target files before rerunning: the annotated GenBank, predictions CSV, and manifest JSON. `--open-output-folder` opens the output folder after a successful run so users can immediately view or copy/download the results.

Equivalent one-line command for Windows Command Prompt without auto-opening the folder:

```bat
seqtrainer annotate promoters C:\Users\Sgoff\Downloads\pAN1717_cyan.gb --model-family dnabert2 --checkpoint outputs\models\dnabert2_kaggle_best\checkpoints\best_model.pt --benchmark-manifest outputs\models\dnabert2_kaggle_best\manifest.json --step-size 300 --no-scan-both-strands --output outputs\annotations\pAN1717_cyan_dnabert2_smoke_annotated.gb --predictions-csv outputs\annotations\pAN1717_cyan_dnabert2_smoke_predictions.csv --manifest outputs\annotations\pAN1717_cyan_dnabert2_smoke_manifest.json
```

Observed sample output on `pAN1717_cyan.gb`:

```text
threshold = 0.677001953125
threshold_source = benchmark_manifest
window_size = 300
step_size = 300
scan_both_strands = false
total_windows_scanned = 20
windows_above_threshold = 6
predicted_promoters_added = 5
```

Predicted promoter regions from the verified Kaggle-checkpoint run:

| Region | Location | Strand | Score |
| --- | ---: | ---: | ---: |
| `predicted_promoter_0` | `901-1200` | `+` | `0.877948` |
| `predicted_promoter_1` | `2701-3300` | `+` | `0.906066` |
| `predicted_promoter_2` | `3901-4200` | `+` | `0.702515` |
| `predicted_promoter_3` | `5101-5400` | `+` | `0.858647` |
| `predicted_promoter_4` | `join(5701-5969,1-31)` | `+` | `0.848701` |

The pAN1717 file contains no labelled promoter gold features in this run, so
these predictions demonstrate end-to-end annotation and SBOL export, not
biological precision or recall. Use a curated plasmid with trusted promoter
annotations for that evaluation.

## 7. Run A Denser Real DNABERT2 Annotation

For a more complete scan, use a smaller step size and both strands:

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

This is the actual dense annotation mode, but it is slow on native Windows CPU.
Prefer Colab GPU, HPC, or a CUDA Linux environment for full plasmid scans.

Use `--model-family cnn_v2` for a CNN-v2 checkpoint once a compatible CNN-v2
annotation loader is added.

The `dummy` predictor is only for checking that files are produced correctly.
Only DNABERT2/CNN-v2 runs with real checkpoints should be used for scientific
interpretation.
