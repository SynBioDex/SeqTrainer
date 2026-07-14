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
outputs\benchmarks\dnabert2_finetune_t4_seed42\manifest.json
outputs\benchmarks\dnabert2_finetune_t4_seed42\checkpoints\best_model.pt
```

The manifest stores the model settings and validation-selected threshold. The
checkpoint stores the trained weights.

If `best_model.pt` was downloaded to `Downloads`, copy it into the expected
benchmark folder:

```powershell
mkdir outputs\benchmarks\dnabert2_finetune_t4_seed42\checkpoints
copy C:\Users\Sgoff\Downloads\best_model.pt outputs\benchmarks\dnabert2_finetune_t4_seed42\checkpoints\best_model.pt
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
  --checkpoint outputs\benchmarks\dnabert2_finetune_t4_seed42\checkpoints\best_model.pt `
  --benchmark-manifest outputs\benchmarks\dnabert2_finetune_t4_seed42\manifest.json `
  --step-size 300 `
  --no-scan-both-strands `
  --output outputs\annotations\pAN1717_cyan_dnabert2_smoke_annotated.gb `
  --predictions-csv outputs\annotations\pAN1717_cyan_dnabert2_smoke_predictions.csv `
  --manifest outputs\annotations\pAN1717_cyan_dnabert2_smoke_manifest.json
```

Equivalent one-line command for Windows Command Prompt:

```bat
seqtrainer annotate promoters C:\Users\Sgoff\Downloads\pAN1717_cyan.gb --model-family dnabert2 --checkpoint outputs\benchmarks\dnabert2_finetune_t4_seed42\checkpoints\best_model.pt --benchmark-manifest outputs\benchmarks\dnabert2_finetune_t4_seed42\manifest.json --output outputs\annotations\pAN1717_cyan_dnabert2_annotated.gb --predictions-csv outputs\annotations\pAN1717_cyan_dnabert2_predictions.csv --manifest outputs\annotations\pAN1717_cyan_dnabert2_manifest.json --clean-output --open-output-folder
```

`--clean-output` removes only this run's three target files before rerunning: the annotated GenBank, predictions CSV, and manifest JSON. `--open-output-folder` opens the output folder after a successful run so users can immediately view or copy/download the results.

Equivalent one-line command for Windows Command Prompt without auto-opening the folder:

```bat
seqtrainer annotate promoters C:\Users\Sgoff\Downloads\pAN1717_cyan.gb --model-family dnabert2 --checkpoint outputs\benchmarks\dnabert2_finetune_t4_seed42\checkpoints\best_model.pt --benchmark-manifest outputs\benchmarks\dnabert2_finetune_t4_seed42\manifest.json --step-size 300 --no-scan-both-strands --output outputs\annotations\pAN1717_cyan_dnabert2_smoke_annotated.gb --predictions-csv outputs\annotations\pAN1717_cyan_dnabert2_smoke_predictions.csv --manifest outputs\annotations\pAN1717_cyan_dnabert2_smoke_manifest.json
```

Observed sample output on `pAN1717_cyan.gb`:

```text
threshold = 0.750244140625
threshold_source = benchmark_manifest
window_size = 300
step_size = 300
scan_both_strands = false
total_windows_scanned = 20
predicted_promoters_added = 2
```

Predicted promoter regions from that quick check:

| Region | Location | Strand | Score |
| --- | ---: | ---: | ---: |
| `predicted_promoter_0` | `900-1200` | `+` | `0.769360` |
| `predicted_promoter_1` | `2700-3000` | `+` | `0.755906` |

## 7. Run A Denser Real DNABERT2 Annotation

For a more complete scan, use a smaller step size and both strands:

```powershell
seqtrainer annotate promoters C:\Users\Sgoff\Downloads\pAN1717_cyan.gb `
  --model-family dnabert2 `
  --checkpoint outputs\benchmarks\dnabert2_finetune_t4_seed42\checkpoints\best_model.pt `
  --benchmark-manifest outputs\benchmarks\dnabert2_finetune_t4_seed42\manifest.json `
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
