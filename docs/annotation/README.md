# Annotation Quickstart

This is the beginner/local-system guide for running the SeqTrainer promoter
annotation MVP and opening its output files.

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

## 5. Real Model Annotation Later

For real model annotation, use a completed benchmark checkpoint and manifest:

```powershell
seqtrainer annotate promoters C:\Users\Sgoff\Downloads\pAN1717_cyan.gb `
  --model-family dnabert2 `
  --checkpoint outputs\benchmarks\dnabert2_finetune\checkpoints\best.pt `
  --benchmark-manifest outputs\benchmarks\dnabert2_finetune\manifest.json `
  --output outputs\annotations\pAN1717_cyan_dnabert2_annotated.gb `
  --predictions-csv outputs\annotations\pAN1717_cyan_dnabert2_predictions.csv `
  --manifest outputs\annotations\pAN1717_cyan_dnabert2_manifest.json
```

Use `--model-family cnn_v2` for a CNN-v2 checkpoint.

The `dummy` predictor is only for checking that files are produced correctly.
Only DNABERT2/CNN-v2 runs with real checkpoints should be used for scientific
interpretation.

