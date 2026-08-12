# Run Promoter Annotation

This guide explains how to run SeqTrainer promoter annotation locally on a
GenBank plasmid file. It covers environment setup, a dummy smoke test, the
trained DNABERT2 workflow, optional evaluation against existing annotations,
SBOL output, and cleanup.

The annotation workflow does not train a model. It loads a previously trained
checkpoint, scans the plasmid in fixed-length DNA windows, applies the
validation-selected threshold from the benchmark manifest, merges passing
windows, and writes annotated files.

## 1. Set Up The Repository

Open Anaconda Prompt or a VS Code terminal:

~~~powershell
cd C:\Users\Sgoff\MYfile\Desktop\PYThh\SeqTrainer
~~~

SeqTrainer requires Python 3.9 or newer. A clean environment is recommended:

~~~powershell
conda create -n seqtrainer-annotation python=3.11 -y
conda activate seqtrainer-annotation
python -m pip install --upgrade pip
python -m pip install -e ".[annotation,torch]"
~~~

The annotation extra installs Biopython, SBOL3, and Tyto. The Torch extra
installs PyTorch, Transformers, and Einops for DNABERT2. Core dependencies
include pandas, NumPy, scikit-learn, RDFLib, SBOL2, and requests.

Verify the environment:

~~~powershell
python -c "import Bio, torch, transformers, seqtrainer; print('SeqTrainer environment OK')"
seqtrainer annotate promoters --help
~~~

If you already use the Anaconda installation that owns seqtrainer.exe:

~~~powershell
C:\Users\Sgoff\anaconda3\python.exe -m pip install -e ".[annotation,torch]"
~~~

### General repository installation

The root README describes the core editable install:

~~~powershell
python -m pip install -e .
~~~

The annotation setup uses the annotation and torch extras, which already
include the core dependencies. Install other extras only for the corresponding
workflows:

~~~powershell
python -m pip install -e ".[keras]"
python -m pip install -e ".[gnn]"
python -m pip install -e ".[dev]"
~~~

The developer extra is useful when contributing or running the test suite:

~~~powershell
python -m pytest -q
ruff check .
~~~

Record the exact source version before a reproducible run:

~~~powershell
git rev-parse HEAD
~~~

The bundled promoter benchmark CSVs are stored in a ZIP archive. Extract them
only when running or reproducing benchmark commands. An annotation run with
an existing model bundle does not need to extract them:

~~~powershell
python -m zipfile -e data/data_DNABERT/promoter_classification_DNABERT.zip data/promoter_classification
~~~

Basic repository smoke commands are:

~~~powershell
seqtrainer sparql prefixes
seqtrainer inspect-sbol data\sbol_data\sample_design_0.xml
seqtrainer build-dataset data\sbol_data\sample_design_0.xml
~~~

## 2. Check The Input Plasmid

The input must be a valid GenBank file:

~~~powershell
Test-Path C:\Users\Sgoff\Downloads\pAN1717_cyan.gb
~~~

The command should print True. Existing genes, CDSs, promoters, and other
GenBank features are preserved by default.

### Using A Different Scientist's Plasmid

The plasmid does not need to be copied into the repository. A scientist can
keep the file anywhere on their computer and pass its full path:

~~~powershell
seqtrainer annotate promoters C:\Users\Scientist\Downloads\my_plasmid.gb --model-family dnabert2 --model-bundle outputs\models\dnabert2_kaggle_best --step-size 25 --scan-both-strands --output outputs\annotations\my_plasmid\annotated.gb --predictions-csv outputs\annotations\my_plasmid\predictions.csv --manifest outputs\annotations\my_plasmid\manifest.json --clean-output --open-output-folder
~~~

For a plasmid downloaded to the current Windows user's Downloads folder, use
this shorter version. Replace only `my_plasmid.gb` with the downloaded file's
actual name:

~~~powershell
seqtrainer annotate promoters "$env:USERPROFILE\Downloads\my_plasmid.gb" --model-family dnabert2 --model-bundle "outputs\models\dnabert2_kaggle_best" --step-size 25 --scan-both-strands --output "outputs\annotations\my_plasmid\annotated.gb" --predictions-csv "outputs\annotations\my_plasmid\predictions.csv" --manifest "outputs\annotations\my_plasmid\manifest.json" --clean-output --open-output-folder
~~~

`$env:USERPROFILE` automatically means the current user's Windows folder, so
the command works without replacing a username. In Command Prompt, use the
full `C:\Users\YourName\Downloads\...` path instead.

For a local, organized copy, use the repository's ignored external folder:

~~~powershell
New-Item -ItemType Directory -Force external\plasmids\my_plasmid
Copy-Item C:\Users\Scientist\Downloads\my_plasmid.gb external\plasmids\my_plasmid\input.gb
~~~

Then use:

~~~powershell
seqtrainer annotate promoters external\plasmids\my_plasmid\input.gb --model-family dnabert2 --model-bundle outputs\models\dnabert2_kaggle_best --output outputs\annotations\my_plasmid\annotated.gb --predictions-csv outputs\annotations\my_plasmid\predictions.csv --manifest outputs\annotations\my_plasmid\manifest.json --clean-output --open-output-folder
~~~

Do not commit private, unpublished, or very large plasmid files to Git.
Keep those files outside the repository or in a local ignored data directory.
For a small public example, a contributor may add the input through the normal
repository review process, together with its source and license information.

Use a separate output folder for each plasmid. This prevents one scientist's
results from overwriting another scientist's results.

### When Evaluation Is Possible

Prediction works with any valid GenBank plasmid. Optional evaluation requires
the input GenBank file to contain recognizable promoter annotations. If the
new plasmid is unannotated, SeqTrainer can still produce predictions, but it
cannot calculate recovery, overlap, or comparison metrics against known
promoters.

For evaluation, the scientist should provide an annotated GenBank file with
promoter features or explicit promoter labels. The current --gold-csv option
chooses where extracted gold annotations are written; it is not an external
gold-input option.

## 3. Run A Dummy Smoke Test

Dummy mode checks GenBank parsing, DNA window generation, strand handling,
output writing, and SBOL export. It does not produce biological evidence.

Use one line in Windows PowerShell or Command Prompt:

~~~powershell
seqtrainer annotate promoters C:\Users\Sgoff\Downloads\pAN1717_cyan.gb --model-family dummy --threshold 0.80 --window-size 300 --step-size 300 --no-scan-both-strands --output outputs\annotations\pAN1717_smoke\annotated.gb --predictions-csv outputs\annotations\pAN1717_smoke\predictions.csv --manifest outputs\annotations\pAN1717_smoke\manifest.json --sbol-output outputs\annotations\pAN1717_smoke\annotated.nt --sbol2-output outputs\annotations\pAN1717_smoke\annotated.rdf --clean-output --open-output-folder
~~~

## 4. Prepare The Kaggle DNABERT2 Model Bundle

Use the completed Kaggle full-fine-tuning archive as the annotation model. It
contains the complete fine-tuned classifier checkpoint and its matching
benchmark manifest. Prepare it with the repository helper:

~~~powershell
.\scripts\prepare_dnabert2_annotation_bundle.ps1 `
  -Archive "C:\Users\Scientist\Downloads\dnabert2_final_training_t4_seed42.zip"
~~~

If Windows blocks local PowerShell scripts, run the same helper with a
temporary policy bypass:

~~~powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\prepare_dnabert2_annotation_bundle.ps1 -Archive "C:\Users\Scientist\Downloads\dnabert2_final_training_t4_seed42.zip"
~~~

The helper creates this local, Git-ignored directory:

~~~text
outputs\models\dnabert2_kaggle_best\
|-- manifest.json
|-- metrics.csv
|-- metrics.json
|-- history.csv
|-- input_split_audit.json
|-- config.json
|-- checkpoints\
    |-- best_model.pt
~~~

`best_model.pt` is the full fine-tuned DNABERT2 model. For every new GenBank
sequence, annotation tokenizes each sliding window, runs the DNABERT2 encoder
and trained classifier head, applies the validation-selected threshold from
`manifest.json`, and merges passing windows into predicted promoter regions.
The Kaggle bundle threshold is `0.677001953125`; do not tune it on the new
plasmid.

Keep the trained checkpoint and matching benchmark manifest together:

~~~text
outputs\models\dnabert2_kaggle_best\
|-- manifest.json
|-- checkpoints\
|-- best_model.pt
~~~

The original benchmark output directory can also be used when it contains the
same Kaggle checkpoint and manifest:

~~~text
outputs\models\dnabert2_kaggle_best\
|-- manifest.json
|-- checkpoints\
|-- best_model.pt
~~~

The selected Kaggle bundle is stored here after preparation:

~~~text
outputs\models\dnabert2_kaggle_best\
|-- manifest.json
|-- checkpoints\
|-- best_model.pt
~~~

Check the bundle before starting annotation:

~~~powershell
Test-Path "outputs\models\dnabert2_kaggle_best\manifest.json"
Test-Path "outputs\models\dnabert2_kaggle_best\checkpoints\best_model.pt"
~~~

Both commands must print `True`. The bundle path is valid only after it has
actually been created and populated with these two files. Relative paths are
resolved from the repository directory shown by `Get-Location`.

The checkpoint contains learned model weights. The manifest records the model
preprocessing, sequence length, and validation-selected threshold. Both files
must come from the same training run. Large checkpoints should remain in local
storage, Google Drive, or HPC storage rather than being committed to Git.

## 5. Run Real DNABERT2 Annotation

The model-bundle option automatically finds the checkpoint and manifest:

~~~powershell
seqtrainer annotate promoters C:\Users\Sgoff\Downloads\pAN1717_cyan.gb --model-family dnabert2 --model-bundle outputs\models\dnabert2_kaggle_best --step-size 25 --scan-both-strands --output outputs\annotations\pAN1717_dnabert2\annotated.gb --predictions-csv outputs\annotations\pAN1717_dnabert2\predictions.csv --manifest outputs\annotations\pAN1717_dnabert2\manifest.json --sbol-output outputs\annotations\pAN1717_dnabert2\annotated.nt --sbol2-output outputs\annotations\pAN1717_dnabert2\annotated.rdf --clean-output --open-output-folder
~~~

Do not add --threshold for this run. The threshold comes from the validation
benchmark manifest and is not tuned on the plasmid.

The equivalent explicit-path form is:

~~~powershell
seqtrainer annotate promoters "C:\Users\Sgoff\Downloads\pAN1717_cyan.gb" --model-family dnabert2 --checkpoint "outputs\models\dnabert2_kaggle_best\checkpoints\best_model.pt" --benchmark-manifest "outputs\models\dnabert2_kaggle_best\manifest.json" --step-size 25 --scan-both-strands --output "outputs\annotations\pAN1717_dnabert2\annotated.gb" --predictions-csv "outputs\annotations\pAN1717_dnabert2\predictions.csv" --manifest "outputs\annotations\pAN1717_dnabert2\manifest.json" --sbol-output "outputs\annotations\pAN1717_dnabert2\annotated.nt" --sbol2-output "outputs\annotations\pAN1717_dnabert2\annotated.rdf" --clean-output
~~~

## 6. Optional Evaluation Against Existing Annotations

Evaluation requires a GenBank file containing explicit promoter annotations.
Add an evaluation directory:

~~~powershell
seqtrainer annotate promoters "C:\Users\Sgoff\Downloads\pAN1717_cyan.gb" --model-family dnabert2 --model-bundle "outputs\models\dnabert2_kaggle_best" --step-size 25 --scan-both-strands --evaluation-dir "outputs\annotations\pAN1717_dnabert2\evaluation" --promoter-label-mode labelled --annotation-completeness partial --iou-threshold 0.50 --output "outputs\annotations\pAN1717_dnabert2\annotated.gb" --predictions-csv "outputs\annotations\pAN1717_dnabert2\predictions.csv" --manifest "outputs\annotations\pAN1717_dnabert2\manifest.json" --sbol-output "outputs\annotations\pAN1717_dnabert2\annotated.nt" --sbol2-output "outputs\annotations\pAN1717_dnabert2\annotated.rdf" --clean-output --open-output-folder
~~~

The evaluator recognizes a gold promoter when:

- the feature type is promoter;
- a regulatory feature has regulatory_class=promoter;
- the feature has the promoter ontology identifier SO:0000167;
- in labelled mode, a label, name, or note contains the standalone word
  promoter.

Use --promoter-label-mode strict for only the strongest explicit rules.
The labelled mode also accepts explicit promoter words in qualifiers.

### Annotation completeness

| Value | Meaning |
| --- | --- |
| partial | Some promoters are labelled, but unlabelled regions are not confirmed negatives. Recommended for most deposited plasmids. |
| verified_complete | Every promoter and negative region is believed to be annotated. Full classification metrics are allowed. |
| unknown | Completeness has not been assessed. Conservative recovery results are reported. |

Use partial unless you know that the GenBank annotations are complete. With
partial annotations, recovery and overlap are reported instead of presenting
ordinary accuracy or MCC as definitive.

## 7. Evaluation Outputs

Evaluation creates a folder such as:

~~~text
outputs\annotations\pAN1717_dnabert2\evaluation\
~~~

- gold_promoters.csv: promoters extracted from the original GenBank file;
- window_predictions.csv: every scanned window, score, and gold label;
- merged_predictions.csv: merged predicted regions compared with gold;
- promoter_matches.csv: prediction-to-gold matching table;
- metrics.csv: evaluation metrics in table form;
- metrics.json: the same metrics in machine-readable JSON;
- sbol_validation.json: SBOL validation results when SBOL output is requested.

The window evaluator labels a window positive when its centre lies inside a
same-strand gold promoter. The merged evaluator matches predicted regions to
gold promoters by strand and intersection-over-union.

--iou-threshold 0.50 requires at least 50 percent overlap for a primary match.
This is separate from the DNABERT2 probability threshold.

--gold-csv controls where the extracted gold table is written. It is not
currently an external gold-input option.

## 8. SBOL Outputs

annotated.gb is the primary sequence result with predicted promoter features.

annotated.nt is SBOL3 N-Triples for SBOL-aware data exchange.

annotated.rdf is SBOL2 RDF/XML with linked components and sequence annotations
for tools such as SBOLCanvas.

For SBOLCanvas, import the `.rdf` file through `File -> Import`; do not upload
the SBOL3 `.nt` file. SeqTrainer assigns the generic Sequence Ontology
`sequence_feature` role to GenBank features that have no narrower role, so
untyped source features remain compatible with Canvas glyph rendering.

Canvas can report warnings such as `Component ... does not have a sequence`.
These are non-blocking: the complete plasmid sequence and each feature's
coordinate range are present, so the design map and predicted promoter
locations remain usable. Writing each feature's individual DNA subsequence is
a future enhancement, not required for annotation or Canvas import.

Open the output folder:

~~~powershell
explorer outputs\annotations\pAN1717_dnabert2
~~~

SBOL is structured data, not itself a flowchart. SBOLCanvas reads the RDF/XML
file and renders a visual design map.

## 9. Command Option Reference

The basic command format is:

~~~powershell
seqtrainer annotate promoters INPUT_FILE OPTION VALUE
~~~

Use one space between an option and its value. Boolean options do not take a
value. Paths may be relative to the repository or absolute. Put quotation
marks around paths containing spaces.

| Option | Format | Purpose |
| --- | --- | --- |
| input file | INPUT_FILE | GenBank plasmid to scan; required |
| --model-family | dummy, dnabert2, or cnn_v2 | Selects the predictor; dummy is for smoke tests |
| --model-bundle | PATH | Folder containing the checkpoint and matching manifest |
| --checkpoint | PATH | Explicit trained checkpoint path |
| --benchmark-manifest | PATH | Explicit benchmark manifest path |
| --threshold | FLOAT | Manual probability threshold; mainly for dummy or controlled tests |
| --window-size | INTEGER | DNA window length; omit for DNABERT2 to use the manifest |
| --step-size | INTEGER | Distance between windows; 25 is the normal detailed scan |
| --scan-both-strands | flag | Scans both plus and minus strands |
| --no-scan-both-strands | flag | Scans only the forward strand |
| --merge-distance | INTEGER | Maximum gap for merging positive windows; default is 25 |
| --min-score | FLOAT | Optional additional minimum score filter |
| --output | PATH | Annotated GenBank output path |
| --predictions-csv | PATH | Per-window prediction table path |
| --manifest | PATH | Annotation-run manifest output path |
| --preserve-existing-features | flag | Keeps original GenBank features; default behavior |
| --no-preserve-existing-features | flag | Removes original features from the output copy |
| --clean-output | flag | Removes old outputs for this run before writing new files |
| --open-output-folder | flag | Opens the output folder after a successful run |
| --evaluation-dir | PATH | Folder for gold tables, matches, and evaluation metrics |
| --gold-csv | PATH | Output path for extracted gold promoters |
| --promoter-label-mode | labelled or strict | Controls which GenBank features count as gold promoters |
| --annotation-completeness | partial, verified_complete, or unknown | Controls how evaluation metrics are interpreted |
| --iou-threshold | FLOAT | Overlap threshold for a predicted region to match a gold promoter |
| --sbol-output | PATH | SBOL3 N-Triples output path |
| --sbol2-output | PATH | SBOL2 RDF/XML output path for SBOLCanvas |
| --sbol-namespace | URL | Namespace used for generated SBOL identifiers |

For real DNABERT2 runs, prefer --model-bundle and omit --threshold. The
threshold must come from the validation-selected benchmark manifest.

For a quick smoke test, use --model-family dummy, provide --threshold and
--window-size, and use a large step size such as 300.

For a labelled-plasmid comparison, add --evaluation-dir,
--promoter-label-mode labelled, and --annotation-completeness partial unless
the original GenBank annotations are verified complete.

## 10. Repeat Runs And Cleanup

Use --clean-output when repeating the same run. It removes the named primary
files and clears the named evaluation directory so old results are not mixed
with new results.

To remove all local annotation outputs:

~~~powershell
if (Test-Path outputs\annotations) { Remove-Item outputs\annotations -Recurse -Force }
~~~

This does not delete the input GenBank file or the trained model bundle.

## 11. Common Errors

ModuleNotFoundError: No module named Bio:

~~~powershell
python -m pip install -e ".[annotation]"
~~~

DNABERT2 requires torch and transformers:

~~~powershell
python -m pip install -e ".[annotation,torch]"
~~~

Model bundle directory not found means the path passed to --model-bundle is
wrong or the command was launched outside the repository root. Run the Kaggle
bundle preparation script first and use
`outputs\models\dnabert2_kaggle_best`, or provide the explicit checkpoint and
manifest paths shown above. Model bundle is incomplete
when the folder does not contain both `checkpoints\best_model.pt` and
`manifest.json`.

Use one command line on Windows. PowerShell uses a backtick for continuation,
while Command Prompt does not. Copying Linux multiline commands can cause
options such as --threshold to be interpreted as separate commands.

## 12. Future Command Updates

New promoter-annotation commands, required options, environment changes, and
model-bundle conventions should be added to this file. Keep the root README
focused on the repository overview and use this guide as the detailed runbook.

When a future model is added, document:

- its model-family value;
- its checkpoint and manifest bundle layout;
- required Python extras;
- whether its threshold comes from validation or is externally supplied;
- compatible input formats;
- output and evaluation files;
- a complete sample command;
- the exact commit or release used for the run.
