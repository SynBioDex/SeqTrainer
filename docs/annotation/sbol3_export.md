# SBOL3 Export

SeqTrainer writes a validated SBOL3 representation alongside the preserved GenBank output. The export contains one DNA `Component`, one complete `Sequence`, original GenBank features where their mapping is unambiguous, deposited promoter features, predicted promoter `SequenceFeature`s, and one provenance `Activity` for the inference run.

Install the optional dependency group:

```powershell
python -m pip install -e ".[annotation]"
```

The initial stable serialization is N-Triples (`.nt`). Coordinates are converted from Biopython 0-based/end-exclusive to SBOL3 1-based/inclusive. A reverse-strand feature receives reverse-complement orientation. A circular-origin feature is represented as multiple ordered ranges, each bounded by the complete sequence length.

For graphical tools that expect SBOL2 RDF/XML, request a second output with
`--sbol2-output`. This exporter creates a linked SBOL2 `Component` for every
source, deposited, and predicted feature, which allows SBOLCanvas to draw
feature glyphs on the plasmid backbone. The SBOL3 `.nt` file remains the
canonical data-exchange output.

### SBOLCanvas workflow

Upload the SBOL2 `.rdf` file through `File -> Import` in SBOLCanvas. Do not
upload the SBOL3 `.nt` file: it is a different serialization intended for
machine exchange.

SBOLCanvas expects every visual child component to have a Sequence Ontology
role. SeqTrainer therefore uses the generic `SO:0000110` (`sequence_feature`)
role when a source GenBank feature has no narrower mapping. This preserves the
source feature rather than dropping it solely for graphical compatibility.

Canvas may warn that individual components do not have sequences. Those
warnings are non-blocking because the parent plasmid component has the full
sequence and all child features have explicit coordinate ranges. Per-feature
subsequences are intentionally not created yet.

Every export is validated before writing and read back into a fresh `sbol3.Document`. Validation diagnostics are saved in `sbol_validation.json`; an invalid document fails the command rather than being presented as SBOL3-compliant.

## Command-line usage

The SBOL3 file is an optional companion to the annotated GenBank output. Run the
annotation command with `--sbol-output`:

```powershell
seqtrainer annotate promoters C:\Users\Sgoff\Downloads\pAN1717_cyan.gb --model-family dummy --threshold 0.80 --window-size 300 --step-size 25 --scan-both-strands --output outputs\annotations\pAN1717_cyan_annotated.gb --predictions-csv outputs\annotations\pAN1717_cyan_predictions.csv --manifest outputs\annotations\pAN1717_cyan_manifest.json --sbol-output outputs\annotations\pAN1717_cyan.nt --sbol2-output outputs\annotations\pAN1717_cyan_sbol2.rdf
```

For a real model, replace `dummy` and provide the checkpoint and benchmark
manifest as described in the annotation README. The export contains the full
DNA sequence, mapped original GenBank features, deposited promoter labels when
evaluation is enabled, predicted promoter `SequenceFeature`s, and an
`Activity` describing the SeqTrainer inference run. The GenBank file remains
the primary visual output; the `.nt` file is the machine-readable SBOL3 result
and the `.rdf` file is the SBOL2 RDF/XML compatibility output for SBOLCanvas.
