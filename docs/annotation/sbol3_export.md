# SBOL3 Export

SeqTrainer writes a validated SBOL3 representation alongside the preserved GenBank output. The export contains one DNA `Component`, one complete `Sequence`, original GenBank features where their mapping is unambiguous, deposited promoter features, predicted promoter `SequenceFeature`s, and one provenance `Activity` for the inference run.

Install the optional dependency group:

```powershell
python -m pip install -e ".[annotation]"
```

The initial stable serialization is N-Triples (`.nt`). Coordinates are converted from Biopython 0-based/end-exclusive to SBOL3 1-based/inclusive. A reverse-strand feature receives reverse-complement orientation. A circular-origin feature is represented as multiple ordered ranges, each bounded by the complete sequence length.

Every export is validated before writing and read back into a fresh `sbol3.Document`. Validation diagnostics are saved in `sbol_validation.json`; an invalid document fails the command rather than being presented as SBOL3-compliant.
