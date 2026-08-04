# Draft: ambitious A100 deep-memory discovery run

Status: exploratory draft. Do not interpret this as a confirmatory adaptive-memory comparison.

## What the 5M run establishes

The completed `c16_deep_adaptive_5m_paper_exact` run establishes that the exact two-layer paper-style memory can train stably at compact scale. It reached 1.98278 held-out BPB, had finite active memory updates, and passed the controlled immediate and delayed association probes. It does not establish that adaptive memory improves over no memory, that generated genes are functional, or that the model represents taxonomy. The validation and PCA sample also covered too little taxonomic diversity.

The 5M run consumed only about 0.0388% of the approximately 12.87-billion-base training corpus. A whole-corpus epoch is therefore not a realistic target for 171.06 Colab compute units: it would require about 2,575 times the c16 base exposure before accounting for the larger model.

## Efficient allocation of the remaining units

Treat the number shown by Colab as an account balance, not as a portable GPU-hour conversion. The current A100 compute-unit burn rate must be measured in the Colab UI because it varies by runtime and service policy.

Use the then-current balance as follows:

- 10% for A100 capacity, throughput, checkpoint, and resume qualification.
- 75% for the adaptive-only discovery training run.
- 15% held back for periodic validation, generation diagnostics, analysis, failures, and one clean resume.

If `R` is the observed A100 compute-unit burn per hour and `T` is measured valid bases per second for the selected batch, calculate:

`training_hours = 0.75 * remaining_compute_units / R`

`safe_base_budget = 0.85 * training_hours * 3600 * T`

The additional 0.85 factor leaves wall-time margin for validation and checkpoint I/O. Record `R`, `T`, and the resulting budget in the protocol amendment before the long run.

## Model and hardware qualification

Start with the already specified Medium paper-deep model: 12 blocks, width 256, 8 heads, horizon 3, two-layer residual MLP memory, expansion factor 4, normalized projected keys/queries, exact paper recurrence, and FP32 neural-memory state. It is about 24.8M trainable parameters—roughly ten times c16—and its per-stream functional memory state grows substantially because memory scales quadratically with width.

On the A100, test batch sizes 1, 2, 4, and 8 for 20–50 optimizer steps each. Prefer BF16 for the ordinary model/attention path while preserving the memory recurrence and functional state in FP32. Select the largest batch satisfying all of:

- peak allocated GPU memory at or below 80–85%;
- finite forward state, backward gradients, and optimizer state;
- no emergency memory-gradient intervention;
- successful checkpoint reload and exact continuation;
- best valid-bases/second after checkpoint overhead.

Copy the token-stream dataset to Colab local SSD for training and keep authoritative manifests, logs, and checkpoints on Drive. Dataset fingerprints must match before and after staging.

## Training exposure and gates

Do not precommit to “the whole dataset.” Use measured throughput to freeze an attainable first budget, normally 25M if capacity permits. Evaluate at fixed cumulative base counts (for example 5M, 10M, 25M), not only at the end. If the 25M gate is strong and enough units remain, record a dated amendment and resume toward 50M; do not silently change the budget.

Before the long run, repair sampling so both training exposure and held-out evaluation are stratified across clades/accessions rather than selecting the first streams. Freeze the selected validation accessions so c16 and the larger model can be reevaluated on exactly the same panel.

Continue or extend only when:

- all finite-state and resume guards pass;
- held-out BPB improves consistently across the fixed, diverse panel rather than one accession;
- the BPB trajectory has not plateaued;
- generated sequences do not collapse and improve held-out-reference 3–6-mer divergence, GC calibration, and gene/intergenic diagnostics;
- memory traces remain active and controlled association behavior remains positive.

A practical strong signal for this tokenizer is held-out BPB at or below about 1.95 on the frozen diverse panel, but this is an engineering scale gate, not a biological threshold. Stop rather than extend if BPB remains near or above c16 after enough matched exposure, becomes unstable, or generation diagnostics deteriorate.

## How generation evidence changes the run

Run `03h` against c16 first. Compare each sampling temperature to equal-length held-out *E. coli* continuations:

- GC, entropy, homopolymer, ambiguous-base, and unique-6-mer distributions;
- Jensen–Shannon divergence for 1–6-mers;
- six-frame ORF diagnostics;
- Prodigal coding density, genes/10 kb, gene-length, intergenic-length, and complete-call fractions;
- neural-memory telemetry during generation.

Choose the temperature with the lowest high-order k-mer divergence without low-entropy or homopolymer collapse. Freeze that decoding policy for comparisons at every larger-model checkpoint. Poor c16 generation does not by itself falsify the architecture—it can reflect limited exposure—but it supplies a baseline. If the larger model improves BPB but not generation, prioritize data diversity/context or decoding calibration rather than blindly adding parameters.

These tests measure conditional sequence realism. Prodigal predictions are not evidence that genes are expressed or functional, and sequence resemblance is not evidence of safety, fitness, or promoter activity.

## Decision after the adaptive-only discovery

If the larger adaptive model is stable, improves the frozen diverse-panel BPB, and improves preregistered generation diagnostics, it is worth spending a later allocation on a matched no-memory control. Only that matched comparison, uncertainty analysis, and replication can support an adaptive-memory benefit claim. If the adaptive model itself remains poor after the scale gate, stop without spending units on the complete control suite and report the negative exploratory result.
