# Stage C v3 implementation decision

The v3 study keeps training exclusively within broad *E. coli* while using
other *Escherichia* only as held-out outgroups. It selects complete assemblies
from independent train-split ANI99 groups and retains every selected replicon
as an ordered stream.

The stateful-rotation scheduler pauses each accession after 96 contiguous
segments and later resumes at the exact next coordinate with its private
functional-memory state intact. This prevents one chromosome from consuming a
bounded experiment while preserving the long ordered context that motivates
Titans.

E100 is a declared warm start from E25. Slow weights, optimizer, RNG and
cumulative base-indexed learning-rate position continue; stream cursors and
functional memory reset because E100-minus-E25 contains new replicons. This is
not represented as an independent run.

The original v1 and v2 protocols remain frozen. The adaptive-only E25/E100
progression can establish utility and justify controls, but cannot establish a
causal adaptive-memory advantage.

## 2026-07-31 held-out assembly policy amendment

The frozen source dataset contains only two complete *E. coli* ANI99
representatives in its validation partition, fewer than the eight required for
the predeclared accession-level gate. Reducing the gate to two or consuming the
test partition for model selection would invalidate the intended evidence.

Training therefore remains restricted to complete, CheckM-high-quality
assemblies. Validation and test remain in their native, mutually isolated
partitions and retain the same CheckM thresholds, but select one assembly per
ANI99 group with this deterministic priority: complete genome, RefSeq,
representative status, quality score, stream count, accession. A high-quality
chromosome/scaffold is permitted only as an evaluation-only fallback. Every
selected held-out ANI99 group is excluded from all training panels, and each
manifest records the achieved assembly-level counts. This broadens the
assembly-level scope of evaluation; it does not change training exposure or
permit a whole-genome claim for draft-derived evaluation streams.

## 2026-08-01 resumable c16 baseline amendment

The original c16 baseline attempted all 41,615,672 predictable validation
bases in one T4 process. Adaptive evaluation performs the neural-memory inner
update with gradients on every segment, so this is not equivalent to a cheap
forward-only Transformer pass. It also produced no durable cursor or progress
artifact. A Colab disconnect could consequently erase many hours of work.

Evaluation now freezes a deterministic segment plan and periodically writes an
atomic checkpoint containing its next cursor, aggregate metrics, and the
detached per-stream fast weights, surprise momentum, and projection histories.
Resume is accepted only when the checkpoint, dataset, panel, model
configuration, memory mode, limits, and work-plan hashes match. Replayed work
starts from the last durable checkpoint and cannot be counted twice.

Notebook 03j first runs an exploratory T4 pilot capped at 256 segments per
validation accession. Equal segment caps avoid allowing one accession to
dominate this operational check. The pilot is explicitly ineligible for model
selection and cannot substitute for the uncapped baseline. The uncapped run is
reserved for an A100 and remains the evidence associated with
`c16_broad_ecoli_baseline_v1`.
