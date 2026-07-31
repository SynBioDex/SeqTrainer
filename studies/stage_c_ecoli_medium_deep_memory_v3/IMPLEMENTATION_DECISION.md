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
