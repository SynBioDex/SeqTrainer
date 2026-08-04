# Exact-scan feasibility for the nonlinear Titans memory update

## Decision

**Classification: exact only under a restricted affine-gradient formulation.**

No compact exact associative scan is established for the implemented nonlinear
MLP update. `exact_scan` therefore remains unavailable. A frozen-gradient or
stale-window construction changes the implemented recurrence and belongs only
to the explicitly approximate B5 backend.

## Implemented recurrence

For fast weights `M_i` and surprise/momentum state `S_i`, Stage A computes

```text
u_i(M_(i-1)) = grad_M 1/2 ||M_(i-1)(k_i) - v_i||^2
S_i = eta_i S_(i-1) - theta_i u_i(M_(i-1))
M_i = (1 - alpha_i) M_(i-1) + S_i
```

The important dependency is `u_i(M_(i-1))`: token `i` differentiates the
associative loss at the fast weights produced by tokens `1..i-1`. The current
`FunctionalNeuralMemory.update_segment` enforces exactly this order by passing
each replacement state into `update_one`.

## Associative-scan condition

A useful fixed-size scan representation exists when every token induces a map
from a closed family with an associative composition. For an affine state map

```text
T_i(z) = A_i z + b_i
T_j compose T_i = (A_j A_i, A_j b_i + b_j),
```

the pair composition is associative. Parallel prefix scan can therefore recover
all states when `A_i` and `b_i` depend only on token-local inputs and gates, not
on an unknown prefix state.

### Restricted scalar linear memory

For `M(k)=w k`, squared loss gives

```text
u_i(w) = (w k_i - v_i) k_i = k_i^2 w - v_i k_i,
```

which is affine in `w`. With `z=[w,S]`, one token has an exact two-dimensional
affine map. `compose_affine` in `scan_feasibility.py` implements the closed
operator. The FP64 harness verifies sequential versus composed output, final
`M`, surprise, and gradients, plus both groupings of three maps, to `1e-12`.

This proves a restricted case consistent with the paper's linear-memory
matrix-product discussion. It does not prove the implemented MLP recurrence.

## Why the generic MLP is not that scan

For a multilayer MLP, weight gradients contain activations and activation
derivatives evaluated at the entering weights. Those terms are nonlinear
functions of `M_(i-1)`. A token-local `(A_i,b_i)` cannot be constructed before
the prefix state is known. Function composition is mathematically associative
in the abstract, but carrying an arbitrary composed nonlinear function is not
a fixed-size tensor scan or an acceleration of this implementation.

Freezing every `u_i` at the chunk-start state makes the later recurrence affine
in the precomputed gradients, but then

```text
u_i(M_chunk_start) != u_i(M_(i-1))
```

in general. It is an approximation, not an exact transformation.

## Numerical counterexample

`run_scan_feasibility_harness` constructs matched two-layer FP64 memories. One
path evaluates all three gradients at the evolving states; the other freezes
them at the chunk-start state. With seed 20260727 it records nonzero divergence
for retrieval output, final fast weights, surprise, and outer gradients. The
result is written to:

```text
artifacts/titans_stage_b/b4_exact_scan_feasibility.json
```

Reproduce:

```bash
PYTHONPATH=src .venv/bin/python -c \
  'from seqtrainer.torch.titans_paper_mac_stage_b import run_scan_feasibility_harness; print(run_scan_feasibility_harness().to_dict())'
```

## Conditions that could change the decision

`exact_scan` may be reconsidered only if one of these is adopted explicitly:

1. restrict neural memory so the associative-loss gradient is affine in a
   fixed-size state (for example, the proved linear case);
2. derive a different exact closed family for the full chosen MLP and verify
   composition, outputs, state, surprise, and outer gradients;
3. change the recurrence definition itself, in which case it is a new backend
   and cannot claim Stage A semantic parity.

Until then, the exact sequential recurrence remains the default oracle,
`exact_accelerated` may optimize its execution without reordering gradients,
and stale-window scans are approximate research ablations only.
