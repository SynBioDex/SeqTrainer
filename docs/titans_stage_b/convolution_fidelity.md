# B2 causal convolution fidelity note

## Decision

SeqTrainer implements convolution as an **opt-in repository interpretation**, not
as a paper-exact reproduction. `GateBackend.CAUSAL_CONVOLUTION` applies a causal
temporal mixer to the representations that drive the neural-memory update gates.
`GateBackend.TOKEN_WISE` remains the default and invokes the unchanged Stage A
path exactly.

The distinction matters because the Titans paper describes two incomplete and
potentially different ideas. Section 3.2 says chunk-constant update gates make a
global convolution possible, but also says the reported experiments use
token-dependent gates. Section 5.9 reports an ablation "w/o Convolution" without
specifying placement, width, grouping, padding, or its relationship to the
memory gates. Those statements do not determine a unique implementation.

## Exact repository semantics

For integrated segment representations `x[0:32]`, B2 computes

```text
c_t = depthwise_conv1d(left_pad(x), kernel_size=K)_t
[alpha_t, eta_t, theta_t] = sigmoid(W_gate c_t + b_gate)
```

- Placement: immediately before the existing three-logit adaptive gate
  projection in the post-core neural-memory write.
- Padding: `K - 1` zeros on the left and none on the right.
- Grouping: `groups=d_model`, so every channel has its own temporal kernel.
- Default kernel: `K=3`, exposed as `convolution_kernel_size`.
- Initialization: an equal-weight causal moving average (`1/K`).
- Projection: initialized from the Stage A token-wise gate projection.
- Read/write timing: unchanged. All retrievals use incoming `M_(t-1)`; the
  convolution affects only the one post-core write that publishes `M_t`.
- Recurrence: unchanged equations and exact evolving-gradient token order.
- State: no convolution buffer crosses segments. Long-term state remains only
  the stream-local fast weights, surprise, lifecycle, and segment index.

The convolution module is owned by `StageBMACStack` when enabled, so its
depthwise kernel and gate projection participate in ordinary outer training and
receive gradients. Standalone dispatch requires the caller to pass the module
explicitly; this prevents an unregistered parameter set from being constructed
inside an execution call.

## Causality argument

At time `t`, the left-padded depthwise kernel uses only
`x[t-K+1], ..., x[t]`. A perturbation at future position `j` therefore cannot
alter gates before `j`. The current segment output is also unaffected by these
gates because MAC reads and integrates before it writes; the newly gated state
is visible only to the next segment. Executable tests perturb a future token and
require exactly zero prefix error in both the gates and current outputs.

## Evidence and limitations

`b2_convolution_comparison.json` contains a seeded, matched FP64 comparison of:

- token-wise versus convolutional `alpha`, `eta`, and `theta` statistics;
- memory update norms and the resulting fast-weight difference;
- causal prefix errors;
- input, convolution-kernel, and gate-projection gradient norms; and
- the unchanged Stage A adaptive/frozen/no-memory mechanism evidence.

The comparison demonstrates that the opt-in path is active, causal, trainable,
and measurably different. It does **not** claim that this small random-segment
mechanism comparison establishes an accuracy benefit, nor that it reconstructs
the unspecified Section 5.9 component. Long-stream task effects belong to B7.

Primary implementation seams:

- `convolution.py`: convolutional gates and exact recurrence adapter.
- `config.py`: explicit feature flag and kernel width.
- `backends.py`: dispatch and conservative compatibility check.
- `stack.py`: registered per-block convolution modules.
- `tests/test_titans_paper_mac_stage_b_convolution.py`: recovery, causality,
  gradients, state timing, and artifact coverage.

