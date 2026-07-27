# Paper-deep neural-memory implementation decision

Date: 2026-07-26

This is a new v2 prerequisite study. It does not edit the frozen
`stage_c_ecoli_escherichia_medium_25m_v1` protocol or reinterpret its prior
runs.

## Architecture

The v2 fast memory is

`M(x) = x + LayerNorm(W_out GELU(W_in x))`,

where `W_in` maps `d` to `4d` and `W_out` maps `4d` to `d`. This is the
paper-default two-linear-layer, four-times-expanded residual memory. “Depth 2”
means these two learned affine transformations; it is not inferred from the
separate key and value projections.

Keys, values, and queries have independent learned projections followed by a
causal depthwise one-dimensional convolution with kernel size four. Query and
key vectors are L2-normalized after convolution. Each stream checkpoint stores
the preceding three raw projection inputs independently for query and
key/value paths, so a segment boundary is not a convolution reset.

## Associative objective and recurrence

The exact policy implements

`ell_t = 1/2 ||M_(t-1)(k_t) - v_t||_2^2`,

`g_t = grad_(M_(t-1)) ell_t`,

`S_t = eta_t elementwise S_(t-1) - theta_t elementwise g_t`,

`M_t = (1 - alpha_t) elementwise M_(t-1) + S_t`.

`S_(t-1)` is the encoded past surprise. The newly computed
`-theta_t elementwise g_t` is momentary surprise. They are not blended before
measurement: training records the RMS of each term, their combined RMS, and
their cosine before committing `S_t`.

The gate heads emit one value for every output channel of each memory layer.
For a weight matrix with shape `(out_channels, in_channels)`, a gate vector is
broadcast as `(out_channels, 1)`. Biases and LayerNorm parameters receive the
corresponding `(out_channels,)` vector. Thus forgetting, momentum, and local
learning can differ by memory layer and output channel while remaining shared
across the incoming coordinates of one neuron.

Initial gate values are `alpha=0.001`, `eta=0.9`, and `theta=0.001`. These are
learned through sigmoid-parameterized heads. Under the code’s equation,
`alpha` is the fraction removed, so `alpha=0.001` retains 99.9% of the previous
fast weight before adding surprise.

## Two preregistered numerical policies

`paper_exact` uses the summed objective above, no inner-gradient conditioning,
no surprise-vector cap, and `theta_max=1`.

`stabilized_rms_v1` is a declared engineering alternative. It uses
`ell_t/d`, so its unconditioned gradient is exactly `g_t/d`, then applies one
global differentiable scale

`c_t = min(1, 10 RMS(M_(t-1)) / RMS(g_t/d))`.

The conditioned gradient is `c_t g_t/d`; `theta_max=0.5`. It also has no
surprise-vector cap. The previous absolute norm-4 cap remains source-compatible
only for legacy v1 configurations and is disabled in every v2 run.

The exact policy is selected for scaling only if its 500-step finite-state and
deterministic-resume gates pass. Otherwise the stabilized policy may be
selected, with every conditioning intervention retained in the evidence.

## Scale decision

The compact T4 model is 4 blocks at `d=128` (2,517,504 parameters for a
4,098-token vocabulary). If compact gates pass, the retargeted medium model is
12 blocks at `d=256`, 8 heads, and 24,785,920 parameters. Its FP32 functional
fast-weight plus surprise state is 50,577,408 bytes per active stream,
excluding autograd history and optimizer state. The previous 11-by-512
geometry is not reused because the four-times-expanded deep memory and
channel-gate heads materially change the parameter budget.
