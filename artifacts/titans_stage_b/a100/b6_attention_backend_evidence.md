# B6 attention backend evidence

| Evidence | FP64 oracle | FP32 numerical |
| --- | ---: | ---: |
| Sequence max abs error | 0.000e+00 | 0.000e+00 |
| Input-gradient max abs error | 0.000e+00 | 0.000e+00 |
| Persistent-gradient max abs error | 0.000e+00 | 0.000e+00 |

Causal future-prefix error: `0.000e+00`.

| Attention path | Tokens/s |
| --- | ---: |
| MultiheadAttention | 6367.19 |
| Functional SDPA | 6296.10 |

BF16 behavioral status: `behavioral_parity_not_numerical_parity`.
FP16 behavioral status: `behavioral_parity_not_numerical_parity`.
Flash exact-mask probe: `False` — forced Flash SDP rejected the exact mask: No available kernel. Aborting execution.

The JSON stores full state/surprise/attention-gradient errors, raw timing samples, mask counts, mixed-precision state dtypes, and hardware provenance.
