# B6 attention backend evidence

| Evidence | FP64 oracle | FP32 numerical |
| --- | ---: | ---: |
| Sequence max abs error | 0.000e+00 | 0.000e+00 |
| Input-gradient max abs error | 0.000e+00 | 0.000e+00 |
| Persistent-gradient max abs error | 0.000e+00 | 0.000e+00 |

Causal future-prefix error: `0.000e+00`.

| CPU attention path | Tokens/s |
| --- | ---: |
| MultiheadAttention | 10355.89 |
| Functional SDPA | 11130.96 |

BF16 behavioral status: `behavioral_parity_not_numerical_parity`.
FP16 behavioral status: `unavailable`.
Flash exact-mask probe: `False` — unavailable: no CUDA device attached

The JSON stores full state/surprise/attention-gradient errors, raw timing samples, mask counts, mixed-precision state dtypes, and hardware provenance.
