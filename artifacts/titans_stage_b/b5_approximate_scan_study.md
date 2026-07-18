# B5 approximate-scan speed–fidelity study

> Classification: experimental approximation; never parity-equivalent.

| Window | Speedup | State rel L2 | Surprise rel L2 | Gradient rel L2 | Gradient cosine | Delayed >32 | Eval BPB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1.187 | 7.561e-01 | 9.458e-01 | 1.167e+00 | 0.740979 | 1.000 | 0.049 |
| 4 | 1.107 | 1.110e+01 | 1.211e+01 | 8.096e+02 | 0.298122 | 1.000 | 0.049 |
| 8 | 1.039 | 1.797e+00 | 1.532e+00 | 6.547e+00 | 0.382031 | 1.000 | 0.049 |
| 16 | 0.914 | 1.095e+00 | 8.139e-01 | 1.422e+00 | 0.647026 | 1.000 | 0.049 |
| 32 | 0.690 | 9.850e-01 | 7.203e-01 | 1.603e+00 | 0.609165 | 1.000 | 0.049 |

## Decision

- Default: `reference`.
- Approximate scan: `experimental_only`.
- Promotion allowed: `False`.
- Reason: every supported window changes dense-update state/gradients and no approximation may be described as parity-equivalent; task controls do not exercise dense writes

Fixture limitation: the controlled Stage A fixture has at most one valid memory write in each segment, so within-window staleness may be inactive; dense mechanism drift is decisive

The JSON includes raw timing samples, peak-memory telemetry, gates/update statistics, reference/exact/control metrics, and complete reproducibility metadata.
