# B7 long-context synthetic validation

> Synthetic system validation only; no genomic training or biology-performance claim.

## MacBook CPU

| Scale | Variant | Tokens/s | Segment latency s | State/resume s | State bytes | Final state rel L2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| debug_d64 | reference | 198.28 | 0.1531 | 0.00071 | 66560 | 0.000e+00 |
| debug_d64 | exact_accelerated | 263.11 | 0.1127 | 0.00071 | 66560 | 0.000e+00 |
| debug_d64 | approximate_w2 | 233.46 | 0.1282 | 0.00077 | 66560 | 6.198e-04 |
| debug_d64 | approximate_w4 | 247.37 | 0.1226 | 0.00054 | 66560 | 1.024e-03 |
| debug_d64 | approximate_w8 | 220.78 | 0.1368 | 0.00070 | 66560 | 1.720e-03 |
| debug_d64 | approximate_w16 | 173.18 | 0.1760 | 0.00068 | 66560 | 2.352e-03 |
| debug_d64 | approximate_w32 | 112.69 | 0.2705 | 0.00065 | 66560 | 3.658e-03 |
| debug_d64 | causal_convolution | 241.56 | 0.1241 | 0.00068 | 66560 | 3.223e-04 |
| debug_d64 | sdpa | 258.57 | 0.1169 | 0.00055 | 66560 | 0.000e+00 |
| debug_d64 | frozen_memory | 1527.48 | 0.0151 | 0.00080 | 66560 | 1.255e-01 |
| debug_d64 | no_memory | 1847.68 | 0.0115 | 0.00067 | 66560 | 1.255e-01 |
| debug_d128 | reference | 146.64 | 0.2075 | 0.00132 | 264192 | 0.000e+00 |
| debug_d128 | exact_accelerated | 184.65 | 0.1604 | 0.00091 | 264192 | 0.000e+00 |
| debug_d128 | approximate_w2 | 163.46 | 0.1795 | 0.00102 | 264192 | 7.519e-04 |
| debug_d128 | approximate_w4 | 155.78 | 0.1916 | 0.00128 | 264192 | 1.169e-03 |
| debug_d128 | approximate_w8 | 132.78 | 0.2248 | 0.00151 | 264192 | 1.798e-03 |
| debug_d128 | approximate_w16 | 110.20 | 0.2712 | 0.00181 | 264192 | 2.612e-03 |
| debug_d128 | approximate_w32 | 69.72 | 0.4365 | 0.00114 | 264192 | 3.703e-03 |
| debug_d128 | causal_convolution | 169.29 | 0.1742 | 0.00137 | 264192 | 3.544e-04 |
| debug_d128 | sdpa | 183.13 | 0.1620 | 0.00102 | 264192 | 0.000e+00 |
| debug_d128 | frozen_memory | 1309.74 | 0.0140 | 0.00127 | 264192 | 1.068e-01 |
| debug_d128 | no_memory | 1289.77 | 0.0151 | 0.00106 | 264192 | 1.068e-01 |
| nimble | reference | 53.11 | 0.5684 | 0.00748 | 2105344 | 0.000e+00 |
| nimble | exact_accelerated | 54.12 | 0.5322 | 0.00786 | 2105344 | 0.000e+00 |
| nimble | approximate_w2 | 51.65 | 0.5658 | 0.00636 | 2105344 | 9.067e-04 |
| nimble | approximate_w4 | 53.78 | 0.5407 | 0.00562 | 2105344 | 1.476e-03 |
| nimble | approximate_w8 | 35.71 | 0.8146 | 0.01129 | 2105344 | 2.223e-03 |
| nimble | approximate_w16 | 22.76 | 1.3465 | 0.00668 | 2105344 | 3.185e-03 |
| nimble | approximate_w32 | 21.97 | 1.3429 | 0.00763 | 2105344 | 4.534e-03 |
| nimble | causal_convolution | 73.79 | 0.3901 | 0.00529 | 2105344 | 3.989e-04 |
| nimble | sdpa | 78.61 | 0.3689 | 0.00483 | 2105344 | 0.000e+00 |
| nimble | frozen_memory | 486.54 | 0.0309 | 0.00521 | 2105344 | 9.998e-02 |
| nimble | no_memory | 480.06 | 0.0286 | 0.00497 | 2105344 | 9.998e-02 |

## A100

- `a100_pilot` unavailable: named Colab Pro A100 environment is unavailable in this execution

## Controlled long recall

| Variant | 64 | 128 | 256 | 512 | Overwrite | Reset | Mean BPB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reference | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.837 |
| exact_accelerated | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.837 |
| approximate_w2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.837 |
| approximate_w4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.837 |
| approximate_w8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.837 |
| approximate_w16 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.837 |
| approximate_w32 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.837 |
| causal_convolution | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.837 |
| sdpa | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.837 |
| frozen_memory | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 4.322 |
| no_memory | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 4.322 |

All backend future-prefix checks passed: **True**.

Independent rerun: `/Users/gonzalovidal/opt/anaconda3/envs/seqtrainer/bin/seqtrainer-titans-stage-b-long-context`

The JSON contains raw segment/resume timings, context-indexed drift, peak memory, gates/updates, hardware separation, and limitations.
