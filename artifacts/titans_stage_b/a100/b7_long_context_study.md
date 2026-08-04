# B7 long-context synthetic validation

> Synthetic system validation only; no genomic training or biology-performance claim.

## MacBook CPU

| Scale | Variant | Tokens/s | Segment latency s | State/resume s | State bytes | Final state rel L2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| a100_pilot | reference | 64.39 | 0.4477 | 0.01382 | 9461760 | 0.000e+00 |
| a100_pilot | exact_accelerated | 64.97 | 0.4513 | 0.01235 | 9461760 | 0.000e+00 |
| a100_pilot | approximate_w2 | 65.57 | 0.4499 | 0.01151 | 9461760 | 1.575e-03 |
| a100_pilot | approximate_w4 | 64.01 | 0.4637 | 0.01042 | 9461760 | 2.784e-03 |
| a100_pilot | approximate_w8 | 59.16 | 0.5038 | 0.01045 | 9461760 | 4.210e-03 |
| a100_pilot | approximate_w16 | 50.85 | 0.5913 | 0.01056 | 9461760 | 6.173e-03 |
| a100_pilot | approximate_w32 | 39.79 | 0.7649 | 0.01086 | 9461760 | 8.788e-03 |
| a100_pilot | causal_convolution | 63.57 | 0.4632 | 0.01112 | 9461760 | 6.678e-04 |
| a100_pilot | sdpa | 62.18 | 0.4766 | 0.01104 | 9461760 | 0.000e+00 |
| a100_pilot | frozen_memory | 353.36 | 0.0571 | 0.01019 | 9461760 | 1.770e-01 |
| a100_pilot | no_memory | 364.32 | 0.0534 | 0.01132 | 9461760 | 1.770e-01 |

## A100

- A100 scale measured; see JSON for complete per-variant telemetry.

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
