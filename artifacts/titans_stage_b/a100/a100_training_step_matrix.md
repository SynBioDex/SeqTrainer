# Stage B two-segment outer-training matrix

Device: `NVIDIA A100-SXM4-40GB`; scale: `a100_pilot`.

| Variant | Available | Median step s | Tokens/s | CUDA peak allocated | State dtype |
| --- | --- | ---: | ---: | ---: | --- |
| reference_fp32 | yes | 1.055266 | 60.65 | 1640768512 | float32 |
| exact_fp32 | yes | 1.007188 | 63.54 | 1641346048 | float32 |
| exact_sdpa_fp32 | yes | 0.983561 | 65.07 | 1641346048 | float32 |
| exact_sdpa_bfloat16 | yes | 1.020431 | 62.72 | 1655469056 | float32 |
| exact_sdpa_float16 | yes | 1.046340 | 61.17 | 1654957056 | float32 |

Each step processes two 32-token segments. Segment two reads the differentiable memory state written by segment one before backward and AdamW update.
