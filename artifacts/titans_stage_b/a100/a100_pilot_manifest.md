# Stage B named-A100 pilot manifest

Commit: `68262e4e4c8e6207c208fc5e7e7b31baa2ab191f`; captured: `2026-07-19T02:50:24.099789+00:00`.

Strict verification: `PASS`.
Selected default: `exact_sdpa_fp32`.

| Check | Result |
| --- | --- |
| Named A100 preflight | PASS |
| Repeated tensor-exact B3 run | PASS |
| Attention parity, exact mask, and causality | PASS |
| BF16/FP16 preserve the FP32 memory island | PASS |
| Forced exact-mask Flash probe attempted | PASS |
| A100 long-stream matrix is finite and causal | PASS |
| Adaptive memory beats controls at 512 tokens | PASS |
| Repeated outer-training steps cross the written state | PASS |
| Target-hardware default uses a 5% exact-backend threshold | PASS |

## SHA-256

- `a100_preflight.json`: `ab9d978937ad7f265dc6582e432841f4f1dfb82e8306d3c4ac8c5234b1ff066e`
- `b3_exact_acceleration_matrix.json`: `635b785dacf06c2ed508a1283d70f8459ba048267e515fae5397f8dcb4c39d44`
- `b6_attention_backend_evidence.json`: `d93ee68662127d4a9c84ac9c1c81cc62f1ebdc4fd066042db18f320556022645`
- `b7_long_context_study.json`: `b5749bd2559047f4dc3600b4412af6a14df7ce2ddba47f53471da210589d2b9a`
- `a100_training_step_matrix.json`: `c8cfb6201b4696657a0c430ad872f6a7a8696674b031bd29dd7116879388f8a3`
