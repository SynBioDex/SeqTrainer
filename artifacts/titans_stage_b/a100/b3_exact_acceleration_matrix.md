# B3 exact acceleration matrix

Host: `Linux-6.6.122+-x86_64-with-glibc2.35`; PyTorch `2.11.0+cu128`.

| Scale | Geometry | Device | Exact | Reference s | Exact s | Speedup |
| --- | --- | --- | --- | ---: | ---: | ---: |
| a100_pilot | 8x d=384 | NVIDIA A100-SXM4-40GB | True | 0.306500 | 0.302568 | 1.013x |

The functional-loop path preserves evolving gradients and token order. A speedup below 1x is reported as a regression, not an acceleration claim.
