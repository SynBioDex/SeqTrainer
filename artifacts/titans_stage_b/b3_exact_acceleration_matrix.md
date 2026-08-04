# B3 exact acceleration matrix

Host: `macOS-15.7.4-x86_64-i386-64bit`; PyTorch `2.2.2`.

| Scale | Geometry | Device | Exact | Reference s | Exact s | Speedup |
| --- | --- | --- | --- | ---: | ---: | ---: |
| debug | 2x d=64 | CPU x86_64 | True | 1.039259 | 0.843553 | 1.232x |
| nimble | 4x d=256 | CPU x86_64 | True | 1.772608 | 2.160158 | 0.821x |
| a100_pilot | 8x d=384 | CPU x86_64 | unavailable | - | - | - |

Unavailable reason: named Colab Pro A100 environment is unavailable in this execution


The functional-loop path preserves evolving gradients and token order. A speedup below 1x is reported as a regression, not an acceleration claim.
