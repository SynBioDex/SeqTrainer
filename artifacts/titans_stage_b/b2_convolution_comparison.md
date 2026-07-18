# B2 causal convolution comparison

> This is a repository-defined, opt-in interpretation; it is not claimed paper-exact.

| Check | Result |
| --- | ---: |
| Token-wise memory update norm | 3.47222163 |
| Convolutional memory update norm | 3.38081651 |
| Final fast-weight max difference | 0.38647969 |
| Gate prefix maximum error | 0.000e+00 |
| Current-output prefix maximum error | 0.000e+00 |

## Unchanged Stage A control evidence

| Variant | Delayed accuracy >32 | Update norm mean | Overwrite | Reset |
| --- | ---: | ---: | ---: | ---: |
| adaptive | 1.000 | 1.847852 | 1.000 | 1.000 |
| frozen_memory | 0.125 | 0.000000 | 0.125 | 1.000 |
| no_memory | 0.250 | 0.000000 | 0.250 | 1.000 |

Stage A gates passed: **True**.
The JSON contains matched gate statistics, gradient norms, configuration, and provenance.
