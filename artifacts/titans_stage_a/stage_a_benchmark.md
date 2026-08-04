# Stage A synthetic-memory benchmark

This is a deterministic synthetic correctness probe, not a biology or DNA-performance result.

| Variant | Delayed >32 accuracy | Overwrite | Reset | Eval BPB | Train/eval gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| adaptive | 1.000 | 1.000 | 1.000 | 0.049 | 0.000 |
| frozen_memory | 0.125 | 0.125 | 1.000 | 2.112 | -0.065 |
| no_memory | 0.250 | 0.250 | 1.000 | 2.292 | -0.042 |

## Acceptance gates

- PASS: adaptive_beats_controls_beyond_32
- PASS: adaptive_lifecycle_tasks_correct
- PASS: lifecycle_reference_passed
- PASS: leakage_reference_passed
- PASS: mask_reference_passed
- PASS: overall
