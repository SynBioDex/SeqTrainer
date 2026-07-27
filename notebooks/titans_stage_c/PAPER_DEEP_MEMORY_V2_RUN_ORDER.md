# Paper-deep memory v2 run order

This sequence is a prerequisite study. It does not replace or rewrite the
frozen 25M v1 protocol.

1. `03c_stage_c_paper_deep_memory_oracle.ipynb` — Mac/CPU FP64 mathematics and
   backend parity.
2. `03d_stage_c_paper_deep_memory_t4_stability.ipynb` — T4 smoke, exact and
   stabilized 500-step soaks, and deterministic resume verification.
3. Review both 03d runs. Prefer `paper_exact` only if it is finite and passes
   resume verification; otherwise choose `stabilized_rms_v1` and retain its
   intervention telemetry.
4. `03e_stage_c_paper_deep_compact_behavior.ipynb` — four independent compact
   one-million-base conditions and held-out memory-state visualization.
5. Scale the adaptive model only if the preregistered stability and behavior
   gates pass. Compact evidence cannot support the final adaptive-memory claim.
