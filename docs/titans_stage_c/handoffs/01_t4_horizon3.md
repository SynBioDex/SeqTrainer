# Handoff 01 — T4 horizon-3 scaling gate

Notebook: `notebooks/titans_stage_c/01_stage_c_t4_horizon3.ipynb`

## Runtime and editable values

Select a Colab T4 GPU. Edit only the tagged cell with the pushed reviewed
commit, Drive root, and run name. The notebook reads the frozen tokenizer
selection from Handoff 00 and resolves the matching Handoff-00b dataset under
`stage_c_dataset/ordered_streams/`; it refuses a missing manifest rather than
using a placeholder path.

The notebook aborts before capacity testing if the device name does not contain
`T4`, the focused CPU contract tests fail, or the production-geometry GPU smoke
gate fails. The smoke gate runs one FP32 and one FP16 optimizer step, checks
finite parameters/gradients/functional state, verifies causal masking, and
compares CPU exact attention with GPU FP32 SDPA while TF32 is disabled for that
comparison. Stage C constrains this stateful higher-order-gradient path to
PyTorch's exact math-SDPA kernel on CUDA; fast Flash/efficient kernels do not
provide the required double backward on all Colab builds. It writes
`gpu_smoke.json` alongside the capacity evidence.

The provisional budget is at most 20 compute units. Record the balance before
and after execution in the returned run notes so later runs can calibrate
units/hour.

## Expected evidence and recovery

Success produces `hardware.json`, `gpu_smoke.json`, and
`t4_bounded_evidence.json/.md`, including one full-geometry horizon-3 optimizer
step for both FP32 and FP16, finite functional-state evidence, CPU/GPU FP32
parity, and FP16 causal masking. The extended multi-step capacity, checkpoint,
throughput, and validation matrix is deliberately deferred to Notebook 02 on
A100: Colab T4 terminates a second long-lived full-geometry capacity process.
If the
full geometry cannot fit after batch size one, retain the failure logs; do not
change the architecture in the notebook. Rerunning the notebook creates or
reuses only the named output directory. Return the directory printed at the
end, including `colab_run_manifest.json`, `logs/`, and `FAILED.txt` if present.
