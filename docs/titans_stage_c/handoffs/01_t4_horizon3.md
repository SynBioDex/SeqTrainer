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
comparison. It writes `gpu_smoke.json` alongside the capacity evidence.

The provisional budget is at most 20 compute units. Record the balance before
and after execution in the returned run notes so later runs can calibrate
units/hour.

## Expected evidence and recovery

Success produces `hardware.json`, `gpu_smoke.json`, `capacity_matrix.json/.md`,
FP32/FP16 checkpoint files, nonzero horizon-3 written-state gradients, FP32
memory-state dtypes, matched bounded validation BPB, peak CUDA memory,
bases/sec, and checkpoint save/load timings. If the
full geometry cannot fit after batch size one, retain the failure logs; do not
change the architecture in the notebook. Rerunning the notebook creates or
reuses only the named output directory. Return the directory printed at the
end, including `colab_run_manifest.json`, `logs/`, and `FAILED.txt` if present.
