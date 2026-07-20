# Handoff 01 — T4 horizon-3 scaling gate

Notebook: `notebooks/titans_stage_c/01_stage_c_t4_horizon3.ipynb`

## Runtime and editable values

Select a Colab T4 GPU. Edit only the tagged cell with the pushed reviewed
commit, Drive root, locked tokenizer dataset directory, and run name. The
notebook aborts before training if the device name does not contain `T4` or if
the Stage C contract tests fail.

The provisional budget is at most 20 compute units. Record the balance before
and after execution in the returned run notes so later runs can calibrate
units/hour.

## Expected evidence and recovery

Success produces `hardware.json`, `capacity_matrix.json/.md`, FP32/FP16
checkpoint files, nonzero horizon-3 written-state gradients, FP32 memory-state
dtypes, matched bounded validation BPB, peak CUDA memory, bases/sec, and
checkpoint save/load timings. If the
full geometry cannot fit after batch size one, retain the failure logs; do not
change the architecture in the notebook. Rerunning the notebook creates or
reuses only the named output directory. Return the directory printed at the
end, including `colab_run_manifest.json`, `logs/`, and `FAILED.txt` if present.
