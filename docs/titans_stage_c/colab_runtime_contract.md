# Stage C Colab runtime contract

Stage C supports the current Colab Python 3.12 runtime and uses its prebuilt
NumPy, Pandas, PyArrow, and PyTorch packages as a single binary-compatible
unit. SeqTrainer itself is installed editable, without dependency resolution,
in a lightweight virtual environment that inherits that unit.

The code-level minimums are Python 3.10, NumPy 1.24, Pandas 1.5, PyArrow 12,
and PyTorch 2.2. Notebook 00b records the actual Colab versions in
`runs/c2_stream_dataset/logs/bootstrap.log`. A missing import or incompatible
base environment is therefore a logged preflight failure, not a partially
modified runtime.

Do not pin or reinstall individual binary packages in a live Colab runtime.
Colab images change over time; the safe compatibility contract is to validate
the image's already-coordinated stack at startup and only make a targeted,
fully resolved environment change when that preflight log shows it is needed.
