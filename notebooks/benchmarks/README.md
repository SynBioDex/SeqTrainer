# Benchmark Notebooks

Benchmark notebooks are grouped by model family so future experiments do not get mixed with the starter tutorials.

## Folders

- [`cnn_benchmark/`](cnn_benchmark/): CNN reference and CNN-v2 promoter classification benchmarks on the shared train/eval/test split.

Future model families should get their own folders, for example `dnabert2_benchmark/` or `ipromp_benchmark/`, while reusing the same dataset split and metric policy whenever possible.

## Reproducibility Rule

Benchmark notebooks should record:

- dataset source and split files
- model configuration
- random seed
- selected validation threshold
- validation metrics used for model choice
- held-out test metrics used for final reporting

