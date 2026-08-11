# CNN Benchmark Results and Specifications

This document explains the CNN reference and CNN-v2 experiments in sufficient
detail for another researcher to understand what was trained and reproduce the
comparison. Both use the predefined GSE144621 split, 300-base sequences, binary
labels, and seed 42.

## Complete Results

### Reference CNN

| Split | Threshold | Accuracy | Balanced accuracy | Precision | Recall | F1 | MCC | Specificity | TN | FP | FN | TP | AUROC | AUPRC | Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 0.670 | 0.584259 | 0.584797 | 0.788628 | 0.231952 | 0.358471 | 0.239275 | 0.937641 | 63,889 | 4,249 | 52,493 | 15,853 | 0.636586 | 0.664295 | 0.652761 |
| Validation | 0.670 | 0.559596 | 0.559267 | 0.705379 | 0.203285 | 0.315613 | 0.168827 | 0.915249 | 8,931 | 827 | 7,760 | 1,980 | 0.598002 | 0.611386 | 0.677147 |
| Test | 0.670 | 0.568622 | 0.566921 | 0.720034 | 0.217647 | 0.334257 | 0.187208 | 0.916195 | 17,951 | 1,642 | 15,180 | 4,223 | 0.599882 | 0.618783 | 0.673401 |

### CNN-v2

| Experiment | Split | Threshold | Accuracy | Balanced accuracy | Precision | Recall | F1 | MCC | Specificity | TN | FP | FN | TP | AUROC | AUPRC | Loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 cycles | Validation | 0.630 | 0.578111 | 0.577776 | 0.782252 | 0.215400 | 0.337788 | **0.225811** | 0.940152 | 9,174 | 584 | 7,642 | 2,098 | 0.608860 | 0.646326 | 0.660528 |
| 50 cycles | Test | 0.630 | 0.578239 | 0.576483 | 0.772091 | 0.216152 | 0.337749 | **0.220884** | 0.936814 | 18,355 | 1,238 | 15,209 | 4,194 | 0.610727 | **0.645976** | 0.660236 |
| 100 cycles | Validation | 0.555 | 0.575700 | 0.575365 | 0.774616 | 0.212423 | 0.333414 | 0.219189 | 0.938307 | 9,156 | 602 | 7,671 | 2,069 | 0.604743 | 0.635708 | 0.670571 |
| 100 cycles | Test | 0.555 | 0.574315 | 0.572571 | 0.753849 | 0.214503 | 0.333975 | 0.208165 | 0.930638 | 18,234 | 1,359 | 15,241 | 4,162 | 0.601339 | 0.634116 | 0.669822 |

The executed notebook did not retain a consolidated CNN-v2 train-metric row,
so no train values are inferred.

## Reference CNN Architecture

```text
300 bases -> five-channel one-hot tensor
Conv1d(5, 32, kernel 7) -> ReLU -> MaxPool
Conv1d(32, 64, kernel 5) -> ReLU
Adaptive max pool -> 64 values
Linear(64, 32) -> ReLU -> Linear(32, 2)
```

Training used Adam, learning rate 0.001, batch size 16, cross-entropy loss, no
dropout, no weight decay, no scheduler, and 10 cycles. The final cycle was
retained. The probability threshold 0.670 was selected from validation MCC.

## CNN-v2 Architecture

```text
Conv 5->64, kernel 15
Conv 64->64, kernel 7
MaxPool and dropout 0.15
Dilated conv 64->128, kernel 7, dilation 2
Dilated conv 128->128, kernel 7, dilation 4
MaxPool and dropout 0.15
Conv 128->256, kernel 3
Adaptive average pool + adaptive max pool
Concatenate to 512 values
Linear 512->128, GELU, dropout 0.30
Linear 128->2
```

Every convolution is followed by batch normalization and GELU. The dilated
layers inspect a wider sequence context, and combining average and maximum
pooling preserves both distributed and strongest motif evidence.

| Setting | 50-cycle selected model | 100-cycle comparison |
| --- | --- | --- |
| Batch size | 32 | 32 |
| Optimizer | AdamW | AdamW |
| Maximum learning rate | 0.0003 | 0.0003 |
| Weight decay | 0.0002 | 0.0002 |
| Scheduler | OneCycleLR per training batch | OneCycleLR |
| Loss | CrossEntropyLoss | CrossEntropyLoss |
| Class weighting | Disabled for the balanced dataset | Disabled |
| Checkpoint | Best validation MCC | Best validation MCC |
| Early-stopping patience | 10 cycles | 12 cycles |
| Validation-selected threshold | 0.630 | 0.555 |

The 50-cycle experiment is the selected CNN baseline because it obtained the
highest validation MCC. Its held-out test MCC and AUPRC also exceeded the
reference CNN.
