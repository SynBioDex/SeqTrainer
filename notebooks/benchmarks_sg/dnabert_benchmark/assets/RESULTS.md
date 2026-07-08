# CNN and DNABERT2 Benchmark Results

This document records the completed CNN, frozen DNABERT2-v1, and DNABERT2
Colab T4 fine-tuning promoter classification experiments. The T4 run is a
resource-constrained full fine-tuning profile; the canonical A100/Alpine
fine-tuning profile still needs to be run for final claim-bearing scores.

## What Was Kept Identical

All models used the same biological prediction task and predefined data:

| Item | Value |
| --- | --- |
| Task | Binary bacterial promoter prediction from DNA sequence |
| Dataset | GSE144621, `EP_DNA_BERT2_genomic_order` |
| Train file | `train_EP_DNA_BERT2_genomic_order.csv` |
| Validation file | `eval_EP_DNA_BERT2_genomic_order.csv` |
| Test file | `test_EP_DNA_BERT2_genomic_order.csv` |
| Rows | 136,484 train; 19,498 validation; 38,996 test |
| Labels | `0` = negative and `1` = promoter-positive |
| Sequence length | 300 nucleotides |
| Random seed | 42 |
| Model selection | Validation data only |
| Probability threshold | Selected by maximizing validation MCC |
| Test use | Final reporting only; never used for tuning |
| Primary metric | MCC |
| Secondary metric | AUPRC |

The training classes were almost equal: 68,138 negatives and 68,346 positives.
Class weighting was therefore not activated.

## Complete Recorded Results

### CNN Reference

| Split | Threshold | Accuracy | Balanced accuracy | Precision | Recall / sensitivity | F1 | MCC | Specificity | TN | FP | FN | TP | AUROC | AUPRC | Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 0.670 | 0.584259 | 0.584797 | 0.788628 | 0.231952 | 0.358471 | 0.239275 | 0.937641 | 63,889 | 4,249 | 52,493 | 15,853 | 0.636586 | 0.664295 | 0.652761 |
| Validation | 0.670 | 0.559596 | 0.559267 | 0.705379 | 0.203285 | 0.315613 | 0.168827 | 0.915249 | 8,931 | 827 | 7,760 | 1,980 | 0.598002 | 0.611386 | 0.677147 |
| Test | 0.670 | 0.568622 | 0.566921 | 0.720034 | 0.217647 | 0.334257 | 0.187208 | 0.916195 | 17,951 | 1,642 | 15,180 | 4,223 | 0.599882 | 0.618783 | 0.673401 |

### CNN-v2 Candidates

The executed notebook retained validation and test result rows for these
candidates. A consolidated CNN-v2 train row was not retained, so it is not
invented here.

| Experiment | Split | Threshold | Accuracy | Balanced accuracy | Precision | Recall / sensitivity | F1 | MCC | Specificity | TN | FP | FN | TP | AUROC | AUPRC | Loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN-v2, 50 cycles | Validation | 0.630 | 0.578111 | 0.577776 | 0.782252 | 0.215400 | 0.337788 | **0.225811** | 0.940152 | 9,174 | 584 | 7,642 | 2,098 | 0.608860 | 0.646326 | 0.660528 |
| CNN-v2, 50 cycles | Test | 0.630 | 0.578239 | 0.576483 | 0.772091 | 0.216152 | 0.337749 | **0.220884** | 0.936814 | 18,355 | 1,238 | 15,209 | 4,194 | 0.610727 | **0.645976** | 0.660236 |
| CNN-v2, 100 cycles | Validation | 0.555 | 0.575700 | 0.575365 | 0.774616 | 0.212423 | 0.333414 | 0.219189 | 0.938307 | 9,156 | 602 | 7,671 | 2,069 | 0.604743 | 0.635708 | 0.670571 |
| CNN-v2, 100 cycles | Test | 0.555 | 0.574315 | 0.572571 | 0.753849 | 0.214503 | 0.333975 | 0.208165 | 0.930638 | 18,234 | 1,359 | 15,241 | 4,162 | 0.601339 | 0.634116 | 0.669822 |

### DNABERT2 Frozen v1

| Split | Threshold | Accuracy | Balanced accuracy | Precision | Recall / sensitivity | F1 | MCC | Specificity | TN | FP | FN | TP | AUROC | AUPRC | Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 0.505220 | 0.553017 | 0.553339 | 0.593138 | 0.341966 | 0.433819 | Not available | 0.764713 | 52,106 | 16,032 | 44,974 | 23,372 | 0.571109 | 0.572131 | 0.689723 |
| Validation | 0.505220 | 0.555339 | 0.555141 | 0.596050 | 0.340862 | 0.433703 | 0.122066 | 0.769420 | 7,508 | 2,250 | 6,420 | 3,320 | 0.572632 | 0.573041 | 0.689663 |
| Test | 0.505220 | 0.557262 | 0.556231 | 0.595158 | 0.344586 | **0.436466** | 0.124165 | 0.767876 | 15,045 | 4,548 | 12,717 | 6,686 | 0.573813 | 0.575073 | 0.689609 |

The missing DNABERT2 train MCC came from integer overflow in the original
custom MCC calculation on the large training confusion matrix. Validation and
test MCC were unaffected. DNABERT2-v2 uses scikit-learn MCC to avoid this.

## Model 1: CNN Reference

### What enters the model

Each 300-base sequence is padded or trimmed to exactly 300 positions. Every
position is represented by five binary channels: A, C, G, T, or N. One batch
therefore has shape:

```text
[batch size, 5 nucleotide channels, 300 positions]
```

### Layer-by-layer architecture

```text
Input: 5 x 300
Conv1d: 5 -> 32 channels, kernel 7, padding 3
ReLU
MaxPool1d: halves sequence length
Conv1d: 32 -> 64 channels, kernel 5, padding 2
ReLU
AdaptiveMaxPool1d: compresses every channel to one value
Flatten: 64 values
Linear: 64 -> 32
ReLU
Linear: 32 -> 2 class logits
```

The first convolution learns short sequence motifs. The second combines those
motifs. Adaptive max pooling retains the strongest activation for each learned
feature, regardless of its position.

### Training details

| Setting | Value |
| --- | --- |
| Trainable component | Entire CNN |
| Cycles / epochs | 10 |
| Batch size | 16 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0 |
| Scheduler | None |
| Loss | Two-class cross-entropy |
| Dropout | None |
| Class weighting | None |
| Checkpoint | Final cycle |
| Threshold | 0.670, chosen on validation MCC after training |

## Model 2: CNN-v2

CNN-v2 is a substantially deeper CNN, not simply a longer run of the reference
model.

### Layer-by-layer architecture

```text
Input: 5 x 300
Conv1d: 5 -> 64, kernel 15
BatchNorm + GELU
Conv1d: 64 -> 64, kernel 7
BatchNorm + GELU
MaxPool1d + dropout 0.15

Dilated Conv1d: 64 -> 128, kernel 7, dilation 2
BatchNorm + GELU
Dilated Conv1d: 128 -> 128, kernel 7, dilation 4
BatchNorm + GELU
MaxPool1d + dropout 0.15

Conv1d: 128 -> 256, kernel 3
BatchNorm + GELU

Adaptive average pooling: 256 values
Adaptive max pooling: 256 values
Concatenate both pools: 512 values
Linear: 512 -> 128
GELU + dropout 0.30
Linear: 128 -> 2 class logits
```

The 15-base first kernel can detect wider promoter motifs. Dilations 2 and 4
increase the receptive field without requiring extremely large kernels.
Average pooling summarizes broadly present evidence, while max pooling captures
the strongest motif response.

### Training details

| Setting | Selected 50-cycle run | 100-cycle comparison |
| --- | ---: | ---: |
| Maximum cycles / epochs | 50 | 100 |
| Batch size | 32 | 32 |
| Optimizer | AdamW | AdamW |
| Maximum learning rate | 0.0003 | 0.0003 |
| Weight decay | 0.0002 | 0.0002 |
| Scheduler | OneCycleLR, updated each training batch | OneCycleLR |
| Loss | Two-class cross-entropy | Two-class cross-entropy |
| Class weighting | Disabled because this dataset is balanced | Disabled |
| Best checkpoint | Highest validation MCC | Highest validation MCC |
| Early-stopping patience | 10 cycles | 12 cycles |
| Selected threshold | 0.630 | 0.555 |

The 50-cycle candidate was selected because validation MCC was higher than both
the reference and the 100-cycle run. The test set only confirmed the selected
candidate afterward.

## Model 3: DNABERT2 Frozen v1

### What “frozen” means

DNABERT2 contains approximately 117 million pretrained parameters. During this
experiment, those encoder parameters were not updated. DNABERT2 converted each
DNA sequence into a 768-number embedding. Only the small final classifier was
trained.

### Data flow

```text
300-base DNA sequence
DNABERT2 native BPE tokenizer
Truncate to at most 104 tokens
Pad each batch to its longest sequence, rounded to a multiple of 8
Frozen DNABERT2 encoder
Hidden state for every token: [tokens, 768]
Attention-mask-aware mean over real tokens
One 768-number sequence embedding
Dropout 0.10
Linear layer: 768 -> 1 logit
Sigmoid probability for promoter class
```

Mean pooling excludes padding tokens. The encoder embeddings for train,
validation, and test were cached so the expensive transformer did not have to
run during every classifier epoch.

### Training details

| Setting | Value |
| --- | --- |
| Model | `zhihan1996/DNABERT-2-117M` |
| Tokenization | Native DNABERT2 BPE; no k-mer conversion |
| Encoder | Frozen |
| Trainable component | Dropout plus one `Linear(768, 1)` head |
| Embedding extraction batch | 16 |
| Maximum head epochs | 50 |
| Optimizer | AdamW |
| Classifier learning rate | 0.0003 |
| Weight decay | 0.0001 |
| Loss | BCEWithLogitsLoss |
| Warmup | First 6% of classifier steps |
| Early-stopping patience | 8 epochs |
| Pooling | Attention-mask-aware mean |
| Threshold | 0.505219758, selected on validation MCC |
| Class weighting | None; imbalance ratio was 1.003 |
| Attention implementation | Stable PyTorch fallback; incompatible Triton path disabled |
| Runtime | 6,619.4 seconds, approximately 110 minutes |
| Peak GPU memory | 1,375.6 MB |

### Important limitation

The original frozen-v1 implementation performed one full-training-set
classifier update per epoch. Fifty epochs therefore meant only about 50
optimizer updates. DNABERT2-v2 corrects this by using shuffled mini-batches for
the classifier head. This is why frozen-v1 should be treated as a reproducible
reference, not the final DNABERT2 capability estimate.

## Interpretation

CNN-v2 currently leads on the project’s primary criteria:

- test MCC: 0.220884 versus 0.124165 for frozen DNABERT2
- test AUPRC: 0.645976 versus 0.575073

Frozen DNABERT2 detects more positive examples, giving it higher recall and F1,
but it also creates many more false positives. Its specificity and MCC are
therefore lower.

The next valid comparison is to execute DNABERT2-v2. Its candidate settings
must still be selected on validation data, and its held-out test result must
remain untouched until selection is complete.

## DNABERT2 Full Fine-Tuning, Colab T4 Result

This run completed in `notebooks/colab_benchmarks/dnabert2_finetune_t4_colab.ipynb`.
It proves that the full fine-tuning workflow can run on a Colab T4 with the
reduced resource profile. It should not yet be treated as the final DNABERT2
claim because the output split sizes differ from the CNN/DNABERT2 frozen table
above and should be verified before claiming a direct model improvement.

| Split | Threshold | Accuracy | Balanced accuracy | Precision | Recall / sensitivity | F1 | MCC | Specificity | TN | FP | FN | TP | AUROC | AUPRC | Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 0.750244 | 0.721295 | 0.539548 | 0.704437 | 0.095954 | 0.168902 | 0.183657 | 0.983142 | 155,771 | 2,671 | 59,978 | 6,366 | 0.541029 | 0.345434 | 0.952912 |
| Validation | 0.750244 | 0.692529 | 0.530771 | 0.655226 | 0.081780 | 0.145410 | 0.146619 | 0.979762 | 42,797 | 884 | 18,863 | 1,680 | 0.532065 | 0.355913 | 0.985773 |
| Test | 0.750244 | 0.683493 | 0.529984 | 0.679803 | 0.078098 | 0.140102 | **0.147631** | 0.981870 | 21,121 | 390 | 9,774 | 828 | 0.531969 | **0.365169** | 0.995106 |

Training history:

| Epoch | Train loss | Validation loss | Validation MCC | Validation threshold | Learning rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.964707 | 0.996619 | 0.108300 | 0.590820 | 0.000017 |
| 2 | 0.957736 | 0.985773 | 0.146619 | 0.750244 | 0.000000 |

The selected threshold came from validation MCC and was then applied unchanged
to the held-out test split. The run wrote the expected artifacts:
`metrics.csv`, `metrics.json`, `predictions.csv`, `manifest.json`,
`history.csv`, `config.json`, `input_split_audit.json`, and
`checkpoints/best_model.pt`.

### T4 Code/Profile Difference From A100/Alpine

The T4 notebook and the A100/Alpine notebook keep the same scientific comparison
surface: dataset identity, predefined split file names, seed `42`, DNABERT2
backbone, model revision, full encoder fine-tuning, mean pooling, AdamW,
learning rate `0.00003`, validation-MCC threshold selection, and final held-out
test reporting.

The differences are resource settings:

| Setting | Colab T4 profile | A100/Alpine profile |
| --- | ---: | ---: |
| Maximum epochs | 2 | 4 |
| Physical batch size | 2 | 4 |
| Gradient accumulation | 16 | 8 |
| Effective batch size | 32 | 32 |
| Precision | FP16 | BF16 |
| Early-stopping patience | 1 | 2 |
| Gradient checkpointing | Enabled | Not required in the canonical profile |
| Purpose | Resource-constrained reproducibility run | Canonical claim-bearing fine-tuning run |

Both profiles use `model_max_length = 70`, native DNABERT2 tokenization,
`classifier_dropout = 0.1`, `weight_decay = 0.01`, `warmup_ratio = 0.1`, and
`max_grad_norm = 1.0`.

The T4 full fine-tuning run did not improve on CNN-v2:

- CNN-v2 test MCC: 0.220884 versus 0.147631 for DNABERT2 T4 full fine-tuning.
- CNN-v2 test AUPRC: 0.645976 versus 0.365169 for DNABERT2 T4 full fine-tuning.

The T4 model has high specificity but very low recall, so it predicts very few
positive promoters. This is why accuracy looks reasonable while MCC and AUPRC
remain below CNN-v2. The next claim-bearing comparison should run the
A100/Alpine full fine-tuning profile, verify the exact split files, and compare
held-out test MCC and AUPRC against CNN-v2.
