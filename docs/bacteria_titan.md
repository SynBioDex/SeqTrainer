# Bacterial Titan MAC

SeqTrainer's bacterial Titan workflow builds genome-level DNA language-model
datasets and trains a custom Memory-as-Context causal LM without Hugging Face
Transformers. The reference dataset is
`bacteria_titan_v1_ecoli_related_15gbp`.

## Dataset layout

GTDB R220 metadata is cached under `raw/gtdb_r220/`; resumable NCBI Datasets
ZIP batches are under `raw/ncbi_dataset_zips/`. Derived files are deterministic:

```text
manifests/accession_manifest{,_train,_val,_test}.parquet
manifests/accession_manifest.csv
manifests/ncbi_batch_manifest.parquet
manifests/fasta_shard_manifest.parquet
manifests/token_shard_manifest.parquet
shards/<dataset_class>/<split>/shard_00000.fa.gz
tokenized/<dataset_class>/ctx2048/<split>/tokens_00000.npy
tokenized/<dataset_class>/ctx2048/tokenizer.json
checksums.sha256
dataset_manifest.json
README.md
```

Every token shard is a `uint8` matrix with shape
`(num_windows, context_length + 1)`. Training shifts each row once:

```python
input_ids = tokens[:, :-1]
labels = tokens[:, 1:]
```

The fixed token contract is `PAD=0, N/UNK=1, A=2, C=3, G=4, T=5`.

## Build v1

Install SeqTrainer with PyTorch and a pandas Parquet engine, mount Drive in
Colab, and install the NCBI Datasets CLI. Then run:

```bash
python scripts/build_bacteria_titan_dataset_colab.py \
  --drive-root /content/drive/MyDrive/seqtrainer \
  --dataset-class bacteria_titan_v1_ecoli_related_15gbp \
  --target-bp 15000000000 \
  --context-length 2048
```

Selection expands from E. coli to Escherichia, Enterobacteriaceae, and finally
Enterobacterales using deterministic 35/15/30/20 percent base-pair budgets.
Completeness must be at least 90%, contamination at most 5%, and genome size
0.5-12 Mbp. Representatives and quality are preferred within diversity-aware
round-robin groups.

**The v1 run must split true genomes/accessions 90/5/5 before FASTA sharding or
tokenization. Never use fallback random token-window splits.** This constraint
prevents homologous windows from the same assembly leaking across evaluation
sets.

## Train on Colab A100

```bash
python scripts/train_titan_mac_dna_lm_colab.py \
  --drive-root /content/drive/MyDrive/seqtrainer \
  --dataset-class bacteria_titan_v1_ecoli_related_15gbp \
  --context-length 2048 \
  --run-name titan_mac_v1_12m \
  --batch-size 8 \
  --grad-accumulation 4 \
  --epochs 10 \
  --compile
```

Token shards are copied to `/content/bacteria_titan_local` before training.
The reference model uses `d_model=384`, six layers, eight heads, feed-forward
width 1536, 64 memory slots, eight retrieved context tokens, and eight
persistent tokens. The script prints the exact trainable parameter count.

TF32, bf16 autocast, fused AdamW, pinned memory, persistent workers, and
prefetching are enabled when supported. Each epoch atomically writes
`latest.pt`, `history.csv`, and `metrics.png` to Drive. `best_overall.pt` tracks
a lower-is-better validation score combining loss, token accuracy, and top-2
accuracy. By default, rerunning the same command restores the
full trusted checkpoint with PyTorch 2.6-compatible `weights_only=False` and
continues at the next epoch. Use `--no-resume` only for a fresh run.

## Metrics

- Loss and perplexity measure next-base negative log likelihood.
- Bits per base is loss divided by `ln(2)` and is comparable to compression.
- Token and top-2 accuracy measure ranking quality.
- Confidence and entropy expose overconfidence or uncertainty.
- GC-bin losses separate low (`<40%`), middle, and high (`>=60%`) GC windows.
- Tokens/sec and allocated/peak GPU memory diagnose A100 utilization.

## Analyze

```bash
python scripts/analyze_titan_mac_dna_lm_colab.py \
  --drive-root /content/drive/MyDrive/seqtrainer \
  --dataset-class bacteria_titan_v1_ecoli_related_15gbp \
  --context-length 2048 \
  --run-name titan_mac_v1_12m \
  --samples 2000 \
  --tsne
```

Analysis loads `best_overall.pt` and falls back to `latest.pt`. It exports PCA
2D and 3D plots and coordinate CSVs for token, position, and sequence hidden
embeddings. Optional t-SNE emphasizes local neighborhoods but does not preserve
global distances. Sequence coordinates include GC fraction, split, and top
memory slot. A row-aligned metadata CSV can add scope, accession, genus, and
family. Memory usage identifies frequently retrieved slots; a cosine heatmap
reveals redundant or specialized memory directions. `REPORT.md` explains every
artifact.

## Scaling plan

| Version | Data | Model |
| --- | --- | --- |
| v1 | E. coli-related, 15 Gbp | ~12M parameters |
| v2 | broader Gram-negative, 30-50 Gbp | 25-50M parameters |
| v3 | bacteria + archaea representatives, 100-300 Gbp | 50-100M parameters |
| v4 | full prokaryotic GTDB R220 + IMG/PR / OpenGenome2 scale | 100M+ parameters |

Later datasets should retain accession-level splits, immutable manifests,
checksums, cached source archives, and deterministic rematerialization.
