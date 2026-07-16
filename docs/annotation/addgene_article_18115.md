# Addgene Article 18115 Evaluation Set

The collection manifest records the plasmids listed in Addgene's “Genetic circuit design automation” article page: [Addgene article 18115](https://www.addgene.org/browse/article/18115/). The page lists the plasmid IDs and names; SeqTrainer does not download them automatically, bypass Addgene access controls, or commit sequence files.

## Local preparation

1. Obtain the available GenBank files through the normal Addgene workflow.
2. Place only the downloaded files under `data/addgene_18115/raw/` using the filenames in `data-manifests/addgene_article_18115.csv`.
3. Do not rename a file to imply that it contains a promoter. The workflow examines the actual GenBank features.
4. Run the collection command with `--continue-on-error` so unavailable files are recorded in `excluded_plasmids.csv`.

Only files with explicitly labelled promoter features are included. A plasmid name, gate name, nearby CDS, or expected circuit architecture is never treated as evidence of a promoter.

## Evidence and limitations

Tier A accepts feature type `promoter`, regulatory features with an explicit promoter regulatory class, or an explicit SO:0000167 cross-reference. Tier B accepts a standalone `promoter` term in a label, name, or note, except phrases such as `promoterless` and `no promoter`. `--promoter-label-mode strict` limits the run to Tier A.

`annotation_completeness` must be recorded as `verified_complete`, `partial`, or `unknown`. For partial or unknown records, unlabeled windows are not confirmed biological negatives and the report emphasizes labelled-promoter recovery rather than definitive classification metrics.

## Reproducibility

The model threshold is loaded from the benchmark manifest and is never tuned on Addgene. The annotation manifest records the checkpoint and benchmark-manifest SHA-256 fingerprints. Raw GenBank files remain local and their SHA-256 values are recorded in collection outputs.
