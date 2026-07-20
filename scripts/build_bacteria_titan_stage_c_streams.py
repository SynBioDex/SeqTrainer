#!/usr/bin/env python3
"""Build clade-separated, ordered Stage C streams from cached bacterial data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from seqtrainer.data.bacteria_titan import (
    assign_hybrid_clade_groups,
    cluster_ani_pairs,
    iter_stage_c_fasta_records,
    materialize_token_stream_dataset,
    read_table,
    regenerate_fasta_shards,
    split_clade_groups,
    write_table,
)
from seqtrainer.torch.titans_paper_mac_stage_c import (
    Evo2CharTokenizer,
    HuggingFaceBPETokenizer,
    SeqTrainerBaseTokenizer,
    SixMerTokenizer,
    TrainOnlyBPETokenizer,
)
from seqtrainer.torch.titans_paper_mac_stage_c.reporting import bar_svg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ani-pairs", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        choices=("auto", "seqtrainer_base", "evo2_char", "6mer", "dnabert2_bpe", "bacterial_bpe"),
        required=True,
    )
    parser.add_argument(
        "--tokenizer-selection",
        type=Path,
        help="tokenizer_selection.json from the CPU gate; required with --tokenizer auto",
    )
    parser.add_argument("--dnabert2-path", type=Path)
    parser.add_argument("--bacterial-bpe-path", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--fasta-shard-bp", type=int, default=500_000_000)
    parser.add_argument("--tokens-per-shard", type=int, default=50_000_000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    selected_spec = None
    tokenizer_choice = args.tokenizer
    if tokenizer_choice == "auto":
        if args.tokenizer_selection is None:
            raise ValueError("--tokenizer-selection is required with --tokenizer auto")
        selection = json.loads(args.tokenizer_selection.read_text(encoding="utf-8"))
        selected_spec = selection.get("selected_tokenizer_spec")
        if not isinstance(selected_spec, dict):
            raise ValueError("tokenizer selection artifact has no selected_tokenizer_spec")
        choices = {
            "seqtrainer_base_v1": "seqtrainer_base",
            "evo2_charlevel_512": "evo2_char",
            "nonoverlap_6mer_v1": "6mer",
            "dnabert2_official_bpe": "dnabert2_bpe",
            "bacterial_train_only_bpe": "bacterial_bpe",
        }
        try:
            tokenizer_choice = choices[str(selected_spec["name"])]
        except KeyError as error:
            raise ValueError("tokenizer selection names an unsupported tokenizer") from error
    source_manifest_path = args.source_root / "manifests" / "accession_manifest.parquet"
    accessions = read_table(source_manifest_path)
    ecoli = accessions.loc[accessions["scope"].eq("ecoli_species"), "accession"].astype(str)
    pairs = pd.read_csv(args.ani_pairs, sep="\t")
    membership = cluster_ani_pairs(ecoli, pairs, threshold=99.0)
    grouped = assign_hybrid_clade_groups(accessions.drop(columns=["split"], errors="ignore"), membership)
    grouped = split_clade_groups(grouped, seed=args.seed)
    manifests = args.output_root / "manifests"
    write_table(membership, manifests / "ani99_membership")
    write_table(grouped, manifests / "accession_manifest")
    split_summary = {
        split: {
            "accessions": int(part.shape[0]),
            "clade_groups": int(part["clade_group"].nunique()),
            "manifest_bases": int(part["genome_size"].sum()),
        }
        for split, part in grouped.groupby("split", sort=True)
    }
    split_summary_path = manifests / "split_summary.json"
    split_summary_path.write_text(
        json.dumps(split_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    labels = ["train", "val", "test"]
    (manifests / "split_bases.svg").write_text(
        bar_svg(
            labels,
            [float(split_summary.get(split, {}).get("manifest_bases", 0)) for split in labels],
            title="Stage C clade-safe split size",
            x_label="Manifest bases",
        ),
        encoding="utf-8",
    )
    (manifests / "split_clade_groups.svg").write_text(
        bar_svg(
            labels,
            [float(split_summary.get(split, {}).get("clade_groups", 0)) for split in labels],
            title="Stage C independent clade groups",
            x_label="Clade groups",
        ),
        encoding="utf-8",
    )

    fasta_manifest = regenerate_fasta_shards(
        args.source_root / "raw" / "ncbi_dataset_zips",
        grouped,
        args.output_root,
        "bacteria_titan_stage_c",
        target_shard_bp=args.fasta_shard_bp,
    )
    write_table(fasta_manifest, manifests / "fasta_shard_manifest")

    if tokenizer_choice == "seqtrainer_base":
        tokenizer = SeqTrainerBaseTokenizer()
    elif tokenizer_choice == "evo2_char":
        tokenizer = Evo2CharTokenizer(require_official=True)
    elif tokenizer_choice == "6mer":
        tokenizer = SixMerTokenizer()
    elif tokenizer_choice == "dnabert2_bpe":
        if args.dnabert2_path is None:
            raise ValueError("--dnabert2-path is required for dnabert2_bpe")
        tokenizer = HuggingFaceBPETokenizer(args.dnabert2_path)
    else:
        if args.bacterial_bpe_path is None:
            raise ValueError(
                "--bacterial-bpe-path must point to the frozen train-only tokenizer from the CPU gate"
            )
        tokenizer = TrainOnlyBPETokenizer(
            args.bacterial_bpe_path,
            name="bacterial_train_only_bpe",
        )
    if selected_spec is not None:
        expected_checksum = str(selected_spec.get("checksum", ""))
        if tokenizer.spec.name != selected_spec.get("name") or tokenizer.spec.checksum != expected_checksum:
            raise ValueError("loaded tokenizer does not match the frozen CPU selection artifact")

    provenance = {
        "source_accession_manifest": str(source_manifest_path),
        "source_accession_manifest_sha256": sha256(source_manifest_path),
        "ani_pairs": str(args.ani_pairs),
        "ani_pairs_sha256": sha256(args.ani_pairs),
        "ani_threshold": 99.0,
        "ani_linkage": "single",
        "split_seed": args.seed,
        "memory_stream_unit": "contig_or_replicon",
    }
    paths = materialize_token_stream_dataset(
        iter_stage_c_fasta_records(fasta_manifest, grouped),
        tokenizer,
        args.output_root / "ordered_streams" / tokenizer.spec.name,
        tokens_per_shard=args.tokens_per_shard,
        provenance=provenance,
    )
    paths.update(
        {
            "split_summary": split_summary_path,
            "split_bases_plot": manifests / "split_bases.svg",
            "split_clade_plot": manifests / "split_clade_groups.svg",
        }
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
