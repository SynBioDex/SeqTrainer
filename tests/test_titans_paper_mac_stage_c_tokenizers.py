from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import zipfile

import pandas as pd
import pytest

from seqtrainer.data.bacteria_titan import (
    assert_no_clade_leakage,
    assign_hybrid_clade_groups,
    build_stream_segments,
    cluster_ani_pairs,
    load_stream_dataset,
    materialize_token_stream_dataset,
    materialize_stream_dataset,
    run_skani_triangle,
    split_clade_groups,
    TokenStreamDataset,
)
from seqtrainer.torch.titans_paper_mac_stage_c import (
    Evo2CharTokenizer,
    SeqTrainerBaseTokenizer,
    SixMerTokenizer,
    TrainOnlyBPETokenizer,
    evaluate_tokenizer,
)
from seqtrainer.torch.titans_paper_mac_stage_c.cli import _select_tokenizer


@pytest.mark.parametrize(
    "tokenizer",
    [SeqTrainerBaseTokenizer(), SixMerTokenizer(), Evo2CharTokenizer()],
)
def test_dependency_free_tokenizers_round_trip_supported_dna(tokenizer) -> None:
    sequence = "ACGTNACGTACGTNNTA"
    encoded = tokenizer.encode(sequence.lower())

    assert tokenizer.decode(encoded.ids) == sequence
    assert encoded.base_count == len(sequence)
    assert encoded.base_spans[0][0] == 0
    assert encoded.base_spans[-1][1] == len(sequence)


def test_sixmer_is_multinucleotide_and_lossless_around_n_and_tail() -> None:
    tokenizer = SixMerTokenizer()
    sequence = "ACGTACACNTACGTACGTA"
    encoded = tokenizer.encode(sequence)

    assert tokenizer.decode(encoded.ids) == sequence
    assert any(end - start == 6 for start, end in encoded.base_spans)
    assert any(end - start == 1 for start, end in encoded.base_spans)
    assert evaluate_tokenizer(tokenizer, [sequence]).eligible is True


def test_evo2_fallback_is_reported_but_cannot_silently_win() -> None:
    tokenizer = Evo2CharTokenizer()
    metrics = evaluate_tokenizer(tokenizer, ["ACGTN"])

    if tokenizer.spec.verification == "ascii_fallback_unverified":
        assert metrics.eligible is False
    else:
        assert tokenizer.spec.verification == "official_vortex"


def test_train_only_bpe_is_lossless_and_has_measured_multibase_tokens(tmp_path) -> None:
    pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")
    sequences = ["ACGTACGTACGTACGT", "GGGGCCCCAAAATTTT", "NNACGTACGTNN"]
    tokenizer = TrainOnlyBPETokenizer.train(sequences, tmp_path / "bpe", vocab_size=32)
    metrics = evaluate_tokenizer(tokenizer, sequences)

    assert metrics.round_trip_passed
    assert metrics.eligible
    assert metrics.bases_per_token > 1
    assert metrics.maximum_token_length > 1
    assert set(metrics.gc_bin_bases_per_token) == {"gc_0_40", "gc_40_60", "gc_60_100"}


def test_tokenizer_selection_uses_equal_bases_threshold_and_freezes_spec() -> None:
    tokenizers = [SeqTrainerBaseTokenizer(), SixMerTokenizer()]
    metrics = [evaluate_tokenizer(tokenizer, ["ACGT" * 20]) for tokenizer in tokenizers]
    runs = [
        {"tokenizer": "seqtrainer_base_v1", "regime": "equal_bases", "validation_bpb": 1.0},
        {"tokenizer": "nonoverlap_6mer_v1", "regime": "equal_bases", "validation_bpb": 0.995},
        {"tokenizer": "nonoverlap_6mer_v1", "regime": "equal_steps", "validation_bpb": 0.1},
    ]
    fallback = _select_tokenizer(metrics, runs, tokenizers)
    assert fallback["selected_tokenizer"] == "seqtrainer_base_v1"
    assert fallback["fallback_applied"] is True

    runs[1]["validation_bpb"] = 0.98
    promoted = _select_tokenizer(metrics, runs, tokenizers)
    assert promoted["selected_tokenizer"] == "nonoverlap_6mer_v1"
    assert promoted["selected_tokenizer_spec"] == tokenizers[1].spec.to_dict()


def test_tokenizer_selection_rejects_nonfinite_validation_bpb() -> None:
    tokenizer = SeqTrainerBaseTokenizer()
    metrics = [evaluate_tokenizer(tokenizer, ["ACGT" * 20])]
    runs = [
        {"tokenizer": "seqtrainer_base_v1", "regime": "equal_bases", "validation_bpb": float("nan")}
    ]

    with pytest.raises(ValueError, match="non-finite"):
        _select_tokenizer(metrics, runs, [tokenizer])


def test_clade_groups_never_cross_splits_and_leakage_is_detected() -> None:
    rows = []
    for scope in ("ecoli_species", "enterobacteriaceae_family"):
        for group in range(12):
            rows.append(
                {
                    "accession": f"{scope}-{group}",
                    "clade_group": f"{scope}-ani-{group}",
                    "scope": scope,
                    "genome_size": 1_000 + group,
                }
            )
    split = split_clade_groups(pd.DataFrame(rows), seed=23)

    assert set(split["split"]) == {"train", "val", "test"}
    assert split.groupby("clade_group")["split"].nunique().max() == 1
    duplicated = pd.concat(
        [split, split.iloc[[0]].assign(split="test" if split.iloc[0]["split"] != "test" else "train")]
    )
    with pytest.raises(ValueError, match="leakage"):
        assert_no_clade_leakage(duplicated)


def test_skani_pairs_form_deterministic_ani99_groups_and_hybrid_contract() -> None:
    accessions = ["GCF_000001.1", "GCF_000002.1", "GCF_000003.1"]
    pairs = pd.DataFrame(
        {
            "Ref_file": [f"/genomes/{accessions[0]}.fna", f"/genomes/{accessions[1]}.fna"],
            "Query_file": [f"/genomes/{accessions[1]}.fna", f"/genomes/{accessions[2]}.fna"],
            "ANI": [99.2, 98.9],
        }
    )
    membership = cluster_ani_pairs(accessions, pairs)
    assert membership.loc[0, "ani_cluster_99"] == membership.loc[1, "ani_cluster_99"]
    assert membership.loc[1, "ani_cluster_99"] != membership.loc[2, "ani_cluster_99"]

    manifest = pd.DataFrame(
        {
            "accession": accessions,
            "scope": ["ecoli_species", "ecoli_species", "escherichia_genus"],
            "species": ["Escherichia coli", "Escherichia coli", "Escherichia albertii"],
        }
    )
    grouped = assign_hybrid_clade_groups(manifest, membership)
    assert grouped.loc[0, "clade_group"].startswith("ani99:")
    assert grouped.loc[2, "clade_group"] == "gtdb_species:Escherichia albertii"


def test_skani_triangle_uses_sparse_list_input_and_screen_below_boundary(
    tmp_path, monkeypatch
) -> None:
    fastas = [tmp_path / "GCF_000001.1.fna", tmp_path / "GCF_000002.1.fna"]
    for path in fastas:
        path.write_text(">contig\nACGT\n", encoding="ascii")
    observed = {}

    def fake_run(command, *, stdout, check):
        observed["command"] = command
        observed["check"] = check
        stdout.write("Ref_file\tQuery_file\tANI\n")

    monkeypatch.setattr(
        "seqtrainer.data.bacteria_titan.stage_c_streams.subprocess.run",
        fake_run,
    )
    output = run_skani_triangle(fastas, tmp_path / "ani.tsv", threads=7)

    command = observed["command"]
    assert command[:3] == ["skani", "triangle", "-l"]
    assert command[command.index("-s") + 1] == "95"
    assert "-E" in command
    assert command[command.index("-t") + 1] == "7"
    inputs = Path(command[command.index("-l") + 1]).read_text(encoding="utf-8").splitlines()
    assert inputs == [str(path.resolve()) for path in fastas]
    assert output.read_text(encoding="utf-8").startswith("Ref_file")


def test_ani_generation_materializes_one_fasta_per_ecoli_accession(tmp_path) -> None:
    script_path = Path(__file__).parents[1] / "scripts" / "generate_stage_c_ani_pairs.py"
    spec = importlib.util.spec_from_file_location("stage_c_ani_generator", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    zip_dir = tmp_path / "zips"
    zip_dir.mkdir()
    with zipfile.ZipFile(zip_dir / "batch.zip", "w") as archive:
        archive.writestr("ncbi_dataset/data/GCF_000001.1/a.fna", ">a\nACGT\n")
        archive.writestr("ncbi_dataset/data/GCF_000002.1/b.fna", ">b\nTGCA\n")
        archive.writestr("ncbi_dataset/data/GCF_000003.1/c.fna", ">c\nNNNN\n")

    paths = module.materialize_ecoli_fastas(
        zip_dir,
        {"GCF_000001.1", "GCF_000002.1"},
        tmp_path / "fastas",
    )

    assert [path.name for path in paths] == ["GCF_000001.1.fna", "GCF_000002.1.fna"]
    assert paths[0].read_text(encoding="ascii") == ">a\nACGT\n"
    assert paths[1].read_text(encoding="ascii") == ">b\nTGCA\n"


def test_stream_segments_preserve_cross_segment_labels_and_mask_tail(tmp_path) -> None:
    tokenizer = SeqTrainerBaseTokenizer()
    sequence = "ACGT" * 20 + "A"
    segments = build_stream_segments(
        sequence=sequence,
        accession="GCF_1",
        contig_id="chromosome",
        split="train",
        clade_group="ani-1",
        tokenizer=tokenizer,
    )

    assert len(segments) == 3
    assert segments[0].labels[-1] == segments[1].input_ids[0]
    assert segments[0].start_of_stream and not segments[0].end_of_stream
    assert segments[0].gc_fraction == pytest.approx(sequence.count("G") / len(sequence) + sequence.count("C") / len(sequence))
    assert segments[-1].end_of_stream
    assert segments[-1].valid_tokens == len(sequence) - 1 - 64
    assert segments[-1].valid_bases == segments[-1].valid_tokens
    assert not any(segments[-1].loss_mask[segments[-1].valid_tokens :])

    records = pd.DataFrame(
        [
            {
                "accession": "GCF_1",
                "contig_id": "chromosome",
                "sequence": sequence,
                "split": "train",
                "clade_group": "ani-1",
            }
        ]
    )
    paths = materialize_stream_dataset(records, tokenizer, tmp_path)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["segments"] == 3
    assert manifest["streams"] == 1
    assert all(path.exists() for path in paths.values())
    restored = load_stream_dataset(tmp_path, split="train")
    assert restored == segments


def test_production_token_stream_format_is_lazy_mmap_and_segment_exact(tmp_path) -> None:
    tokenizer = SixMerTokenizer()
    records = [
        {
            "accession": "GCF_2",
            "contig_id": "chromosome",
            "sequence": "ACGTAC" * 40 + "NTA",
            "split": "train",
            "clade_group": "ani-2",
        }
    ]
    paths = materialize_token_stream_dataset(
        records,
        tokenizer,
        tmp_path,
        tokens_per_shard=35,
        provenance={"fixture": True},
    )
    dataset = TokenStreamDataset(tmp_path, verify_checksums=True)
    streams = dataset.streams(split="train")
    stream = streams["GCF_2:chromosome"]
    direct = build_stream_segments(
        sequence=records[0]["sequence"],
        accession="GCF_2",
        contig_id="chromosome",
        split="train",
        clade_group="ani-2",
        tokenizer=tokenizer,
    )

    assert len(stream) == len(direct)
    assert tuple(stream[index] for index in range(len(stream))) == direct
    assert stream[0].gc_fraction == pytest.approx(0.5, abs=0.02)
    assert paths["manifest"].exists()
