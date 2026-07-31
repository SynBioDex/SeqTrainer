from __future__ import annotations

import itertools
import json

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from seqtrainer.data.bacteria_titan import (  # noqa: E402
    StageCPanelManifest,
    TokenStreamDataset,
    freeze_ecoli_panels,
    materialize_token_stream_dataset,
    validate_panel_against_dataset,
)
from seqtrainer.torch.titans_paper_mac_stage_c import (  # noqa: E402
    BaseCosineLRSchedule,
    SeqTrainerBaseTokenizer,
    StatefulRotationScheduler,
)


def _accession(index: int) -> str:
    return f"GCF_{index:09d}.1"


def _panel_fixture(tmp_path):
    dataset_dir = tmp_path / "dataset"
    tokenizer = SeqTrainerBaseTokenizer()
    rows = []
    records = []
    membership = []
    split_ranges = {"train": range(1, 7), "val": range(101, 109), "test": range(201, 209)}
    for split, indices in split_ranges.items():
        for index in indices:
            accession = _accession(index)
            rows.append(
                {
                    "accession": accession,
                    "scope": "ecoli_species",
                    "completeness": 99.0,
                    "contamination": 0.2,
                    "assembly_level": "Complete Genome",
                    "gtdb_representative": "t",
                    # The production normalized manifest retains this derived
                    # field in addition to the separately frozen membership.
                    "ani_cluster_99": f"ani99_{index:06d}",
                }
            )
            membership.append(
                {
                    "accession": accession,
                    "ani_cluster_99": f"ani99_{index:06d}",
                }
            )
            records.append(
                {
                    "accession": accession,
                    "contig_id": "chromosome",
                    "sequence": ("ACGT" * 30)[: 100 + index % 7],
                    "split": split,
                    "clade_group": f"ani99:ani99_{index:06d}",
                }
            )
    materialize_token_stream_dataset(
        records,
        tokenizer,
        dataset_dir,
        tokens_per_shard=10_000,
    )
    pair_rows = []
    accessions = [_accession(index) for values in split_ranges.values() for index in values]
    for pair_index, (left, right) in enumerate(itertools.combinations(accessions, 2)):
        pair_rows.append(
            {
                "Ref_file": f"/genomes/{left}.fna",
                "Query_file": f"/genomes/{right}.fna",
                "ANI": 96.0 + (pair_index % 250) / 100.0,
            }
        )
    return (
        dataset_dir,
        pd.DataFrame(rows),
        pd.DataFrame(membership),
        pd.DataFrame(pair_rows),
    )


def test_complete_ecoli_panels_are_nested_hashed_and_dataset_linked(tmp_path) -> None:
    dataset_dir, accessions, membership, pairs = _panel_fixture(tmp_path)
    paths = freeze_ecoli_panels(
        dataset_dir=dataset_dir,
        accession_manifest=accessions,
        ani_membership=membership,
        ani_pairs=pairs,
        output_dir=tmp_path / "panels",
        targets={"e25": 180, "e100": 360, "e250": 540},
        seed=31,
    )
    dataset = TokenStreamDataset(dataset_dir, verify_checksums=True)
    e25 = StageCPanelManifest.from_path(paths["e25"])
    e100 = StageCPanelManifest.from_path(paths["e100"])
    additions = StageCPanelManifest.from_path(paths["e100_additions"])
    validation = StageCPanelManifest.from_path(paths["validation"])
    test = StageCPanelManifest.from_path(paths["test"])

    for panel in (e25, e100, additions, validation, test):
        validate_panel_against_dataset(panel, dataset)
        assert len(panel.hash) == 64
    assert set(e25.payload["accessions"]) < set(e100.payload["accessions"])
    assert set(additions.payload["accessions"]) == (
        set(e100.payload["accessions"]) - set(e25.payload["accessions"])
    )
    assert len(validation.payload["accessions"]) == 8
    assert len(test.payload["accessions"]) == 8
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["panels"]["e25"]["sha256"]


def test_panel_freeze_rejects_draft_or_low_quality_assemblies(tmp_path) -> None:
    dataset_dir, accessions, membership, pairs = _panel_fixture(tmp_path)
    accessions["assembly_level"] = "Scaffold"
    with pytest.raises(ValueError, match="no complete E. coli"):
        freeze_ecoli_panels(
            dataset_dir=dataset_dir,
            accession_manifest=accessions,
            ani_membership=membership,
            ani_pairs=pairs,
            output_dir=tmp_path / "panels",
            targets={"e25": 100, "e100": 200, "e250": 300},
        )


def test_stateful_rotation_preserves_stream_order_and_resume_cursor(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    tokenizer = SeqTrainerBaseTokenizer()
    records = [
        {
            "accession": _accession(index),
            "contig_id": "chromosome",
            "sequence": "ACGT" * 90,
            "split": "train",
            "clade_group": f"ani99:{index}",
        }
        for index in range(1, 4)
    ]
    materialize_token_stream_dataset(records, tokenizer, dataset_dir)
    streams = TokenStreamDataset(dataset_dir).streams(split="train")
    scheduler = StatefulRotationScheduler(
        streams,
        batch_size=1,
        burst_segments=2,
        seed=7,
        shuffle=False,
    )
    observed = []
    for _ in range(5):
        segment = scheduler.next_batch()[0]
        observed.append((segment.accession, segment.segment_index))
    assert observed == [
        (_accession(1), 0),
        (_accession(1), 1),
        (_accession(2), 0),
        (_accession(2), 1),
        (_accession(3), 0),
    ]
    state = scheduler.to_state_dict()
    restored = StatefulRotationScheduler(
        streams,
        batch_size=1,
        burst_segments=2,
        seed=7,
        shuffle=False,
    )
    restored.load_state_dict(state)
    assert restored.next_batch()[0] == scheduler.next_batch()[0]


def test_base_cosine_schedule_uses_cumulative_bases() -> None:
    schedule = BaseCosineLRSchedule(
        peak_lr=3e-5,
        minimum_lr=3e-6,
        warmup_bases=2_000_000,
        decay_bases=100_000_000,
    )
    assert schedule(1_000_000) == pytest.approx(1.5e-5)
    assert schedule(2_000_000) == pytest.approx(3e-5)
    assert schedule(100_000_000) == pytest.approx(3e-6)
    assert schedule(150_000_000) == pytest.approx(3e-6)
