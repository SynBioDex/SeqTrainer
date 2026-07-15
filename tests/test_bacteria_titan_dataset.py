import numpy as np
import pandas as pd
import pytest

from seqtrainer.data.bacteria_titan import (
    TokenShardDataset,
    assert_no_accession_leakage,
    prepare_candidates,
    sample_accessions,
    split_accessions,
)


def taxonomy(species: str, genus: str = "Escherichia", family: str = "Enterobacteriaceae") -> str:
    return f"d__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria;o__Enterobacterales;f__{family};g__{genus};s__{species}"


def candidate_frame() -> pd.DataFrame:
    rows = []
    scopes = [
        ("Escherichia coli", "Escherichia", "Enterobacteriaceae"),
        ("Escherichia albertii", "Escherichia", "Enterobacteriaceae"),
        ("Klebsiella pneumoniae", "Klebsiella", "Enterobacteriaceae"),
        ("Yersinia pestis", "Yersinia", "Yersiniaceae"),
    ]
    for scope_index, (species, genus, family) in enumerate(scopes):
        for index in range(8):
            rows.append(
                {
                    "accession": f"GCF_{scope_index:03d}{index:03d}.1",
                    "gtdb_taxonomy": taxonomy(species, genus, family),
                    "checkm_completeness": 98.0 - index / 10,
                    "checkm_contamination": 1.0,
                    "genome_size": 1_000_000,
                    "gtdb_representative": "t" if index == 0 else "f",
                    "organism_name": f"{species} strain {index}",
                }
            )
    return pd.DataFrame(rows)


def test_deterministic_sampling_and_accession_level_splits() -> None:
    candidates = prepare_candidates(candidate_frame())
    first = sample_accessions(candidates, target_bp=12_000_000, seed=31)
    second = sample_accessions(candidates, target_bp=12_000_000, seed=31)
    assert first["accession"].tolist() == second["accession"].tolist()
    assert set(first["scope"]) == {
        "ecoli_species",
        "escherichia_genus",
        "enterobacteriaceae_family",
        "enterobacterales_order",
    }
    split = split_accessions(candidates, seed=31)
    assert_no_accession_leakage(split)
    assert not split["accession"].duplicated().any()
    assert set(split["split"]) == {"train", "val", "test"}


def test_split_leakage_check_rejects_duplicate_accessions() -> None:
    frame = pd.DataFrame(
        {"accession": ["GCF_1", "GCF_1"], "split": ["train", "test"]}
    )
    with pytest.raises(ValueError, match="leakage"):
        assert_no_accession_leakage(frame)


def test_token_shard_dataset_loading(tmp_path) -> None:
    first = np.arange(24, dtype=np.uint8).reshape(4, 6) % 6
    second = np.arange(12, dtype=np.uint8).reshape(2, 6) % 6
    np.save(tmp_path / "tokens_00000.npy", first)
    np.save(tmp_path / "tokens_00001.npy", second)
    dataset = TokenShardDataset(tmp_path)
    assert len(dataset) == 6
    input_ids, labels = dataset[4]
    assert input_ids.dtype == np.int64
    assert input_ids.shape == labels.shape == (5,)
    assert np.array_equal(input_ids[1:], labels[:-1])
