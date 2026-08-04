"""Leakage-safe accession-level train/validation/test splitting."""

from __future__ import annotations

import hashlib
from typing import Mapping

import pandas as pd


DEFAULT_SPLIT_FRACTIONS = {"train": 0.90, "val": 0.05, "test": 0.05}


def _stable_key(accession: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{accession}".encode("utf-8")).hexdigest()


def split_accessions(
    accessions: pd.DataFrame,
    fractions: Mapping[str, float] = DEFAULT_SPLIT_FRACTIONS,
    seed: int = 17,
    stratify_column: str = "scope",
) -> pd.DataFrame:
    """Assign each complete accession to exactly one deterministic split."""

    if set(fractions) != {"train", "val", "test"} or abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("fractions must define train/val/test and sum to 1")
    if accessions["accession"].duplicated().any():
        raise ValueError("accessions must be unique before splitting")
    frame = accessions.copy()
    frame["split"] = ""
    groups = frame.groupby(stratify_column, dropna=False, sort=True) if stratify_column in frame else [("all", frame)]
    for _, group in groups:
        ordered = sorted(group.index, key=lambda index: _stable_key(str(frame.at[index, "accession"]), seed))
        count = len(ordered)
        val_count = round(count * fractions["val"])
        test_count = round(count * fractions["test"])
        if count >= 3:
            val_count = max(1, val_count)
            test_count = max(1, test_count)
        if val_count + test_count >= count:
            val_count = 1 if count >= 2 else 0
            test_count = 1 if count >= 3 else 0
        train_count = count - val_count - test_count
        assignments = ["train"] * train_count + ["val"] * val_count + ["test"] * test_count
        for index, split in zip(ordered, assignments):
            frame.at[index, "split"] = split
    assert_no_accession_leakage(frame)
    return frame.sort_values(["split", "accession"], kind="stable").reset_index(drop=True)


def assert_no_accession_leakage(frame: pd.DataFrame) -> None:
    if frame["accession"].duplicated().any():
        duplicates = frame.loc[frame["accession"].duplicated(False), "accession"].unique()
        raise ValueError(f"accession leakage detected: {duplicates[:5].tolist()}")
    if not set(frame["split"]).issubset({"train", "val", "test"}):
        raise ValueError("every accession must be assigned to train, val, or test")
