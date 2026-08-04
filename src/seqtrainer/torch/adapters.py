"""PyTorch dataset adapters."""

from __future__ import annotations

from seqtrainer.data.materialized import MaterializedDataset


class SequenceDataset:
    """Minimal torch-style dataset over sequence records."""

    def __init__(self, records: list[dict]):
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int):
        row = self._records[idx]
        return row.get("sequence"), row.get("target")


def to_torch_dataset(dataset: MaterializedDataset) -> SequenceDataset:
    return SequenceDataset(dataset.examples)
