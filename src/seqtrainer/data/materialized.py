"""Materialized in-memory dataset abstraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any

import pandas as pd


@dataclass
class MaterializedDataset:
    """Container for examples and metadata produced from a recipe."""

    examples: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_pandas(self) -> pd.DataFrame:
        """Convert examples to a pandas DataFrame."""
        return pd.DataFrame(self.examples)

    def train_val_test_split(
        self,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        seed: int = 42,
    ) -> tuple["MaterializedDataset", "MaterializedDataset", "MaterializedDataset"]:
        """Create shuffled train/val/test splits."""
        if round(train_size + val_size + test_size, 7) != 1.0:
            raise ValueError("train_size + val_size + test_size must equal 1.0")
        rng = Random(seed)
        items = list(self.examples)
        rng.shuffle(items)
        n = len(items)
        train_end = int(n * train_size)
        val_end = train_end + int(n * val_size)
        return (
            MaterializedDataset(items[:train_end], metadata={**self.metadata, "split": "train"}),
            MaterializedDataset(items[train_end:val_end], metadata={**self.metadata, "split": "val"}),
            MaterializedDataset(items[val_end:], metadata={**self.metadata, "split": "test"}),
        )
