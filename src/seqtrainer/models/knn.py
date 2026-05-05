"""Nearest-neighbor retrieval models for sequence design tasks.

The retrievers in this module are intentionally lightweight: they fit a
scikit-learn nearest-neighbor index over scalar labels, then return the stored
sequence records whose labels are closest to a requested target value. This is
useful for design workflows such as: "find promoters near activity 0.4".
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class ScalarKNNRetriever:
    """Retrieve records whose scalar value is closest to a query value.

    Parameters
    ----------
    n_neighbors:
        Default number of nearest records returned by :meth:`search`.
    metric:
        Distance metric passed to ``sklearn.neighbors.NearestNeighbors``.
        For one-dimensional scalar values, ``"euclidean"`` and ``"manhattan"``
        both rank records by absolute error.
    algorithm:
        Neighbor search algorithm passed to scikit-learn.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "euclidean",
        algorithm: str = "auto",
    ) -> None:
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.algorithm = algorithm
        self.value_column: str | None = None
        self.sequence_column: str | None = None
        self.metadata_columns: list[str] | None = None
        self.records_: pd.DataFrame | None = None
        self.values_: np.ndarray | None = None
        self.neighbor_index_: NearestNeighbors | None = None

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        value_column: str = "label",
        sequence_column: str = "sequence",
        metadata_columns: Iterable[str] | None = None,
        **kwargs,
    ) -> "ScalarKNNRetriever":
        """Build and fit a retriever from a CSV file."""
        retriever = cls(**kwargs)
        return retriever.fit(
            pd.read_csv(path),
            value_column=value_column,
            sequence_column=sequence_column,
            metadata_columns=metadata_columns,
        )

    def fit(
        self,
        data: pd.DataFrame,
        *,
        value_column: str = "label",
        sequence_column: str = "sequence",
        metadata_columns: Iterable[str] | None = None,
    ) -> "ScalarKNNRetriever":
        """Fit the nearest-neighbor index from a dataframe.

        The dataframe must include a DNA sequence column and a numeric scalar
        value column, for example ``sequence`` and ``label`` for promoter
        activity retrieval.
        """
        self._validate_columns(data, value_column, sequence_column, metadata_columns)

        columns = [sequence_column, value_column]
        if metadata_columns is not None:
            columns.extend(c for c in metadata_columns if c not in columns)

        records = data.loc[:, columns].copy()
        records[sequence_column] = records[sequence_column].astype(str)
        records[value_column] = pd.to_numeric(records[value_column], errors="coerce")
        records = records.dropna(subset=[sequence_column, value_column]).reset_index(drop=True)

        if records.empty:
            raise ValueError("Cannot fit KNN retriever with no valid records")

        values = records[value_column].to_numpy(dtype=float).reshape(-1, 1)
        n_neighbors = min(self.n_neighbors, len(records))
        neighbor_index = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=self.metric,
            algorithm=self.algorithm,
        )
        neighbor_index.fit(values)

        self.value_column = value_column
        self.sequence_column = sequence_column
        self.metadata_columns = list(metadata_columns) if metadata_columns is not None else None
        self.records_ = records
        self.values_ = values
        self.neighbor_index_ = neighbor_index
        return self

    def search(self, target_value: float, top_k: int | None = None) -> pd.DataFrame:
        """Return ranked records closest to ``target_value``.

        The returned dataframe includes ``rank``, ``query_value``, and
        ``distance`` columns followed by the original record columns. ``distance``
        is the absolute error between the record value and the query value.
        """
        self._require_fitted()
        if top_k is None:
            top_k = self.n_neighbors
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        assert self.records_ is not None
        assert self.neighbor_index_ is not None
        assert self.value_column is not None

        n_neighbors = min(int(top_k), len(self.records_))
        _distances, indices = self.neighbor_index_.kneighbors(
            np.array([[float(target_value)]], dtype=float),
            n_neighbors=n_neighbors,
        )

        ranked = self.records_.iloc[indices[0]].copy().reset_index(drop=True)
        ranked.insert(0, "distance", np.abs(ranked[self.value_column].astype(float) - float(target_value)))
        ranked.insert(0, "query_value", float(target_value))
        ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

        # Stable, explicit ordering in case multiple rows have the same nearest-neighbor distance.
        ranked = ranked.sort_values(["distance", self.value_column], kind="mergesort").reset_index(drop=True)
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        return ranked

    def save(self, path: str | Path) -> Path:
        """Serialize a fitted retriever to disk with pickle."""
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ScalarKNNRetriever":
        """Load a serialized retriever from disk."""
        with Path(path).open("rb") as handle:
            loaded = pickle.load(handle)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}")
        loaded._require_fitted()
        return loaded

    def _validate_columns(
        self,
        data: pd.DataFrame,
        value_column: str,
        sequence_column: str,
        metadata_columns: Iterable[str] | None,
    ) -> None:
        missing = [column for column in [sequence_column, value_column] if column not in data.columns]
        if metadata_columns is not None:
            missing.extend(column for column in metadata_columns if column not in data.columns)
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(sorted(set(missing)))}")

    def _require_fitted(self) -> None:
        if self.records_ is None or self.values_ is None or self.neighbor_index_ is None:
            raise RuntimeError("Fit the KNN retriever before searching or saving")


class PromoterActivityKNN(ScalarKNNRetriever):
    """KNN retriever specialized for promoter activity datasets.

    By default, this expects a dataframe or CSV with ``sequence`` and ``label``
    columns and returns promoters whose observed activity labels are closest to
    the requested activity value.
    """

    def fit(
        self,
        data: pd.DataFrame,
        *,
        value_column: str = "label",
        sequence_column: str = "sequence",
        metadata_columns: Iterable[str] | None = None,
    ) -> "PromoterActivityKNN":
        return super().fit(
            data,
            value_column=value_column,
            sequence_column=sequence_column,
            metadata_columns=metadata_columns,
        )


def build_scalar_knn_retriever(
    data: pd.DataFrame,
    *,
    value_column: str,
    sequence_column: str = "sequence",
    metadata_columns: Iterable[str] | None = None,
    n_neighbors: int = 5,
    metric: str = "euclidean",
    algorithm: str = "auto",
) -> ScalarKNNRetriever:
    """Convenience factory for scalar-label nearest-neighbor retrieval."""
    return ScalarKNNRetriever(
        n_neighbors=n_neighbors,
        metric=metric,
        algorithm=algorithm,
    ).fit(
        data,
        value_column=value_column,
        sequence_column=sequence_column,
        metadata_columns=metadata_columns,
    )
