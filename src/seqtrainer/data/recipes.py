"""Dataset recipe declarations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


LabelExtractor = Callable[[dict[str, Any]], Any]


@dataclass(slots=True)
class DatasetRecipe:
    """Declarative description of how a synbio dataset should be materialized."""

    name: str
    query: str
    sequence_field: str = "sequence"
    label_field: str | None = None
    label_extractor: LabelExtractor | None = None
    metadata_fields: tuple[str, ...] = field(default_factory=tuple)
    provenance: dict[str, Any] = field(default_factory=dict)

    def extract_label(self, row: dict[str, Any]) -> Any:
        """Get row label using either label field or extractor callback."""
        if self.label_extractor is not None:
            return self.label_extractor(row)
        if self.label_field is None:
            return None
        return row.get(self.label_field)
