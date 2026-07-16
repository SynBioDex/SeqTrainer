"""Biopython to SBOL coordinate conversion helpers."""

from __future__ import annotations

from typing import Any


def sbol_ranges_for_location(location: Any, sequence_length: int) -> list[dict[str, Any]]:
    """Return 1-based inclusive SBOL ranges in biological part order."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    parts = list(getattr(location, "parts", None) or [location])
    orientation = getattr(location, "strand", None)
    ranges = []
    for part in parts:
        start = int(part.start) + 1
        end = int(part.end)
        if not (1 <= start <= end <= sequence_length):
            raise ValueError(f"Location {start}..{end} is outside sequence length {sequence_length}")
        ranges.append({"start": start, "end": end, "orientation": orientation})
    return ranges


def sbol_orientation(strand: int | None) -> str | None:
    if strand == 1:
        return "http://sbols.org/v3#inline"
    if strand == -1:
        return "http://sbols.org/v3#reverseComplement"
    return None
