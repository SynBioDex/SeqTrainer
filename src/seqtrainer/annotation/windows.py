"""Sliding-window generation for plasmid promoter annotation."""

from __future__ import annotations

from dataclasses import dataclass


_COMPLEMENT = str.maketrans("ACGTUNacgtun", "TGCAANtgcaan")


@dataclass(frozen=True)
class SequenceWindow:
    """A candidate promoter window extracted from an input sequence."""

    window_id: str
    start: int
    end: int
    strand: str
    sequence: str
    is_circular_boundary_window: bool


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement for DNA-like input."""
    return sequence.translate(_COMPLEMENT)[::-1].upper().replace("U", "T")


def normalize_dna(sequence: str) -> str:
    """Normalize sequence text to A/C/G/T/N."""
    normalized = sequence.upper().replace("U", "T")
    return "".join(base if base in {"A", "C", "G", "T", "N"} else "N" for base in normalized)


def generate_sliding_windows(
    sequence: str,
    *,
    window_size: int,
    step_size: int,
    circular: bool,
    scan_both_strands: bool = True,
) -> list[SequenceWindow]:
    """Generate fixed-size candidate windows over one or both strands."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if step_size <= 0:
        raise ValueError("step_size must be positive")

    normalized = normalize_dna(sequence)
    seq_len = len(normalized)
    if seq_len == 0:
        return []

    starts = range(0, seq_len, step_size) if circular else range(0, max(seq_len - window_size + 1, 0), step_size)
    windows: list[SequenceWindow] = []
    for start in starts:
        fragment, crosses_boundary = _slice_window(normalized, start, window_size, circular)
        if len(fragment) != window_size:
            continue
        end = start + window_size
        windows.append(
            SequenceWindow(
                window_id=f"window_{len(windows)}_plus",
                start=start,
                end=end,
                strand="+",
                sequence=fragment,
                is_circular_boundary_window=crosses_boundary,
            )
        )
        if scan_both_strands:
            windows.append(
                SequenceWindow(
                    window_id=f"window_{len(windows)}_minus",
                    start=start,
                    end=end,
                    strand="-",
                    sequence=reverse_complement(fragment),
                    is_circular_boundary_window=crosses_boundary,
                )
            )
    return windows


def _slice_window(sequence: str, start: int, window_size: int, circular: bool) -> tuple[str, bool]:
    end = start + window_size
    if end <= len(sequence):
        return sequence[start:end], False
    if not circular:
        return sequence[start:], False
    overhang = end - len(sequence)
    return sequence[start:] + sequence[:overhang], True

