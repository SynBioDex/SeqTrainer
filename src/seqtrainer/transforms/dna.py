"""Composable DNA transforms."""

from __future__ import annotations

import itertools
from collections import Counter

import numpy as np

ALPHABET = ("A", "C", "G", "T", "N")


def normalize_sequence(sequence: str) -> str:
    """Uppercase sequence and replace unknown symbols with N."""
    return "".join(base if base in ALPHABET else "N" for base in sequence.upper())


def pad_or_trim(sequence: str, length: int, pad_char: str = "N") -> str:
    """Center trim or center pad sequence to a fixed length."""
    seq = normalize_sequence(sequence)
    if len(seq) > length:
        diff = len(seq) - length
        left = diff // 2
        right = diff - left
        return seq[left : len(seq) - right]
    return seq.center(length, pad_char)


def one_hot_encode(sequences: list[str], alphabet: tuple[str, ...] = ALPHABET) -> np.ndarray:
    """One-hot encode normalized sequences to [N, L, C] tensor."""
    if not sequences:
        return np.zeros((0, 0, len(alphabet)), dtype=np.float32)
    normalized = [normalize_sequence(s) for s in sequences]
    seq_len = len(normalized[0])
    if any(len(s) != seq_len for s in normalized):
        raise ValueError("All sequences must have the same length")

    mapping = {token: idx for idx, token in enumerate(alphabet)}
    encoded = np.zeros((len(normalized), seq_len, len(alphabet)), dtype=np.float32)
    for i, seq in enumerate(normalized):
        for j, token in enumerate(seq):
            encoded[i, j, mapping[token]] = 1.0
    return encoded


def gc_content(sequence: str) -> float:
    """Calculate GC fraction over canonical nucleotides only."""
    seq = normalize_sequence(sequence)
    valid = [b for b in seq if b in {"A", "C", "G", "T"}]
    if not valid:
        return 0.0
    gc = sum(1 for b in valid if b in {"G", "C"})
    return gc / len(valid)


def kmer_counts(sequence: str, k: int, normalize: bool = True) -> dict[str, float]:
    """Count k-mer occurrences using A/C/G/T vocabulary."""
    if k <= 0:
        raise ValueError("k must be positive")
    seq = normalize_sequence(sequence).replace("N", "")
    kmers = ["".join(c) for c in itertools.product("ACGT", repeat=k)]
    counts = Counter(seq[i : i + k] for i in range(len(seq) - k + 1))
    if normalize:
        total = max(len(seq) - k + 1, 1)
        return {kmer: counts.get(kmer, 0) / total for kmer in kmers}
    return {kmer: float(counts.get(kmer, 0)) for kmer in kmers}
