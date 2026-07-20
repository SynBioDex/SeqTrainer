"""Informative CPU-only nucleotide baselines for the Stage C gate."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math
from typing import Iterable

from .tokenizers import normalize_dna


@dataclass(frozen=True)
class BaselineResult:
    name: str
    order: int
    bits_per_base: float
    bases: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MarkovDNABaseline:
    """Laplace-smoothed character model whose order zero is nucleotide frequency."""

    ALPHABET = "ACGTN"

    def __init__(self, order: int, smoothing: float = 1.0) -> None:
        if order < 0 or smoothing <= 0:
            raise ValueError("order must be non-negative and smoothing positive")
        self.order = order
        self.smoothing = smoothing
        self.counts: dict[str, Counter[str]] = defaultdict(Counter)

    def fit(self, sequences: Iterable[str]) -> "MarkovDNABaseline":
        for sequence in sequences:
            normalized = normalize_dna(sequence)
            for index, target in enumerate(normalized):
                context = normalized[max(0, index - self.order) : index] if self.order else ""
                self.counts[context][target] += 1
        return self

    def _probability(self, context: str, target: str) -> float:
        while context not in self.counts and context:
            context = context[1:]
        counts = self.counts.get(context, Counter())
        return (counts[target] + self.smoothing) / (
            sum(counts.values()) + self.smoothing * len(self.ALPHABET)
        )

    def evaluate(self, sequences: Iterable[str]) -> BaselineResult:
        nll = 0.0
        bases = 0
        for sequence in sequences:
            normalized = normalize_dna(sequence)
            for index, target in enumerate(normalized):
                context = normalized[max(0, index - self.order) : index] if self.order else ""
                nll -= math.log(self._probability(context, target))
                bases += 1
        if not bases:
            raise ValueError("baseline evaluation requires bases")
        return BaselineResult(
            name="nucleotide_frequency" if self.order == 0 else f"markov_order_{self.order}",
            order=self.order,
            bits_per_base=nll / (bases * math.log(2.0)),
            bases=bases,
        )


def uniform_baseline(sequences: Iterable[str]) -> BaselineResult:
    bases = sum(len(normalize_dna(sequence)) for sequence in sequences)
    if not bases:
        raise ValueError("uniform baseline requires bases")
    return BaselineResult(name="uniform", order=-1, bits_per_base=math.log2(5), bases=bases)


def run_statistical_baselines(
    train_sequences: Iterable[str],
    evaluation_sequences: Iterable[str],
) -> tuple[BaselineResult, ...]:
    train = tuple(train_sequences)
    evaluation = tuple(evaluation_sequences)
    results = [uniform_baseline(evaluation)]
    for order in (0, 1, 3, 5):
        results.append(MarkovDNABaseline(order).fit(train).evaluate(evaluation))
    return tuple(results)

