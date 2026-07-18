"""Deterministic Stage A stream fixtures for fast-memory correctness tests.

The fixtures use fixed-length 32-token segments and a deliberately small,
disjoint vocabulary.  A query target is never present in its query segment, so
answering it requires state from an earlier segment rather than a shortcut in
the current context.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Mapping, Optional, Sequence


SEGMENT_LENGTH = 32


@dataclass(frozen=True)
class SyntheticVocabulary:
    """Fixed token contract for all Stage A synthetic tasks.

    ``query`` asks for the value stored under the following key.  ``no_memory``
    is the correct answer after an explicit reset.  Keys and values deliberately
    occupy separate ranges, and ``filler`` is outside both ranges.
    """

    pad: int = 0
    query: int = 1
    no_memory: int = 2
    key_start: int = 3
    key_stop: int = 10
    value_start: int = 11
    value_stop: int = 18
    filler: int = 19

    @property
    def size(self) -> int:
        return self.filler + 1

    @property
    def keys(self) -> tuple[int, ...]:
        return tuple(range(self.key_start, self.key_stop + 1))

    @property
    def values(self) -> tuple[int, ...]:
        return tuple(range(self.value_start, self.value_stop + 1))


DEFAULT_VOCABULARY = SyntheticVocabulary()


@dataclass(frozen=True)
class SyntheticSegment:
    """One ordered input segment for a named synthetic stream.

    ``reset`` is applied immediately before this segment.  ``end_of_stream`` is
    applied immediately after it.  Query segments carry the position and the
    expected target token for deterministic scoring.
    """

    task: str
    stream_id: str
    segment_index: int
    tokens: tuple[int, ...]
    query_position: Optional[int] = None
    target_token: Optional[int] = None
    reset: bool = False
    end_of_stream: bool = False
    source_segment_index: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.stream_id:
            raise ValueError("stream_id must be non-empty")
        if self.segment_index < 0:
            raise ValueError("segment_index must be non-negative")
        if len(self.tokens) != SEGMENT_LENGTH:
            raise ValueError(f"synthetic segments must contain exactly {SEGMENT_LENGTH} tokens")
        if (self.query_position is None) != (self.target_token is None):
            raise ValueError("query_position and target_token must be supplied together")
        if self.query_position is not None:
            if not 0 <= self.query_position < SEGMENT_LENGTH:
                raise ValueError("query_position is outside the segment")
            if self.target_token in self.tokens:
                raise ValueError("a query target must be absent from its current segment")
            if self.source_segment_index is None or self.source_segment_index >= self.segment_index:
                raise ValueError("a query must identify an earlier source segment")

    @property
    def query_id(self) -> tuple[str, int]:
        """Stable key used by scoring and prediction dictionaries."""

        return self.stream_id, self.segment_index


@dataclass(frozen=True)
class SyntheticScore:
    """Exact-match query score for a fixture."""

    correct: int
    total: int
    missing: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass(frozen=True)
class SyntheticTaskFixture:
    """A deterministic task containing independently owned ordered streams."""

    task: str
    seed: int
    vocabulary: SyntheticVocabulary
    streams: tuple[tuple[SyntheticSegment, ...], ...]

    def __post_init__(self) -> None:
        stream_ids: set[str] = set()
        for stream in self.streams:
            if not stream:
                raise ValueError("every fixture stream must contain at least one segment")
            stream_id = stream[0].stream_id
            if stream_id in stream_ids:
                raise ValueError("stream IDs must be unique within a fixture")
            stream_ids.add(stream_id)
            expected_indices = tuple(range(len(stream)))
            if tuple(segment.stream_id for segment in stream) != (stream_id,) * len(stream):
                raise ValueError("all segments in a stream must have the same stream_id")
            if tuple(segment.segment_index for segment in stream) != expected_indices:
                raise ValueError("stream segment indices must be contiguous and ordered")
            if not stream[-1].end_of_stream or any(segment.end_of_stream for segment in stream[:-1]):
                raise ValueError("only the final segment in a stream may end the stream")

    @property
    def stream_ids(self) -> tuple[str, ...]:
        return tuple(stream[0].stream_id for stream in self.streams)

    def ordered_segments(self) -> tuple[SyntheticSegment, ...]:
        """Return complete stream blocks in their fixture-defined order."""

        return tuple(segment for stream in self.streams for segment in stream)

    def stream_level_shuffle(self, seed: int) -> tuple[SyntheticSegment, ...]:
        """Shuffle complete streams only; intra-stream segment order never changes."""

        blocks = list(self.streams)
        random.Random(seed).shuffle(blocks)
        return tuple(segment for stream in blocks for segment in stream)

    def round_robin_interleave(self) -> tuple[SyntheticSegment, ...]:
        """Interleave streams while retaining each stream's individual order."""

        pending = [iter(stream) for stream in self.streams]
        result: list[SyntheticSegment] = []
        while pending:
            remaining: list[Iterable[SyntheticSegment]] = []
            for stream in pending:
                try:
                    result.append(next(stream))
                    remaining.append(stream)
                except StopIteration:
                    continue
            pending = remaining
        return tuple(result)

    def batches(self, segments: Sequence[SyntheticSegment], batch_size: int) -> tuple[tuple[SyntheticSegment, ...], ...]:
        """Group a pre-ordered schedule without reordering its individual streams."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return tuple(tuple(segments[index : index + batch_size]) for index in range(0, len(segments), batch_size))

    def query_segments(self) -> tuple[SyntheticSegment, ...]:
        return tuple(segment for segment in self.ordered_segments() if segment.target_token is not None)


def _tokens(*prefix: int, vocabulary: SyntheticVocabulary) -> tuple[int, ...]:
    if len(prefix) > SEGMENT_LENGTH:
        raise ValueError("synthetic prefix exceeds the segment length")
    return tuple(prefix) + (vocabulary.filler,) * (SEGMENT_LENGTH - len(prefix))


def _stream_pairs(
    rng: random.Random, num_streams: int, vocabulary: SyntheticVocabulary
) -> tuple[tuple[int, int], ...]:
    """Assign distinct keys and values so stream-ownership leaks are observable."""

    capacity = min(len(vocabulary.keys), len(vocabulary.values))
    if num_streams > capacity:
        raise ValueError(f"num_streams must be at most {capacity} for distinct synthetic mappings")
    keys = list(vocabulary.keys)
    values = list(vocabulary.values)
    rng.shuffle(keys)
    rng.shuffle(values)
    return tuple(zip(keys[:num_streams], values[:num_streams]))


def delayed_key_value_recall(
    *, seed: int = 20260717, num_streams: int = 2, vocabulary: SyntheticVocabulary = DEFAULT_VOCABULARY
) -> SyntheticTaskFixture:
    """Build recall queries separated from their writes by more than 32 tokens.

    Each stream writes a key/value pair in segment zero, has one complete filler
    segment, then queries the key in segment two.  The target's absolute source
    and query positions differ by at least 63 tokens.
    """

    if num_streams <= 0:
        raise ValueError("num_streams must be positive")
    rng = random.Random(seed)
    streams: list[tuple[SyntheticSegment, ...]] = []
    for index, (key, value) in enumerate(_stream_pairs(rng, num_streams, vocabulary)):
        stream_id = f"delayed-{index}"
        streams.append(
            (
                SyntheticSegment("delayed_recall", stream_id, 0, _tokens(key, value, vocabulary=vocabulary)),
                SyntheticSegment("delayed_recall", stream_id, 1, _tokens(vocabulary=vocabulary)),
                SyntheticSegment(
                    "delayed_recall",
                    stream_id,
                    2,
                    _tokens(vocabulary.query, key, vocabulary=vocabulary),
                    query_position=0,
                    target_token=value,
                    end_of_stream=True,
                    source_segment_index=0,
                ),
            )
        )
    return SyntheticTaskFixture("delayed_recall", seed, vocabulary, tuple(streams))


def overwrite_forgetting(
    *, seed: int = 20260718, num_streams: int = 2, vocabulary: SyntheticVocabulary = DEFAULT_VOCABULARY
) -> SyntheticTaskFixture:
    """Build streams where a second write must replace an earlier key value."""

    if num_streams <= 0:
        raise ValueError("num_streams must be positive")
    rng = random.Random(seed)
    streams: list[tuple[SyntheticSegment, ...]] = []
    pairs = _stream_pairs(rng, num_streams, vocabulary)
    all_values = tuple(vocabulary.values)
    for index, (key, old_value) in enumerate(pairs):
        new_value = all_values[(all_values.index(old_value) + 1) % len(all_values)]
        stream_id = f"overwrite-{index}"
        streams.append(
            (
                SyntheticSegment("overwrite", stream_id, 0, _tokens(key, old_value, vocabulary=vocabulary)),
                SyntheticSegment("overwrite", stream_id, 1, _tokens(key, new_value, vocabulary=vocabulary)),
                SyntheticSegment(
                    "overwrite",
                    stream_id,
                    2,
                    _tokens(vocabulary.query, key, vocabulary=vocabulary),
                    query_position=0,
                    target_token=new_value,
                    end_of_stream=True,
                    source_segment_index=1,
                ),
            )
        )
    return SyntheticTaskFixture("overwrite", seed, vocabulary, tuple(streams))


def context_boundary_reset(
    *, seed: int = 20260719, num_streams: int = 2, vocabulary: SyntheticVocabulary = DEFAULT_VOCABULARY
) -> SyntheticTaskFixture:
    """Build streams whose second segment resets state before a recall query."""

    if num_streams <= 0:
        raise ValueError("num_streams must be positive")
    rng = random.Random(seed)
    streams: list[tuple[SyntheticSegment, ...]] = []
    for index, (key, value) in enumerate(_stream_pairs(rng, num_streams, vocabulary)):
        stream_id = f"reset-{index}"
        streams.append(
            (
                SyntheticSegment("boundary_reset", stream_id, 0, _tokens(key, value, vocabulary=vocabulary)),
                SyntheticSegment(
                    "boundary_reset",
                    stream_id,
                    1,
                    _tokens(vocabulary.query, key, vocabulary=vocabulary),
                    query_position=0,
                    target_token=vocabulary.no_memory,
                    reset=True,
                    end_of_stream=True,
                    source_segment_index=0,
                ),
            )
        )
    return SyntheticTaskFixture("boundary_reset", seed, vocabulary, tuple(streams))


def build_stage_a_fixtures(
    *, seed: int = 20260717, num_streams: int = 2, vocabulary: SyntheticVocabulary = DEFAULT_VOCABULARY
) -> Mapping[str, SyntheticTaskFixture]:
    """Return all three reproducible Stage A fixture families.

    Consecutive seeds make each task deterministic while keeping task token
    assignments independent.
    """

    return {
        "delayed_recall": delayed_key_value_recall(seed=seed, num_streams=num_streams, vocabulary=vocabulary),
        "overwrite": overwrite_forgetting(seed=seed + 1, num_streams=num_streams, vocabulary=vocabulary),
        "boundary_reset": context_boundary_reset(seed=seed + 2, num_streams=num_streams, vocabulary=vocabulary),
    }


def score_query_predictions(
    fixture: SyntheticTaskFixture, predictions: Mapping[tuple[str, int], int]
) -> SyntheticScore:
    """Score only query segments; missing predictions count as incorrect."""

    queries = fixture.query_segments()
    correct = sum(predictions.get(query.query_id) == query.target_token for query in queries)
    missing = sum(query.query_id not in predictions for query in queries)
    return SyntheticScore(correct=correct, total=len(queries), missing=missing)
