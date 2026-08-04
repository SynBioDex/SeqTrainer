"""State ownership and save/resume harness for synthetic paper-MAC streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Mapping, MutableMapping, TypeVar

from .synthetic import SyntheticSegment


StateT = TypeVar("StateT")


@dataclass(frozen=True)
class LifecycleTransition:
    """Audit record for one state transition owned by a single stream."""

    stream_id: str
    segment_index: int
    reset_applied: bool
    end_of_stream: bool


class StreamLifecycleHarness(Generic[StateT]):
    """Apply segments to states keyed exclusively by ``stream_id``.

    The caller supplies the state update operation, so this harness remains
    independent of MAC attention and can test the Stage A fast-memory state
    directly.  A reset is applied before its segment and an end marker after
    it.  The harness rejects reordered segments and any attempt to append to an
    ended stream.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        *,
        initial_state: Callable[[str], StateT],
        update_state: Callable[[StateT, SyntheticSegment], StateT],
        reset_state: Callable[[StateT], StateT],
        end_state: Callable[[StateT], StateT],
        serialize_state: Callable[[StateT], object],
        deserialize_state: Callable[[object], StateT],
    ) -> None:
        self._initial_state = initial_state
        self._update_state = update_state
        self._reset_state = reset_state
        self._end_state = end_state
        self._serialize_state = serialize_state
        self._deserialize_state = deserialize_state
        self._states: MutableMapping[str, StateT] = {}
        self._next_segment_index: MutableMapping[str, int] = {}
        self._ended_streams: set[str] = set()

    @property
    def states(self) -> Mapping[str, StateT]:
        """A shallow snapshot of stream-keyed state ownership."""

        return dict(self._states)

    def process(self, segment: SyntheticSegment) -> LifecycleTransition:
        """Process exactly one ordered segment without mixing any stream state."""

        stream_id = segment.stream_id
        if stream_id in self._ended_streams:
            raise RuntimeError(f"stream {stream_id!r} has already ended")
        expected_index = self._next_segment_index.get(stream_id, 0)
        if segment.segment_index != expected_index:
            raise ValueError(
                f"stream {stream_id!r} expected segment {expected_index}, got {segment.segment_index}"
            )
        state = self._states.get(stream_id)
        if state is None:
            state = self._initial_state(stream_id)
        reset_applied = segment.reset
        if reset_applied:
            state = self._reset_state(state)
        state = self._update_state(state, segment)
        if segment.end_of_stream:
            state = self._end_state(state)
            self._ended_streams.add(stream_id)
        self._states[stream_id] = state
        self._next_segment_index[stream_id] = expected_index + 1
        return LifecycleTransition(
            stream_id=stream_id,
            segment_index=segment.segment_index,
            reset_applied=reset_applied,
            end_of_stream=segment.end_of_stream,
        )

    def process_batch(self, segments: tuple[SyntheticSegment, ...]) -> tuple[LifecycleTransition, ...]:
        """Process an already scheduled mixed-stream batch in its supplied order."""

        return tuple(self.process(segment) for segment in segments)

    def to_state_dict(self) -> dict[str, object]:
        """Serialize stream ownership, order bookkeeping, and each state exactly."""

        return {
            "format_version": self.FORMAT_VERSION,
            "states": {stream_id: self._serialize_state(state) for stream_id, state in self._states.items()},
            "next_segment_index": dict(self._next_segment_index),
            "ended_streams": sorted(self._ended_streams),
        }

    @classmethod
    def from_state_dict(
        cls,
        payload: Mapping[str, object],
        *,
        initial_state: Callable[[str], StateT],
        update_state: Callable[[StateT, SyntheticSegment], StateT],
        reset_state: Callable[[StateT], StateT],
        end_state: Callable[[StateT], StateT],
        serialize_state: Callable[[StateT], object],
        deserialize_state: Callable[[object], StateT],
    ) -> "StreamLifecycleHarness[StateT]":
        """Restore a harness and retain its strict per-stream ordering checks."""

        if payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("unsupported stream lifecycle state version")
        raw_states = payload.get("states")
        raw_next = payload.get("next_segment_index")
        raw_ended = payload.get("ended_streams")
        if not isinstance(raw_states, Mapping) or not isinstance(raw_next, Mapping) or not isinstance(raw_ended, list):
            raise ValueError("invalid stream lifecycle state payload")
        harness = cls(
            initial_state=initial_state,
            update_state=update_state,
            reset_state=reset_state,
            end_state=end_state,
            serialize_state=serialize_state,
            deserialize_state=deserialize_state,
        )
        harness._states = {str(stream_id): deserialize_state(state) for stream_id, state in raw_states.items()}
        harness._next_segment_index = {str(stream_id): int(index) for stream_id, index in raw_next.items()}
        harness._ended_streams = {str(stream_id) for stream_id in raw_ended}
        if set(harness._states) != set(harness._next_segment_index):
            raise ValueError("state payload has inconsistent stream ownership")
        if not harness._ended_streams.issubset(harness._states):
            raise ValueError("state payload marks an unknown stream as ended")
        return harness
