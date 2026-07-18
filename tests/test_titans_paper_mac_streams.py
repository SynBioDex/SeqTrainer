from __future__ import annotations

from dataclasses import dataclass

from seqtrainer.torch.titans_paper_mac import (  # noqa: E402
    StreamLifecycleHarness,
    build_stage_a_fixtures,
    context_boundary_reset,
    delayed_key_value_recall,
    overwrite_forgetting,
    score_query_predictions,
)


@dataclass(frozen=True)
class TraceState:
    stream_id: str
    history: tuple[tuple[int, tuple[int, ...]], ...] = ()
    reset_count: int = 0
    ended: bool = False

    def to_state_dict(self) -> dict[str, object]:
        return {
            "stream_id": self.stream_id,
            "history": self.history,
            "reset_count": self.reset_count,
            "ended": self.ended,
        }

    @classmethod
    def from_state_dict(cls, payload: object) -> "TraceState":
        if not isinstance(payload, dict):
            raise ValueError("invalid trace state")
        return cls(
            stream_id=str(payload["stream_id"]),
            history=tuple((int(index), tuple(tokens)) for index, tokens in payload["history"]),
            reset_count=int(payload["reset_count"]),
            ended=bool(payload["ended"]),
        )


def _initial(stream_id: str) -> TraceState:
    return TraceState(stream_id)


def _update(state: TraceState, segment) -> TraceState:
    assert state.stream_id == segment.stream_id
    return TraceState(
        stream_id=state.stream_id,
        history=state.history + ((segment.segment_index, segment.tokens),),
        reset_count=state.reset_count,
        ended=False,
    )


def _reset(state: TraceState) -> TraceState:
    return TraceState(stream_id=state.stream_id, reset_count=state.reset_count + 1)


def _end(state: TraceState) -> TraceState:
    return TraceState(
        stream_id=state.stream_id,
        history=state.history,
        reset_count=state.reset_count,
        ended=True,
    )


def _harness() -> StreamLifecycleHarness[TraceState]:
    return StreamLifecycleHarness(
        initial_state=_initial,
        update_state=_update,
        reset_state=_reset,
        end_state=_end,
        serialize_state=lambda state: state.to_state_dict(),
        deserialize_state=TraceState.from_state_dict,
    )


def _restore(payload: dict[str, object]) -> StreamLifecycleHarness[TraceState]:
    return StreamLifecycleHarness.from_state_dict(
        payload,
        initial_state=_initial,
        update_state=_update,
        reset_state=_reset,
        end_state=_end,
        serialize_state=lambda state: state.to_state_dict(),
        deserialize_state=TraceState.from_state_dict,
    )


def test_all_synthetic_tasks_are_reproducible_and_scoreable() -> None:
    fixtures = build_stage_a_fixtures(seed=123, num_streams=3)
    assert fixtures == build_stage_a_fixtures(seed=123, num_streams=3)
    assert set(fixtures) == {"delayed_recall", "overwrite", "boundary_reset"}

    for fixture in fixtures.values():
        predictions = {query.query_id: query.target_token for query in fixture.query_segments()}
        score = score_query_predictions(fixture, predictions)
        assert score.correct == score.total == 3
        assert score.missing == 0


def test_delayed_targets_are_absent_and_outside_the_current_32_token_segment() -> None:
    fixture = delayed_key_value_recall(seed=31, num_streams=2)
    for query in fixture.query_segments():
        assert query.target_token not in query.tokens
        assert query.source_segment_index == 0
        source_absolute_position = query.source_segment_index * 32 + 1
        query_absolute_position = query.segment_index * 32 + query.query_position
        assert query_absolute_position - source_absolute_position > 32


def test_stream_level_shuffle_preserves_each_stream_order() -> None:
    fixture = overwrite_forgetting(seed=41, num_streams=4)
    shuffled = fixture.stream_level_shuffle(seed=99)
    per_stream = {stream_id: [] for stream_id in fixture.stream_ids}
    for segment in shuffled:
        per_stream[segment.stream_id].append(segment.segment_index)
    assert all(indices == [0, 1, 2] for indices in per_stream.values())


def test_interleaved_streams_never_share_state() -> None:
    fixture = delayed_key_value_recall(seed=51, num_streams=2)
    interleaved = fixture.round_robin_interleave()
    mixed = _harness()
    for segment in interleaved:
        mixed.process(segment)

    expected = _harness()
    for stream in fixture.streams:
        for segment in stream:
            expected.process(segment)
    assert mixed.states == expected.states
    assert {state.stream_id for state in mixed.states.values()} == set(fixture.stream_ids)


def test_reset_clears_only_the_target_stream() -> None:
    reset_fixture = context_boundary_reset(seed=61, num_streams=1)
    untouched_fixture = delayed_key_value_recall(seed=62, num_streams=1)
    harness = _harness()
    reset_stream = reset_fixture.streams[0]
    untouched_stream = untouched_fixture.streams[0]
    schedule = (
        reset_stream[0],
        untouched_stream[0],
        reset_stream[1],
        untouched_stream[1],
        untouched_stream[2],
    )
    for segment in schedule:
        harness.process(segment)

    reset_id = reset_fixture.stream_ids[0]
    untouched_id = untouched_fixture.stream_ids[0]
    assert harness.states[reset_id].reset_count == 1
    assert harness.states[reset_id].history == ((1, reset_stream[1].tokens),)
    assert harness.states[untouched_id].reset_count == 0
    assert harness.states[untouched_id].history == tuple(
        (segment.segment_index, segment.tokens) for segment in untouched_stream
    )


def test_saved_resume_execution_matches_sequential_state_transitions() -> None:
    fixture = delayed_key_value_recall(seed=71, num_streams=2)
    schedule = fixture.round_robin_interleave()
    sequential = _harness()
    for segment in schedule:
        sequential.process(segment)

    resumed = _harness()
    split = len(schedule) // 2
    for segment in schedule[:split]:
        resumed.process(segment)
    resumed = _restore(resumed.to_state_dict())
    for segment in schedule[split:]:
        resumed.process(segment)

    assert resumed.to_state_dict() == sequential.to_state_dict()
