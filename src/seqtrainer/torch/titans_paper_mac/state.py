"""Explicit, serializable state for one functional Titans paper-MAC stream."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping, Optional

import torch
from torch import Tensor


FastWeights = OrderedDict[str, Tensor]


def _copy_weights(
    weights: Mapping[str, Tensor],
    *,
    device: Optional[torch.device] = None,
    detach: bool,
) -> FastWeights:
    """Copy a fast-weight pytree while preserving tensor dtype and order."""

    copied: FastWeights = OrderedDict()
    for name, value in weights.items():
        tensor = value.detach() if detach else value
        if device is not None:
            tensor = tensor.to(device)
        copied[name] = tensor.clone()
    return copied


@dataclass(frozen=True)
class PaperMACStreamState:
    """The non-mutating state associated with exactly one logical stream.

    ``fast_weights`` holds the neural-memory parameters ``M_t`` and
    ``surprise`` holds its momentum/surprise pytree ``S_t``.  The tensors may
    have an autograd history: callers must replace a state with the returned
    value from an update rather than modifying it in place.
    """

    stream_id: str
    fast_weights: Mapping[str, Tensor]
    surprise: Mapping[str, Tensor]
    segment_index: int = 0
    reset_count: int = 0
    ended: bool = False
    query_history: Optional[Tensor] = None
    write_history: Optional[Tensor] = None

    FORMAT_VERSION = 2

    def __post_init__(self) -> None:
        if not self.stream_id:
            raise ValueError("stream_id must be a non-empty string")
        if self.segment_index < 0 or self.reset_count < 0:
            raise ValueError("segment_index and reset_count must be non-negative")
        if tuple(self.fast_weights) != tuple(self.surprise):
            raise ValueError("fast_weights and surprise must have identical ordered keys")
        for name, parameter in self.fast_weights.items():
            momentum = self.surprise[name]
            if parameter.shape != momentum.shape:
                raise ValueError(f"surprise[{name!r}] must match fast_weights[{name!r}]")
        if (self.query_history is None) != (self.write_history is None):
            raise ValueError("query_history and write_history must either both be present or both absent")
        if self.query_history is not None:
            if self.query_history.ndim != 2 or self.write_history is None:
                raise ValueError("projection histories must have shape (kernel_size - 1, d_model)")
            if self.query_history.shape != self.write_history.shape:
                raise ValueError("query_history and write_history must have identical shapes")

    @classmethod
    def initial(
        cls,
        stream_id: str,
        fast_weights: Mapping[str, Tensor],
        *,
        projection_history_length: int = 0,
        d_model: int | None = None,
    ) -> "PaperMACStreamState":
        """Create initial state without copying or mutating the meta-parameters."""

        ordered = OrderedDict(fast_weights.items())
        surprise = OrderedDict((name, torch.zeros_like(value)) for name, value in ordered.items())
        if projection_history_length < 0:
            raise ValueError("projection_history_length must be non-negative")
        if projection_history_length and (d_model is None or d_model <= 0):
            raise ValueError("d_model is required when projection histories are enabled")
        history = None
        if projection_history_length:
            reference = next(iter(ordered.values()))
            history = reference.new_zeros((projection_history_length, int(d_model)))
        return cls(
            stream_id=stream_id,
            fast_weights=ordered,
            surprise=surprise,
            query_history=history,
            write_history=None if history is None else history.clone(),
        )

    def replace(
        self,
        *,
        fast_weights: Mapping[str, Tensor],
        surprise: Mapping[str, Tensor],
        segment_index: Optional[int] = None,
        ended: Optional[bool] = None,
        query_history: Optional[Tensor] = None,
        write_history: Optional[Tensor] = None,
    ) -> "PaperMACStreamState":
        """Return a replacement state while retaining stream identity/metadata."""

        return PaperMACStreamState(
            stream_id=self.stream_id,
            fast_weights=OrderedDict(fast_weights.items()),
            surprise=OrderedDict(surprise.items()),
            segment_index=self.segment_index if segment_index is None else segment_index,
            reset_count=self.reset_count,
            ended=self.ended if ended is None else ended,
            query_history=self.query_history if query_history is None else query_history,
            write_history=self.write_history if write_history is None else write_history,
        )

    def reset(self, initial_fast_weights: Mapping[str, Tensor]) -> "PaperMACStreamState":
        """Return fresh state for this stream without affecting any other stream."""

        history_length = 0 if self.query_history is None else self.query_history.shape[0]
        d_model = None if self.query_history is None else self.query_history.shape[1]
        initial = PaperMACStreamState.initial(
            self.stream_id,
            initial_fast_weights,
            projection_history_length=history_length,
            d_model=d_model,
        )
        return PaperMACStreamState(
            stream_id=initial.stream_id,
            fast_weights=initial.fast_weights,
            surprise=initial.surprise,
            segment_index=0,
            reset_count=self.reset_count + 1,
            ended=False,
            query_history=initial.query_history,
            write_history=initial.write_history,
        )

    def mark_ended(self) -> "PaperMACStreamState":
        """Record an explicit end-of-stream event without changing fast weights."""

        return self.replace(fast_weights=self.fast_weights, surprise=self.surprise, ended=True)

    def to_state_dict(self) -> dict[str, object]:
        """Create a CPU, tensor-exact payload suitable for ``torch.save``.

        Serialization intentionally detaches the tensors because autograd graphs
        are process-local.  Dtype, values, key order, and stream metadata are
        retained; :meth:`from_state_dict` restores leaf tensors requiring grad.
        """

        return {
            "format_version": self.FORMAT_VERSION,
            "stream_id": self.stream_id,
            "fast_weights": _copy_weights(self.fast_weights, device=torch.device("cpu"), detach=True),
            "surprise": _copy_weights(self.surprise, device=torch.device("cpu"), detach=True),
            "fast_weight_requires_grad": {
                name: value.requires_grad for name, value in self.fast_weights.items()
            },
            "surprise_requires_grad": {name: value.requires_grad for name, value in self.surprise.items()},
            "segment_index": self.segment_index,
            "reset_count": self.reset_count,
            "ended": self.ended,
            "query_history": (
                None
                if self.query_history is None
                else self.query_history.detach().to(device="cpu").clone()
            ),
            "write_history": (
                None
                if self.write_history is None
                else self.write_history.detach().to(device="cpu").clone()
            ),
        }

    @classmethod
    def from_state_dict(
        cls, payload: Mapping[str, object], *, device: Optional[torch.device] = None
    ) -> "PaperMACStreamState":
        """Restore a serializable state payload as differentiable leaf tensors."""

        format_version = payload.get("format_version")
        if format_version not in (1, cls.FORMAT_VERSION):
            raise ValueError("unsupported Titans paper-MAC stream state version")
        raw_weights = payload.get("fast_weights")
        raw_surprise = payload.get("surprise")
        if not isinstance(raw_weights, Mapping) or not isinstance(raw_surprise, Mapping):
            raise ValueError("state payload is missing fast_weights or surprise")
        weight_grad = payload.get("fast_weight_requires_grad", {})
        surprise_grad = payload.get("surprise_requires_grad", {})
        if not isinstance(weight_grad, Mapping) or not isinstance(surprise_grad, Mapping):
            raise ValueError("state payload has invalid requires_grad metadata")

        def restore(raw: Mapping[str, object], requires_grad: Mapping[str, object]) -> FastWeights:
            restored: FastWeights = OrderedDict()
            for name, value in raw.items():
                if not isinstance(value, Tensor):
                    raise ValueError(f"state payload tensor {name!r} is invalid")
                tensor = value.detach().clone()
                if device is not None:
                    tensor = tensor.to(device)
                tensor.requires_grad_(bool(requires_grad.get(name, True)))
                restored[name] = tensor
            return restored

        stream_id = payload.get("stream_id")
        if not isinstance(stream_id, str):
            raise ValueError("state payload has invalid stream_id")
        def restore_history(name: str) -> Optional[Tensor]:
            if format_version == 1:
                return None
            value = payload.get(name)
            if value is None:
                return None
            if not isinstance(value, Tensor):
                raise ValueError(f"state payload {name} is invalid")
            restored = value.detach().clone()
            return restored.to(device) if device is not None else restored

        return cls(
            stream_id=stream_id,
            fast_weights=restore(raw_weights, weight_grad),
            surprise=restore(raw_surprise, surprise_grad),
            segment_index=int(payload.get("segment_index", 0)),
            reset_count=int(payload.get("reset_count", 0)),
            ended=bool(payload.get("ended", False)),
            query_history=restore_history("query_history"),
            write_history=restore_history("write_history"),
        )
