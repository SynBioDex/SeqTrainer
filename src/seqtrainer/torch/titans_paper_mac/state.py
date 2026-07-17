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

    FORMAT_VERSION = 1

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

    @classmethod
    def initial(cls, stream_id: str, fast_weights: Mapping[str, Tensor]) -> "PaperMACStreamState":
        """Create initial state without copying or mutating the meta-parameters."""

        ordered = OrderedDict(fast_weights.items())
        surprise = OrderedDict((name, torch.zeros_like(value)) for name, value in ordered.items())
        return cls(stream_id=stream_id, fast_weights=ordered, surprise=surprise)

    def replace(
        self,
        *,
        fast_weights: Mapping[str, Tensor],
        surprise: Mapping[str, Tensor],
        segment_index: Optional[int] = None,
        ended: Optional[bool] = None,
    ) -> "PaperMACStreamState":
        """Return a replacement state while retaining stream identity/metadata."""

        return PaperMACStreamState(
            stream_id=self.stream_id,
            fast_weights=OrderedDict(fast_weights.items()),
            surprise=OrderedDict(surprise.items()),
            segment_index=self.segment_index if segment_index is None else segment_index,
            reset_count=self.reset_count,
            ended=self.ended if ended is None else ended,
        )

    def reset(self, initial_fast_weights: Mapping[str, Tensor]) -> "PaperMACStreamState":
        """Return fresh state for this stream without affecting any other stream."""

        initial = PaperMACStreamState.initial(self.stream_id, initial_fast_weights)
        return PaperMACStreamState(
            stream_id=initial.stream_id,
            fast_weights=initial.fast_weights,
            surprise=initial.surprise,
            segment_index=0,
            reset_count=self.reset_count + 1,
            ended=False,
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
        }

    @classmethod
    def from_state_dict(
        cls, payload: Mapping[str, object], *, device: Optional[torch.device] = None
    ) -> "PaperMACStreamState":
        """Restore a serializable state payload as differentiable leaf tensors."""

        if payload.get("format_version") != cls.FORMAT_VERSION:
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
        return cls(
            stream_id=stream_id,
            fast_weights=restore(raw_weights, weight_grad),
            surprise=restore(raw_surprise, surprise_grad),
            segment_index=int(payload.get("segment_index", 0)),
            reset_count=int(payload.get("reset_count", 0)),
            ended=bool(payload.get("ended", False)),
        )
