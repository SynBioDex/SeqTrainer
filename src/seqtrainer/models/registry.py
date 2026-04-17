"""Backbone/head registry stubs for pluggable model composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class BackboneSpec:
    name: str
    framework: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HeadSpec:
    name: str
    task_type: str
    config: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """Simple registry for framework-neutral backbone/head metadata."""

    def __init__(self) -> None:
        self._backbones: dict[str, BackboneSpec] = {}
        self._heads: dict[str, HeadSpec] = {}

    def register_backbone(self, spec: BackboneSpec) -> None:
        self._backbones[spec.name] = spec

    def register_head(self, spec: HeadSpec) -> None:
        self._heads[spec.name] = spec

    def get_backbone(self, name: str) -> BackboneSpec:
        return self._backbones[name]

    def get_head(self, name: str) -> HeadSpec:
        return self._heads[name]
