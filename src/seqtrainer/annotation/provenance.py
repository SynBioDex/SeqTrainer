"""Stable provenance objects for SeqTrainer annotation exports."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_identity(namespace: str, *parts: object) -> str:
    """Build deterministic, URL-safe identities without random UUIDs."""
    clean = [str(part).strip("/").replace(" ", "_") for part in parts]
    return "/".join([namespace.rstrip("/")] + clean)


def model_provenance(
    *,
    checkpoint: str | Path | None,
    benchmark_manifest: str | Path | None,
    model_family: str,
    threshold: float,
    threshold_source: str,
) -> dict[str, Any]:
    return {
        "model_family": model_family,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": file_sha256(checkpoint) if checkpoint and Path(checkpoint).exists() else None,
        "benchmark_manifest": str(benchmark_manifest) if benchmark_manifest else None,
        "benchmark_manifest_sha256": file_sha256(benchmark_manifest) if benchmark_manifest and Path(benchmark_manifest).exists() else None,
        "threshold": float(threshold),
        "threshold_source": threshold_source,
    }
