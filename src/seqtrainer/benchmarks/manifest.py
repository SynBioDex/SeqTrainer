"""Run manifest helpers for reproducible benchmark artifacts."""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig


def build_run_manifest(
    config: BenchmarkConfig,
    *,
    split_summary: dict[str, Any] | None = None,
    threshold: float | None = None,
    model_metadata: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a manifest with config, split, environment, and git metadata."""
    manifest: dict[str, Any] = {
        "experiment": asdict(config.experiment),
        "dataset": {
            **asdict(config.dataset),
            "split_summary": split_summary or {},
        },
        "label": asdict(config.label),
        "split": asdict(config.split),
        "preprocessing": asdict(config.preprocessing),
        "model": {
            **asdict(config.model),
            "metadata": model_metadata or {},
        },
        "training": asdict(config.training),
        "evaluation": {
            **asdict(config.evaluation),
            "selected_threshold": threshold,
        },
        "outputs": asdict(config.outputs),
        "environment": {
            **asdict(config.environment),
            "runtime": runtime_metadata(),
            "git": git_metadata(),
        },
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def runtime_metadata() -> dict[str, Any]:
    """Capture lightweight Python and package-version metadata."""
    packages = {}
    for name in ("seqtrainer", "numpy", "pandas", "scikit-learn", "torch", "transformers"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def git_metadata(repo_dir: str | Path | None = None) -> dict[str, Any]:
    """Capture git commit and branch when the code is run inside a git repo."""
    cwd = Path(repo_dir) if repo_dir is not None else Path.cwd()
    return {
        "commit": _git(["rev-parse", "HEAD"], cwd),
        "branch": _git(["branch", "--show-current"], cwd),
        "is_dirty": _git(["status", "--porcelain"], cwd) not in ("", None),
    }


def to_plain_data(value: Any) -> Any:
    """Convert dataclasses and common containers to JSON-serializable data."""
    if is_dataclass(value):
        return to_plain_data(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()

