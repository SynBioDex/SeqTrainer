#!/usr/bin/env python3
"""Create the directory skeleton for an offline Alpine benchmark bundle."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--repo-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.bundle_root.resolve()
    for relative in (
        "image",
        "repository",
        "models/DNABERT-2-117M",
        "models/DNABERT-6",
        "models/ipromp_ecoli",
        "data/promoter_classification",
        "manifests",
        "runs",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)

    if args.repo_dir:
        source = args.repo_dir.resolve()
        target = root / "repository" / "SeqTrainer"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(
            source,
            target,
            ignore=shutil.ignore_patterns(".git", ".pytest_cache", ".ruff_cache", "__pycache__"),
        )
        print(f"Copied repository: {source} -> {target}")

    sums = root / "manifests" / "SHA256SUMS"
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == sums:
            continue
        rows.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    sums.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    print(f"Bundle skeleton ready: {root}")
    print(f"Checksum manifest: {sums}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
