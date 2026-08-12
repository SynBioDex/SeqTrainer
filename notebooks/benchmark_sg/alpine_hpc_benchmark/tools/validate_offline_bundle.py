#!/usr/bin/env python3
"""Validate required offline data and model assets before an Alpine job."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SPLITS = {
    "train": "train_EP_DNA_BERT2_genomic_order.csv",
    "validation": "eval_EP_DNA_BERT2_genomic_order.csv",
    "test": "test_EP_DNA_BERT2_genomic_order.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    root = parse_args().bundle_root.resolve()
    errors: list[str] = []
    required = [
        root / "image" / "seqtrainer-alpine-gpu.sif",
        root / "repository" / "SeqTrainer" / "src" / "seqtrainer" / "cli" / "main.py",
        root / "models" / "DNABERT-2-117M" / "config.json",
        root / "models" / "DNABERT-6" / "config.json",
        root / "models" / "DNABERT-6" / "pytorch_model.bin",
        root / "models" / "DNABERT-6" / "vocab.txt",
    ]
    required.extend(root / "models" / "ipromp_ecoli" / f"10_fold_{fold}.pth" for fold in range(1, 6))
    for path in required:
        if not path.is_file():
            errors.append(f"missing file: {path}")

    data_dir = root / "data" / "promoter_classification"
    for split, name in SPLITS.items():
        path = data_dir / name
        if not path.is_file():
            errors.append(f"missing {split} CSV: {path}")
            continue
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = set(reader.fieldnames or [])
                rows = list(reader)
            missing = {"sequence", "label"}.difference(fieldnames)
            if missing:
                errors.append(f"{split} CSV missing columns: {sorted(missing)}")
            else:
                labels = {int(row["label"]) for row in rows if row.get("label", "") != ""}
                if not labels.issubset({0, 1}):
                    errors.append(f"{split} labels are not binary 0/1: {sorted(labels)}")
                print(f"{split}: {len(rows)} rows; labels={sorted(labels)}")
        except Exception as exc:
            errors.append(f"could not read {path}: {exc}")

    if errors:
        print("Offline bundle validation failed:")
        for error in errors:
            print(f"- {error}")
        return 2
    print(f"Offline bundle is ready: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
