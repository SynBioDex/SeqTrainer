"""Prepare the ai x bio promoter dataset for shared benchmarks."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SEQUENCE_ALIASES = ("sequence", "seq", "dna", "dna_sequence", "nucleotide_sequence")
LABEL_ALIASES = ("label", "labels", "target", "class", "y", "promoter", "is_promoter")
SPLIT_ALIASES = ("split", "set", "subset", "partition", "split_name")
ID_ALIASES = ("id", "seq_id", "sequence_id", "name", "record_id", "accession")
POSITIVE_LABELS = {"1", "true", "yes", "y", "pos", "positive", "promoter"}
NEGATIVE_LABELS = {
    "0",
    "false",
    "no",
    "n",
    "neg",
    "negative",
    "nonpromoter",
    "non_promoter",
    "non-promoter",
    "background",
}
SPLIT_NORMALIZATION = {
    "train": "train",
    "training": "train",
    "tr": "train",
    "validation": "validation",
    "validate": "validation",
    "val": "validation",
    "eval": "validation",
    "dev": "validation",
    "test": "test",
    "testing": "test",
    "te": "test",
}


@dataclass(frozen=True)
class PreparedAiXBioDataset:
    """Paths and metadata written by the ai x bio preparation flow."""

    output_dir: Path
    split_paths: dict[str, Path]
    metadata_path: Path
    metadata: dict[str, Any]


def prepare_ai_x_bio_splits(
    *,
    drive_root: str | Path = "/content/drive/MyDrive",
    output_dir: str | Path = "data/benchmarks/ai_x_bio",
    source_file: str | Path | None = None,
    seed: int = 42,
) -> PreparedAiXBioDataset:
    """Find/convert an ai x bio file into sequence,label,id split CSVs."""
    source = Path(source_file) if source_file is not None else find_ai_x_bio_file(drive_root)
    frame, detected_format = read_ai_x_bio_source(source)
    standardized, column_metadata = standardize_ai_x_bio_frame(frame)
    split_frames, split_strategy = split_ai_x_bio_frame(standardized, seed=seed)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    split_paths: dict[str, Path] = {}
    split_counts: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        split_frame = split_frames[split][["sequence", "label", "id"]].reset_index(drop=True)
        path = output / f"{split}.csv"
        split_frame.to_csv(path, index=False)
        split_paths[split] = path
        split_counts[split] = {
            "rows": int(len(split_frame)),
            "class_counts": {
                str(key): int(value)
                for key, value in split_frame["label"].value_counts().sort_index().to_dict().items()
            },
        }

    metadata = {
        "source_file": str(source),
        "source_format": detected_format,
        "output_schema": ["sequence", "label", "id"],
        "output_dir": str(output),
        "seed": seed,
        "split_strategy": split_strategy,
        "split_files": {split: str(path) for split, path in split_paths.items()},
        "split_counts": split_counts,
        "columns": column_metadata,
    }
    metadata_path = output / "dataset_prep_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return PreparedAiXBioDataset(output, split_paths, metadata_path, metadata)


def find_ai_x_bio_file(drive_root: str | Path) -> Path:
    """Search a mounted Drive root for a file whose name matches ai x bio."""
    root = Path(drive_root)
    if not root.exists():
        raise FileNotFoundError(f"Drive root does not exist: {root}")

    candidates = [
        path
        for path in root.rglob("*")
        if path.is_file() and _looks_like_ai_x_bio_name(path.name) and _supported_suffix(path)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find an ai x bio dataset file below {root}. "
            "Pass --source-file to prepare a specific file."
        )
    candidates.sort(key=lambda path: (_format_priority(path), len(str(path))))
    return candidates[0]


def read_ai_x_bio_source(path: str | Path) -> tuple[pd.DataFrame, str]:
    """Read CSV, TSV, FASTA, or JSONL ai x bio data."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source), "csv"
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(source, sep="\t"), "tsv"
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(source, lines=True), "jsonl"
    if suffix in {".fa", ".fasta", ".fna"}:
        return parse_fasta_with_metadata(source), "fasta"
    raise ValueError(f"Unsupported ai x bio source format: {source}")


def parse_fasta_with_metadata(path: str | Path) -> pd.DataFrame:
    """Parse FASTA records with label/split/id metadata in headers."""
    records: list[dict[str, Any]] = []
    header: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        if header is None:
            return
        metadata = _parse_fasta_header(header)
        records.append({**metadata, "sequence": "".join(chunks)})

    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)
        flush()

    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return pd.DataFrame(records)


def standardize_ai_x_bio_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Infer columns, normalize DNA/labels, and return sequence,label,id rows."""
    if frame.empty:
        raise ValueError("ai x bio source file is empty")

    sequence_col = _find_column(frame.columns, SEQUENCE_ALIASES)
    label_col = _find_column(frame.columns, LABEL_ALIASES)
    split_col = _find_column(frame.columns, SPLIT_ALIASES, required=False)
    id_col = _find_column(frame.columns, ID_ALIASES, required=False)
    if sequence_col is None:
        raise ValueError(f"Could not infer sequence column. Tried aliases: {SEQUENCE_ALIASES}")
    if label_col is None:
        raise ValueError(f"Could not infer label column. Tried aliases: {LABEL_ALIASES}")

    out = pd.DataFrame()
    out["sequence"] = frame[sequence_col].map(normalize_dna_sequence)
    out["label"] = frame[label_col].map(normalize_binary_label)
    if out["label"].isna().any():
        bad = frame.loc[out["label"].isna(), label_col].head(5).tolist()
        raise ValueError(f"Could not normalize labels to binary 0/1. Example values: {bad}")
    out["label"] = out["label"].astype(int)
    if id_col is not None:
        out["id"] = frame[id_col].astype(str)
    else:
        out["id"] = [f"ai_x_bio_{idx:07d}" for idx in range(len(out))]
    if split_col is not None:
        normalized_split = frame[split_col].map(normalize_split_name)
        if normalized_split.isna().any():
            bad = frame.loc[normalized_split.isna(), split_col].head(5).tolist()
            raise ValueError(f"Could not normalize split names. Example values: {bad}")
        out["split"] = normalized_split

    out = out[out["sequence"].str.len() > 0].reset_index(drop=True)
    return out, {
        "sequence_column": sequence_col,
        "label_column": label_col,
        "split_column": split_col,
        "id_column": id_col,
    }


def split_ai_x_bio_frame(frame: pd.DataFrame, *, seed: int = 42) -> tuple[dict[str, pd.DataFrame], str]:
    """Preserve source splits when present; otherwise create a seeded stratified split."""
    if "split" in frame.columns:
        split_frames = {
            split: frame[frame["split"] == split][["sequence", "label", "id"]].reset_index(drop=True)
            for split in ("train", "validation", "test")
        }
        missing = [split for split, split_frame in split_frames.items() if split_frame.empty]
        if missing:
            raise ValueError(f"Source split column is present but missing split(s): {missing}")
        return split_frames, "preserved_source_split"

    train, temp = _stratified_train_test_split(frame, test_size=0.30, seed=seed)
    validation, test = _stratified_train_test_split(temp, test_size=0.50, seed=seed)
    return {
        "train": train[["sequence", "label", "id"]].reset_index(drop=True),
        "validation": validation[["sequence", "label", "id"]].reset_index(drop=True),
        "test": test[["sequence", "label", "id"]].reset_index(drop=True),
    }, "seeded_stratified_70_15_15"


def normalize_dna_sequence(value: Any) -> str:
    """Normalize DNA to uppercase A/C/G/T/N, replacing U with T."""
    text = str(value).strip().upper().replace("U", "T")
    return re.sub("[^ACGTN]", "N", text)


def normalize_binary_label(value: Any) -> int | None:
    """Normalize common binary label encodings to 0/1."""
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and float(value) in {0.0, 1.0}:
        return int(value)
    text = str(value).strip().lower()
    if text in POSITIVE_LABELS:
        return 1
    if text in NEGATIVE_LABELS:
        return 0
    return None


def normalize_split_name(value: Any) -> str | None:
    """Normalize train/validation/test split labels."""
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    return SPLIT_NORMALIZATION.get(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare ai x bio data for SeqTrainer benchmarks.")
    parser.add_argument("--drive-root", default="/content/drive/MyDrive")
    parser.add_argument("--source-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/benchmarks/ai_x_bio"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    result = prepare_ai_x_bio_splits(
        drive_root=args.drive_root,
        source_file=args.source_file,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(f"output_dir={result.output_dir}")
    print(f"metadata={result.metadata_path}")
    for split, path in result.split_paths.items():
        print(f"{split}={path}")
    return 0


def _stratified_train_test_split(frame: pd.DataFrame, *, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    labels = frame["label"]
    stratify = labels if labels.value_counts().min() >= 2 else None
    try:
        left, right = train_test_split(
            frame,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        left, right = train_test_split(frame, test_size=test_size, random_state=seed, stratify=None)
    return left.reset_index(drop=True), right.reset_index(drop=True)


def _find_column(columns: Iterable[str], aliases: tuple[str, ...], *, required: bool = True) -> str | None:
    normalized = {_normalize_name(column): column for column in columns}
    for alias in aliases:
        column = normalized.get(_normalize_name(alias))
        if column is not None:
            return column
    if required:
        return None
    return None


def _parse_fasta_header(header: str) -> dict[str, str]:
    parts = re.split(r"[\s|]+", header.strip())
    metadata: dict[str, str] = {"id": parts[0] if parts and parts[0] else header.strip()}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            metadata[_normalize_name(key)] = value
        elif ":" in part:
            key, value = part.split(":", 1)
            metadata[_normalize_name(key)] = value
        elif _normalize_name(part) in POSITIVE_LABELS | NEGATIVE_LABELS:
            metadata["label"] = part
        elif normalize_split_name(part) is not None:
            metadata["split"] = part
    return metadata


def _looks_like_ai_x_bio_name(name: str) -> bool:
    normalized = _normalize_name(name)
    return all(token in normalized for token in ("ai", "x", "bio"))


def _supported_suffix(path: Path) -> bool:
    return path.suffix.lower() in {".csv", ".tsv", ".tab", ".fa", ".fasta", ".fna", ".jsonl", ".ndjson"}


def _format_priority(path: Path) -> int:
    suffix = path.suffix.lower()
    order = {".csv": 0, ".tsv": 1, ".tab": 1, ".jsonl": 2, ".ndjson": 2, ".fasta": 3, ".fa": 3, ".fna": 3}
    return order.get(suffix, 99)


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
