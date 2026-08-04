"""Deterministic FASTA and uint8 next-token shard materialization."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import shutil
import zipfile
from bisect import bisect_right
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

DNA_TOKEN_TO_ID = {"N": 1, "A": 2, "C": 3, "G": 4, "T": 5}


def iter_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="ascii", errors="replace") as handle:
        name: Optional[str] = None
        sequence: list[str] = []
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(sequence)
                name, sequence = line[1:], []
            elif line and name is not None:
                sequence.append(line)
        if name is not None:
            yield name, "".join(sequence)


class _DeterministicGzipWriter:
    def __init__(self, path: Path) -> None:
        self.raw = path.open("wb")
        self.compressed = gzip.GzipFile(filename="", mode="wb", fileobj=self.raw, mtime=0)
        self.text = io.TextIOWrapper(self.compressed, encoding="ascii", newline="\n")

    def write(self, value: str) -> int:
        return self.text.write(value)

    def close(self) -> None:
        self.text.close()
        self.raw.close()


def _open_deterministic_gzip(path: Path) -> _DeterministicGzipWriter:
    return _DeterministicGzipWriter(path)


def regenerate_fasta_shards(
    zip_dir: str | Path,
    accession_manifest: pd.DataFrame,
    output_root: str | Path,
    dataset_class: str,
    target_shard_bp: int = 500_000_000,
) -> pd.DataFrame:
    """Rebuild deterministic split FASTA shards from cached NCBI Datasets ZIPs."""

    output = Path(output_root) / "shards" / dataset_class
    if output.exists():
        shutil.rmtree(output)
    rows: list[dict[str, object]] = []
    zip_paths = sorted(Path(zip_dir).glob("*.zip"))
    for split in ("train", "val", "test"):
        split_accessions = set(accession_manifest.loc[accession_manifest["split"].eq(split), "accession"].astype(str))
        shard_index, shard_bp, records = 0, 0, 0
        destination = output / split / f"shard_{shard_index:05d}.fa.gz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle = _open_deterministic_gzip(destination)
        try:
            for zip_path in zip_paths:
                with zipfile.ZipFile(zip_path) as archive:
                    members = sorted(name for name in archive.namelist() if name.endswith((".fna", ".fa", ".fasta")))
                    for member in members:
                        parts = member.split("/")
                        accession = next((part for part in parts if part.startswith(("GCF_", "GCA_"))), "")
                        if accession not in split_accessions:
                            continue
                        text = archive.read(member).decode("ascii", errors="replace")
                        for line in text.splitlines():
                            if line.startswith(">"):
                                handle.write(f">accession={accession}|{line[1:]}\n")
                                records += 1
                            else:
                                sequence = line.strip().upper()
                                handle.write(sequence + "\n")
                                shard_bp += len(sequence)
                        if shard_bp >= target_shard_bp:
                            handle.close()
                            rows.append(_fasta_row(destination, dataset_class, split, shard_index, shard_bp, records))
                            shard_index, shard_bp, records = shard_index + 1, 0, 0
                            destination = output / split / f"shard_{shard_index:05d}.fa.gz"
                            handle = _open_deterministic_gzip(destination)
        finally:
            handle.close()
        if shard_bp or records:
            rows.append(_fasta_row(destination, dataset_class, split, shard_index, shard_bp, records))
        elif destination.exists():
            destination.unlink()
    return pd.DataFrame(rows)


def _fasta_row(path: Path, dataset_class: str, split: str, index: int, bases: int, records: int) -> dict[str, object]:
    return {
        "dataset_class": dataset_class,
        "split": split,
        "shard_index": index,
        "path": str(path),
        "bases": bases,
        "records": records,
        "sha256": _sha256(path),
    }


def tokenize_fasta_shards(
    fasta_manifest: pd.DataFrame,
    output_root: str | Path,
    dataset_class: str,
    context_length: int = 2048,
    stride: Optional[int] = None,
    windows_per_shard: int = 100_000,
) -> pd.DataFrame:
    """Rebuild deterministic ``(windows, context_length + 1)`` uint8 shards."""

    if context_length <= 0:
        raise ValueError("context_length must be positive")
    stride = stride or context_length
    token_root = Path(output_root) / "tokenized" / dataset_class / f"ctx{context_length}"
    if token_root.exists():
        shutil.rmtree(token_root)
    rows: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        buffer: list[np.ndarray] = []
        shard_index = 0
        sources = fasta_manifest.loc[fasta_manifest["split"].eq(split), "path"].astype(str)
        for source in sources:
            for header, sequence in iter_fasta(source):
                encoded = np.fromiter(
                    (DNA_TOKEN_TO_ID.get(base, 1) for base in sequence.upper()), dtype=np.uint8, count=len(sequence)
                )
                for start in range(0, max(0, len(encoded) - context_length), stride):
                    window = encoded[start : start + context_length + 1]
                    if len(window) == context_length + 1:
                        buffer.append(window)
                    if len(buffer) >= windows_per_shard:
                        rows.append(_write_token_shard(token_root, split, shard_index, buffer, context_length, header))
                        shard_index, buffer = shard_index + 1, []
        if buffer:
            rows.append(_write_token_shard(token_root, split, shard_index, buffer, context_length, ""))
    (token_root / "tokenizer.json").write_text(
        json.dumps(
            {
                "type": "DNABaseTokenizer",
                "version": 1,
                "token_to_id": {"PAD": 0, "N": 1, "A": 2, "C": 3, "G": 4, "T": 5},
                "max_length": context_length,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return pd.DataFrame(rows)


def _write_token_shard(
    root: Path,
    split: str,
    index: int,
    windows: list[np.ndarray],
    context_length: int,
    final_header: str,
) -> dict[str, object]:
    path = root / split / f"tokens_{index:05d}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.stack(windows).astype(np.uint8, copy=False)
    np.save(path, values, allow_pickle=False)
    return {
        "split": split,
        "shard_index": index,
        "path": str(path),
        "num_windows": len(values),
        "context_length": context_length,
        "dtype": "uint8",
        "shape": list(values.shape),
        "final_source_header": final_header,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class TokenShardDataset:
    """Memory-mapped index over deterministic NumPy token shards."""

    def __init__(self, paths: list[str | Path] | str | Path) -> None:
        if isinstance(paths, (str, Path)):
            source = Path(paths)
            paths = sorted(source.glob("tokens_*.npy")) if source.is_dir() else [source]
        self.paths = [Path(path) for path in paths]
        if not self.paths:
            raise ValueError("no token shards found")
        self.arrays = [np.load(path, mmap_mode="r", allow_pickle=False) for path in self.paths]
        widths = {array.shape[1] for array in self.arrays if array.ndim == 2}
        if len(widths) != 1 or any(array.dtype != np.uint8 for array in self.arrays):
            raise ValueError("token shards must be uint8 matrices with a consistent width")
        self.cumulative: list[int] = []
        total = 0
        for array in self.arrays:
            total += len(array)
            self.cumulative.append(total)

    def __len__(self) -> int:
        return self.cumulative[-1]

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = bisect_right(self.cumulative, index)
        previous = self.cumulative[shard - 1] if shard else 0
        row = np.asarray(self.arrays[shard][index - previous], dtype=np.int64)
        return row[:-1], row[1:]
