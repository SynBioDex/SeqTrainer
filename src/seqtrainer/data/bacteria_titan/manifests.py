"""Manifest and download helpers for reproducible bacterial datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Optional
import zipfile

import pandas as pd
import requests


GTDB_R220_METADATA_URL = (
    "https://data.gtdb.ecogenomic.org/releases/release220/220.0/"
    "bac120_metadata_r220.tsv.gz"
)
GTDB_R220_TAXONOMY_URL = (
    "https://data.gtdb.ecogenomic.org/releases/release220/220.0/"
    "bac120_taxonomy_r220.tsv.gz"
)
REQUIRED_MANIFEST_COLUMNS = ("accession", "scope", "split", "genome_size")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_cached(url: str, path: str | Path, timeout: int = 120) -> Path:
    """Download to a partial file and atomically publish completed downloads."""

    destination = Path(path)
    if destination.exists() and destination.stat().st_size:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    partial.replace(destination)
    return destination


def download_gtdb_r220(output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    return {
        "metadata": download_cached(GTDB_R220_METADATA_URL, output / "bac120_metadata_r220.tsv.gz"),
        "taxonomy": download_cached(GTDB_R220_TAXONOMY_URL, output / "bac120_taxonomy_r220.tsv.gz"),
    }


def download_ncbi_batches(
    accessions: Iterable[str],
    output_dir: str | Path,
    batch_size: int = 500,
    datasets_command: str = "datasets",
    allow_download: bool = True,
) -> pd.DataFrame:
    """Cache NCBI Datasets genome ZIP batches and return their manifest.

    Existing non-empty ZIP files are checksum-verified and reused, which makes
    interrupted Colab builds resumable without redownloading completed batches.
    """

    unique = sorted(set(str(accession) for accession in accessions))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for batch_index, start in enumerate(range(0, len(unique), batch_size)):
        batch = unique[start : start + batch_size]
        path = output / f"ncbi_batch_{batch_index:05d}.zip"
        if not _verified_zip(path, write_checksum=True):
            if not allow_download:
                raise FileNotFoundError(f"missing or invalid cached NCBI batch: {path}")
            partial = path.with_suffix(".zip.partial")
            command = [
                datasets_command,
                "download",
                "genome",
                "accession",
                *batch,
                "--include",
                "genome",
                "--filename",
                str(partial),
                "--no-progressbar",
            ]
            subprocess.run(command, check=True)
            partial.replace(path)
            if not _verified_zip(path, write_checksum=True):
                raise ValueError(f"NCBI Datasets produced an invalid ZIP: {path}")
        rows.append(
            {
                "batch_index": batch_index,
                "path": str(path),
                "accession_count": len(batch),
                "first_accession": batch[0],
                "last_accession": batch[-1],
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def _verified_zip(path: Path, write_checksum: bool = False) -> bool:
    if not path.exists() or not path.stat().st_size or not zipfile.is_zipfile(path):
        return False
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    digest = sha256_file(path)
    if checksum_path.exists():
        return checksum_path.read_text(encoding="ascii").strip() == digest
    try:
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                return False
    except zipfile.BadZipFile:
        return False
    if write_checksum:
        checksum_path.write_text(digest + "\n", encoding="ascii")
    return True


def normalize_accession(accession: str) -> str:
    """Remove GTDB's RS_/GB_ prefix while retaining the assembly version."""

    value = str(accession).strip()
    return value[3:] if value.startswith(("RS_", "GB_")) else value


def read_gtdb_metadata(metadata_path: str | Path, taxonomy_path: Optional[str | Path] = None) -> pd.DataFrame:
    frame = pd.read_csv(metadata_path, sep="\t", compression="infer", low_memory=False)
    accession_column = "accession" if "accession" in frame.columns else frame.columns[0]
    frame = frame.rename(columns={accession_column: "gtdb_accession"})
    frame["accession"] = frame["gtdb_accession"].map(normalize_accession)
    if taxonomy_path is not None and "gtdb_taxonomy" not in frame.columns:
        taxonomy = pd.read_csv(
            taxonomy_path,
            sep="\t",
            names=["gtdb_accession", "gtdb_taxonomy"],
            compression="infer",
        )
        frame = frame.merge(taxonomy, on="gtdb_accession", how="left", validate="one_to_one")
    return frame


def write_table(frame: pd.DataFrame, base_path: str | Path) -> tuple[Path, Path]:
    """Write matching Parquet and CSV manifests.

    Parquet support is supplied by pandas' optional ``pyarrow`` or ``fastparquet``
    engine. A clear error is raised instead of writing a mislabeled fallback.
    """

    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    parquet = base.with_suffix(".parquet")
    csv = base.with_suffix(".csv")
    frame.to_csv(csv, index=False)
    try:
        frame.to_parquet(parquet, index=False)
    except ImportError as exc:
        raise RuntimeError("Parquet manifests require pyarrow or fastparquet") from exc
    return parquet, csv


def read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    return pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)


def validate_accession_manifest(frame: pd.DataFrame, require_splits: bool = True) -> None:
    required = set(REQUIRED_MANIFEST_COLUMNS if require_splits else ("accession", "scope", "genome_size"))
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"accession manifest is missing columns: {sorted(missing)}")
    if frame["accession"].isna().any() or frame["accession"].duplicated().any():
        raise ValueError("accessions must be non-null and unique")
    if require_splits:
        unexpected = set(frame["split"]) - {"train", "val", "test"}
        if unexpected:
            raise ValueError(f"unexpected split values: {sorted(unexpected)}")


def write_dataset_metadata(
    dataset_root: str | Path,
    manifest: dict[str, Any],
    readme: str,
    checksum_paths: Optional[Iterable[str | Path]] = None,
) -> None:
    root = Path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "README.md").write_text(readme.rstrip() + "\n", encoding="utf-8")
    paths = sorted(Path(path) for path in (checksum_paths or root.rglob("*")) if Path(path).is_file())
    excluded = {root / "checksums.sha256"}
    lines = [f"{sha256_file(path)}  {path.relative_to(root)}" for path in paths if path not in excluded]
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
