"""GenBank read/write helpers for promoter annotation."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _require_biopython() -> Any:
    try:
        from Bio import SeqIO
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional extra
        raise ModuleNotFoundError(
            "GenBank annotation requires Biopython. Install with `pip install -e \".[annotation]\"`."
        ) from exc
    return SeqIO


def read_genbank(path: str | Path) -> Any:
    """Read one GenBank record and return a Biopython SeqRecord."""
    SeqIO = _require_biopython()
    records = list(SeqIO.parse(str(path), "genbank"))
    if not records:
        raise ValueError(f"No GenBank records found in {path}")
    if len(records) > 1:
        raise ValueError(f"Expected one GenBank record in {path}, found {len(records)}")
    return records[0]


def write_genbank(record: Any, path: str | Path) -> Path:
    """Write one GenBank record, preserving existing features and annotations."""
    SeqIO = _require_biopython()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record.annotations.setdefault("molecule_type", "DNA")
    SeqIO.write(record, str(out_path), "genbank")
    return out_path


def is_circular(record: Any) -> bool:
    """Return whether a GenBank record is annotated as circular."""
    topology = str(record.annotations.get("topology", "")).lower()
    return topology == "circular"


def record_topology(record: Any) -> str:
    """Return a normalized topology string."""
    return "circular" if is_circular(record) else "linear"

