"""iPro-MP FASTA conversion helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from seqtrainer.benchmarks.config import BenchmarkConfig


def write_ipromp_fastas(
    config: BenchmarkConfig,
    frames: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write SeqTrainer split CSV rows as iPro-MP-compatible FASTA files."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for split, frame in frames.items():
        path = out_dir / f"{split}.fasta"
        path.write_text(_frame_to_fasta(config, frame, split), encoding="utf-8")
        written[split] = str(path)
    return written


def _frame_to_fasta(config: BenchmarkConfig, frame: pd.DataFrame, split: str) -> str:
    lines: list[str] = []
    id_field = config.dataset.id_field
    for idx, row in frame.reset_index(drop=True).iterrows():
        stable_id = str(row[id_field]) if id_field and id_field in frame.columns else f"{split}_{idx}"
        label = int(row[config.dataset.label_field])
        sequence = _clean_sequence(str(row[config.dataset.sequence_field]))
        lines.append(f">{stable_id}|split={split}|idx={idx}|label={label}")
        lines.append(sequence)
    return "\n".join(lines) + "\n"


def _clean_sequence(sequence: str) -> str:
    cleaned = "".join(base for base in sequence.upper() if base in {"A", "C", "G", "T", "N"})
    if not cleaned:
        raise ValueError("Cannot write an empty sequence to FASTA")
    return cleaned
