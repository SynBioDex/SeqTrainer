"""iPro-MP external benchmark preparation and prediction normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from seqtrainer.benchmarks.config import BenchmarkConfig, load_benchmark_config
from seqtrainer.benchmarks.splits import load_predefined_split_frames

DNA_ALPHABET = frozenset({"A", "C", "G", "T", "N"})
SPLIT_ORDER = ("train", "validation", "test")


@dataclass(frozen=True)
class IprompPreparationResult:
    """Paths written for external iPro-MP inference."""

    output_dir: Path
    fasta_paths: dict[str, Path]
    mapping_csv: Path
    command_script: Path
    prediction_schema: Path


def prepare_ipromp_inputs(
    config_or_path: BenchmarkConfig | str | Path,
    *,
    base_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> IprompPreparationResult:
    """Write FASTA, mapping, command, and schema files for external iPro-MP."""
    config = load_benchmark_config(config_or_path) if not isinstance(config_or_path, BenchmarkConfig) else config_or_path
    frames = load_predefined_split_frames(config, base_dir=base_dir)
    out_dir = Path(output_dir or config.outputs.output_dir)
    params = dict(config.model.params)
    fasta_dir = _resolve_output_path(
        params.get("fasta_output_dir"),
        out_dir / "ipromp_fasta",
        run_dir=out_dir,
        configured_root=config.outputs.output_dir,
    )
    mapping_csv = _resolve_output_path(
        params.get("mapping_csv"),
        out_dir / "ipromp_id_mapping.csv",
        run_dir=out_dir,
        configured_root=config.outputs.output_dir,
    )

    mapping = build_ipromp_mapping(config, frames)
    fasta_paths = write_ipromp_fastas(config, frames, fasta_dir, mapping=mapping)
    mapping_csv.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(mapping_csv, index=False)
    command_script = write_ipromp_run_commands(config, fasta_paths, out_dir)
    prediction_schema = write_external_prediction_schema(out_dir)
    return IprompPreparationResult(
        output_dir=out_dir,
        fasta_paths=fasta_paths,
        mapping_csv=mapping_csv,
        command_script=command_script,
        prediction_schema=prediction_schema,
    )


def build_ipromp_mapping(config: BenchmarkConfig, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create stable row IDs for every configured benchmark split."""
    rows: list[dict[str, Any]] = []
    for split in SPLIT_ORDER:
        frame = frames[split].reset_index(drop=True)
        for row_index, row in frame.iterrows():
            sequence_id = _sequence_id(config, row, split, int(row_index))
            sequence = normalize_dna_sequence(str(row[config.dataset.sequence_field]))
            label = int(row[config.dataset.label_field])
            rows.append(
                {
                    "split": split,
                    "row_index": int(row_index),
                    "sequence_id": sequence_id,
                    "label": label,
                    "sequence": sequence,
                }
            )
    return pd.DataFrame(rows)


def write_ipromp_fastas(
    config: BenchmarkConfig,
    frames: dict[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    mapping: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Write SeqTrainer split CSV rows as iPro-MP-compatible FASTA files."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = mapping if mapping is not None else build_ipromp_mapping(config, frames)
    written: dict[str, Path] = {}
    for split in SPLIT_ORDER:
        split_mapping = mapping[mapping["split"] == split].sort_values("row_index")
        lines: list[str] = []
        for row in split_mapping.itertuples(index=False):
            header = (
                f">seqtrainer|split={row.split}|row_index={row.row_index}|"
                f"sequence_id={row.sequence_id}|label={row.label}"
            )
            lines.append(header)
            lines.append(str(row.sequence))
        path = out_dir / f"{split}.fasta"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written[split] = path
    return written


def write_ipromp_run_commands(
    config: BenchmarkConfig,
    fasta_paths: dict[str, Path],
    output_dir: str | Path,
) -> Path:
    """Write a shell script with official iPro-MP prediction commands."""
    params = dict(config.model.params)
    out_dir = Path(output_dir)
    external_predictions_dir = out_dir / "external_predictions"
    external_predictions_dir.mkdir(parents=True, exist_ok=True)
    species_id = int(params.get("species_id", 10))
    repo_dir = str(params.get("ipromp_repo_dir", "./external/iPro-MP"))
    script = out_dir / "ipromp_run_commands.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'SEQTRAINER_ROOT="$(pwd)"',
        f'IPROMP_REPO_DIR="{repo_dir}"',
        'cd "${IPROMP_REPO_DIR}"',
        "",
    ]
    for split in SPLIT_ORDER:
        fasta_path = fasta_paths[split]
        pred_path = external_predictions_dir / f"{split}_predictions.csv"
        lines.extend(
            [
                f"python iPro-MP_predict.py \\",
                f"  -i {_script_path(fasta_path)} \\",
                f"  -s {species_id} \\",
                f"  -o {_script_path(pred_path)}",
                "",
            ]
        )
    script.write_text("\n".join(lines), encoding="utf-8")
    return script


def write_external_prediction_schema(output_dir: str | Path) -> Path:
    """Document accepted external iPro-MP prediction formats."""
    path = Path(output_dir) / "external_prediction_schema.md"
    path.write_text(
        """# External iPro-MP Prediction Schema

SeqTrainer accepts two external prediction formats.

## Official iPro-MP CSV

The official iPro-MP output is expected to include:

```text
Sequence,Prediction,Probability
```

Use separate files for `validation`, `test`, and optionally `train`. SeqTrainer
joins official rows back to `ipromp_id_mapping.csv` by exact sequence. If a split
contains duplicate sequences, official output is ambiguous unless a normalized
`sequence_id` column is provided.

## SeqTrainer-Normalized CSV

```text
split,sequence_id,label,probability
```

Hard-label-only files may use `prediction` instead of `probability`, but then
AUROC/AUPRC and validation-threshold selection are unavailable.
""",
        encoding="utf-8",
    )
    return path


def normalize_ipromp_predictions(
    config: BenchmarkConfig,
    *,
    mapping_csv: str | Path,
    validation_predictions_csv: str | Path | None = None,
    test_predictions_csv: str | Path | None = None,
    train_predictions_csv: str | Path | None = None,
    predictions_csv: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Normalize official iPro-MP or SeqTrainer prediction files."""
    mapping_path = _resolve_input_path(mapping_csv, base_dir)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing iPro-MP mapping CSV: {mapping_path}")
    mapping = pd.read_csv(mapping_path)
    _validate_mapping(mapping)

    if predictions_csv is not None:
        combined = _read_prediction_table(_resolve_input_path(predictions_csv, base_dir))
        if "split" in combined.columns:
            return _normalize_seqtrainer_predictions(config, combined, mapping)
        raise ValueError("Combined iPro-MP predictions must include a split column.")

    split_paths = {
        "train": train_predictions_csv,
        "validation": validation_predictions_csv,
        "test": test_predictions_csv,
    }
    if not validation_predictions_csv or not test_predictions_csv:
        raise FileNotFoundError("Both validation and test iPro-MP prediction CSVs are required for evaluation.")

    normalized = []
    for split, raw_path in split_paths.items():
        if not raw_path:
            continue
        table = _read_prediction_table(_resolve_input_path(raw_path, base_dir))
        if "split" in table.columns:
            normalized.append(_normalize_seqtrainer_predictions(config, table, mapping, expected_split=split))
        else:
            normalized.append(_normalize_official_predictions(config, table, mapping, split))
    return pd.concat(normalized, ignore_index=True)


def normalize_dna_sequence(sequence: str) -> str:
    """Normalize a DNA sequence and fail on invalid bases."""
    cleaned = sequence.strip().upper().replace("U", "T")
    if not cleaned:
        raise ValueError("Cannot write an empty sequence to FASTA")
    invalid = sorted(set(cleaned).difference(DNA_ALPHABET))
    if invalid:
        raise ValueError(f"Invalid DNA bases for iPro-MP FASTA: {invalid}")
    return cleaned


def _normalize_seqtrainer_predictions(
    config: BenchmarkConfig,
    predictions: pd.DataFrame,
    mapping: pd.DataFrame,
    expected_split: str | None = None,
) -> pd.DataFrame:
    table = predictions.copy()
    if expected_split is not None:
        if "split" not in table.columns:
            table["split"] = expected_split
        elif set(table["split"].astype(str)) != {expected_split}:
            raise ValueError(f"Prediction file for {expected_split} contains rows from another split.")
    required = {"split"}
    if "sequence_id" not in table.columns:
        required.update({"label"})
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Normalized iPro-MP predictions are missing columns: {sorted(missing)}")

    if "sequence_id" in table.columns:
        merged = table.merge(mapping, on=["split", "sequence_id"], how="left", suffixes=("_pred", ""))
    else:
        merged = table.copy()
        merged["row_index"] = merged.groupby("split").cumcount()
        merged = merged.merge(mapping, on=["split", "row_index"], how="left", suffixes=("_pred", ""))
    if merged["label"].isna().any() or merged["sequence"].isna().any():
        raise ValueError("Could not map every normalized iPro-MP prediction row back to the benchmark split.")
    if "label_pred" in merged.columns:
        merged = merged.drop(columns=["label_pred"])
    return _standard_prediction_columns(merged)


def _normalize_official_predictions(
    config: BenchmarkConfig,
    predictions: pd.DataFrame,
    mapping: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    params = dict(config.model.params)
    sequence_col = str(params.get("external_sequence_column", "Sequence"))
    probability_col = str(params.get("external_probability_column", "Probability"))
    prediction_col = str(params.get("external_prediction_column", "Prediction"))
    missing = {sequence_col}.difference(predictions.columns)
    if missing:
        raise ValueError(f"Official iPro-MP predictions are missing columns: {sorted(missing)}")

    table = predictions.copy()
    table["sequence"] = table[sequence_col].astype(str).map(normalize_dna_sequence)
    if probability_col in table.columns:
        table["probability"] = table[probability_col].astype(float)
    if prediction_col in table.columns:
        table["source_prediction"] = table[prediction_col]
        if "probability" not in table.columns:
            table["prediction"] = table[prediction_col].astype(int)
    split_mapping = mapping[mapping["split"] == split].copy()
    duplicated = split_mapping["sequence"].duplicated(keep=False)
    if duplicated.any() and "sequence_id" not in table.columns:
        examples = split_mapping.loc[duplicated, "sequence"].head(3).tolist()
        raise ValueError(
            "Official iPro-MP output joins by Sequence, but this split has duplicate sequences. "
            f"Use SeqTrainer-normalized output with sequence_id. Examples: {examples}"
        )
    merged = split_mapping.merge(table, on="sequence", how="left", suffixes=("", "_pred"))
    if merged["probability"].isna().any() and "prediction" not in merged.columns:
        raise ValueError(f"Missing iPro-MP predictions for at least one {split} sequence.")
    if len(merged) != len(split_mapping):
        raise ValueError(f"iPro-MP prediction count mismatch for {split}: expected {len(split_mapping)}, got {len(merged)}")
    return _standard_prediction_columns(merged)


def _standard_prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for score_column in ("score", "positive_score", "promoter_score"):
        if "probability" not in out.columns and score_column in out.columns:
            out = out.rename(columns={score_column: "probability"})
            break
    if "label_pred" in out.columns and "label" not in out.columns:
        out = out.rename(columns={"label_pred": "label"})
    if "probability" in out.columns:
        out["probability"] = out["probability"].astype(float)
    if "prediction" in out.columns:
        out["prediction"] = out["prediction"].astype(int)
    if "source_prediction" not in out.columns and "prediction" in out.columns:
        out["source_prediction"] = out["prediction"]
    columns = ["split", "row_index", "sequence_id", "label", "sequence"]
    for optional in ("probability", "prediction", "source_prediction"):
        if optional in out.columns:
            columns.append(optional)
    return out[columns].sort_values(["split", "row_index"]).reset_index(drop=True)


def _sequence_id(config: BenchmarkConfig, row: pd.Series, split: str, row_index: int) -> str:
    id_field = config.dataset.id_field
    if id_field and id_field in row and pd.notna(row[id_field]):
        return str(row[id_field])
    return f"{split}_{row_index:06d}"


def _validate_mapping(mapping: pd.DataFrame) -> None:
    required = {"split", "row_index", "sequence_id", "label", "sequence"}
    missing = required.difference(mapping.columns)
    if missing:
        raise ValueError(f"iPro-MP mapping CSV is missing columns: {sorted(missing)}")


def _read_prediction_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing iPro-MP prediction file: {path}")
    return pd.read_csv(path, sep=None, engine="python")


def _resolve_output_path(
    configured: Any,
    fallback: Path,
    *,
    run_dir: Path | None = None,
    configured_root: str | None = None,
) -> Path:
    if configured:
        path = Path(str(configured))
        if run_dir is not None and configured_root:
            try:
                relative = path.relative_to(Path(configured_root))
                return run_dir / relative
            except ValueError:
                pass
        return path
    return fallback


def _resolve_input_path(path: str | Path, base_dir: str | Path | None) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return Path(base_dir or Path.cwd()) / resolved


def _as_posix(path: str | Path) -> str:
    return Path(path).as_posix()


def _script_path(path: str | Path) -> str:
    resolved = Path(path)
    if resolved.is_absolute():
        return _as_posix(resolved)
    return '"${SEQTRAINER_ROOT}/' + _as_posix(resolved) + '"'
