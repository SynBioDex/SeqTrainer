"""DNABERT2 tokenization helpers for shared benchmark splits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import BenchmarkConfig, load_benchmark_config
from .runner import BenchmarkSkipped
from .splits import load_predefined_split_frames, summarize_split_frames


@dataclass(frozen=True)
class DnaBert2TokenizationResult:
    """Paths and metadata produced by DNABERT2 tokenization."""

    output_dir: Path
    tokenized_paths: dict[str, Path]
    metadata_path: Path
    metadata: dict[str, Any]


def prepare_dnabert2_tokenized_splits(
    config: BenchmarkConfig | str | Path,
    *,
    base_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    tokenizer: Any | None = None,
) -> DnaBert2TokenizationResult:
    """Tokenize shared train/validation/test CSV splits for DNABERT2.

    This prepares inspectable CSV artifacts instead of training a model. It is
    useful for Colab setup checks and for verifying that DNABERT2 sees the same
    split rows as CNN/iPro-MP.
    """
    config = load_benchmark_config(config) if not isinstance(config, BenchmarkConfig) else config
    params = dict(config.model.params)
    preprocessing = dict(config.preprocessing.params)
    model_name = str(params.get("model_name", config.model.name))
    trust_remote_code = bool(params.get("trust_remote_code", True))
    allow_download = bool(params.get("allow_download", False))
    require_model_files = bool(params.get("require_model_files", True))
    max_length = int(preprocessing.get("model_max_length", preprocessing.get("max_length", config.preprocessing.sequence_length or 512)))
    padding = str(preprocessing.get("padding", "longest"))
    pad_to_multiple_of = preprocessing.get("pad_to_multiple_of")
    max_rows_per_split = preprocessing.get("max_rows_per_split")

    tokenizer = tokenizer or _load_tokenizer(
        model_name,
        trust_remote_code=trust_remote_code,
        allow_download=allow_download,
        require_model_files=require_model_files,
    )
    frames = load_predefined_split_frames(config, base_dir=base_dir)
    if max_rows_per_split is not None:
        frames = {split: frame.head(int(max_rows_per_split)).copy() for split, frame in frames.items()}

    out_dir = Path(output_dir or config.outputs.output_dir)
    tokenized_dir = out_dir / "dnabert2_tokenized"
    tokenized_dir.mkdir(parents=True, exist_ok=True)
    tokenized_paths: dict[str, Path] = {}
    for split, frame in frames.items():
        encoded = tokenizer(
            frame[config.dataset.sequence_field].astype(str).tolist(),
            padding=padding,
            truncation=True,
            max_length=max_length,
            pad_to_multiple_of=int(pad_to_multiple_of) if pad_to_multiple_of is not None else None,
        )
        tokenized = _tokenized_frame(config, frame, encoded)
        path = tokenized_dir / f"{split}.csv"
        tokenized.to_csv(path, index=False)
        tokenized_paths[split] = path

    metadata = {
        "model_name": model_name,
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_vocab_size": _safe_vocab_size(tokenizer),
        "trust_remote_code": trust_remote_code,
        "allow_download": allow_download,
        "require_model_files": require_model_files,
        "max_length": max_length,
        "padding": padding,
        "pad_to_multiple_of": int(pad_to_multiple_of) if pad_to_multiple_of is not None else None,
        "split_summary": summarize_split_frames(config, frames),
        "tokenized_paths": {split: str(path) for split, path in tokenized_paths.items()},
    }
    metadata_path = out_dir / "dnabert2_tokenization_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return DnaBert2TokenizationResult(out_dir, tokenized_paths, metadata_path, metadata)


def _load_tokenizer(
    model_name: str,
    *,
    trust_remote_code: bool,
    allow_download: bool,
    require_model_files: bool,
) -> Any:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise BenchmarkSkipped(
            "DNABERT2 tokenization requires transformers. Install with `python -m pip install -e \".[torch]\"`."
        ) from exc

    local_files_only = require_model_files and not allow_download
    try:
        return AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    except OSError as exc:
        raise BenchmarkSkipped(
            "DNABERT2 tokenizer files are not available. In Colab, set "
            "`model.params.allow_download=true` for the first run or pre-cache the Hugging Face model files."
        ) from exc


def _tokenized_frame(config: BenchmarkConfig, frame: pd.DataFrame, encoded: dict[str, Any]) -> pd.DataFrame:
    labels = frame[config.dataset.label_field].astype(int).tolist()
    ids = (
        frame[config.dataset.id_field].astype(str).tolist()
        if config.dataset.id_field and config.dataset.id_field in frame
        else [str(i) for i in range(len(frame))]
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", [[1] * len(tokens) for tokens in input_ids])
    rows = []
    for idx, (row_id, label, ids_row, mask_row) in enumerate(zip(ids, labels, input_ids, attention_mask)):
        rows.append(
            {
                "idx": idx,
                "id": row_id,
                "label": label,
                "input_ids": " ".join(str(int(token)) for token in ids_row),
                "attention_mask": " ".join(str(int(token)) for token in mask_row),
                "token_count": int(sum(int(value) for value in mask_row)),
            }
        )
    return pd.DataFrame(rows)


def _safe_vocab_size(tokenizer: Any) -> int | None:
    try:
        return int(len(tokenizer))
    except TypeError:
        return None
