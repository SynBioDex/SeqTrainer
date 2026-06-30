"""Run the official iPro-MP fold ensemble with SeqTrainer-stable outputs."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class FastaRecord:
    """One FASTA record with the SeqTrainer split and stable sequence ID."""

    sequence_id: str
    sequence: str
    split: str


def read_seqtrainer_fasta(path: str | Path, *, expected_split: str | None = None) -> list[FastaRecord]:
    """Read FASTA records produced by ``benchmark prepare-ipromp``."""
    records: list[FastaRecord] = []
    header: str | None = None
    sequence_lines: list[str] = []

    def append_record() -> None:
        if header is None:
            return
        metadata = _parse_fasta_header(header)
        split = metadata.get("split", expected_split or "")
        if expected_split is not None and split != expected_split:
            raise ValueError(f"Expected split {expected_split!r}, found {split!r} in {path}")
        sequence_id = metadata.get("sequence_id") or metadata.get("id")
        if not sequence_id:
            raise ValueError(f"FASTA header is missing sequence_id: {header}")
        sequence = "".join(sequence_lines).strip().upper().replace("U", "T")
        if not sequence:
            raise ValueError(f"FASTA record {sequence_id!r} has an empty sequence")
        invalid = sorted(set(sequence).difference("ACGTN"))
        if invalid:
            raise ValueError(f"FASTA record {sequence_id!r} contains invalid bases: {invalid}")
        records.append(FastaRecord(sequence_id=sequence_id, sequence=sequence, split=split))

    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            append_record()
            header = line[1:]
            sequence_lines = []
        elif header is None:
            raise ValueError(f"Sequence encountered before the first FASTA header in {path}")
        else:
            sequence_lines.append(line)
    append_record()

    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def fold_checkpoint_paths(model_dir: str | Path, species_id: int, *, folds: int = 5) -> list[Path]:
    """Return the official fold paths and fail clearly when weights are missing."""
    root = Path(model_dir)
    paths = [root / f"{species_id}_fold_{fold}.pth" for fold in range(1, folds + 1)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing iPro-MP fold checkpoint(s): " + ", ".join(missing) + ". "
            "Download the species-specific weights from Zenodo record 15180139."
        )
    return paths


def normalize_state_dict(raw_state: Any) -> dict[str, Any]:
    """Normalize common checkpoint wrappers without changing tensor values."""
    state = raw_state
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = state.get(key)
            if isinstance(nested, dict):
                state = nested
                break
    if not isinstance(state, dict):
        raise TypeError("iPro-MP checkpoint must contain a model state dictionary")
    return {
        (key.removeprefix("module.") if isinstance(key, str) else key): value
        for key, value in state.items()
    }


def run_ipromp_ensemble(
    *,
    input_fasta: str | Path,
    output_csv: str | Path,
    split: str,
    dnabert_dir: str | Path,
    model_dir: str | Path,
    species_id: int = 10,
    kmer_size: int = 6,
    max_length: int = 128,
    batch_size: int = 32,
    seed: int = 42,
    device: str = "auto",
) -> dict[str, Any]:
    """Average the five official iPro-MP fold probabilities sequentially."""
    try:
        import numpy as np
        import pandas as pd
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from transformers import BertModel, BertTokenizer
    except ImportError as exc:  # pragma: no cover - depends on the external environment
        raise RuntimeError(
            "iPro-MP inference requires torch, transformers, numpy, and pandas. "
            "Install the Alpine environment documented in the iPro-MP benchmark README."
        ) from exc

    _seed_everything(seed, random_module=random, numpy_module=np, torch_module=torch)
    resolved_device = _resolve_device(device, torch)
    records = read_seqtrainer_fasta(input_fasta, expected_split=split)
    checkpoints = fold_checkpoint_paths(model_dir, species_id)
    tokenizer = BertTokenizer.from_pretrained(str(dnabert_dir))

    class SequenceDataset(Dataset):
        def __len__(self) -> int:
            return len(records)

        def __getitem__(self, index: int) -> dict[str, Any]:
            sequence = records[index].sequence
            kmers = [sequence[i : i + kmer_size] for i in range(len(sequence) - kmer_size + 1)]
            encoded = tokenizer(
                kmers,
                is_split_into_words=True,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
            }

    class PromoterClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bert = BertModel.from_pretrained(str(dnabert_dir))
            self.dropout = nn.Dropout(p=0.3)
            self.fc1 = nn.Linear(self.bert.config.hidden_size, 512)
            self.layer_norm1 = nn.LayerNorm(512)
            self.fc2 = nn.Linear(512, 256)
            self.layer_norm2 = nn.LayerNorm(256)
            self.classifier = nn.Linear(256, 2)
            self.activation = nn.GELU()

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            hidden = self.dropout(hidden)
            hidden = self.layer_norm1(self.activation(self.fc1(hidden)))
            hidden = self.dropout(hidden)
            hidden = self.layer_norm2(self.activation(self.fc2(hidden)))
            return self.classifier(self.dropout(hidden))

    loader = DataLoader(
        SequenceDataset(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=resolved_device.type == "cuda",
    )
    probability_sum = np.zeros(len(records), dtype=np.float64)
    started = time.perf_counter()
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)

    for checkpoint in checkpoints:
        model = PromoterClassifier()
        try:
            raw_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - for older supported torch releases
            raw_state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(normalize_state_dict(raw_state), strict=True)
        model.to(resolved_device)
        model.eval()
        fold_probabilities: list[float] = []
        with torch.inference_mode():
            for batch in loader:
                logits = model(
                    input_ids=batch["input_ids"].to(resolved_device, non_blocking=True),
                    attention_mask=batch["attention_mask"].to(resolved_device, non_blocking=True),
                )
                fold_probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
        probability_sum += np.asarray(fold_probabilities, dtype=np.float64)
        del model, raw_state
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()

    probabilities = probability_sum / len(checkpoints)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "split": [record.split for record in records],
            "sequence_id": [record.sequence_id for record in records],
            "probability": probabilities,
            "prediction": (probabilities >= 0.5).astype(int),
        }
    ).to_csv(output_path, index=False)

    metadata = {
        "status": "completed",
        "species_id": species_id,
        "folds": len(checkpoints),
        "kmer_size": kmer_size,
        "max_length": max_length,
        "batch_size": batch_size,
        "seed": seed,
        "device": str(resolved_device),
        "rows": len(records),
        "runtime_seconds": time.perf_counter() - started,
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated(resolved_device) / (1024**2)
            if resolved_device.type == "cuda"
            else None
        ),
        "input_fasta": str(Path(input_fasta)),
        "output_csv": str(output_path),
        "dnabert_dir": str(Path(dnabert_dir)),
        "checkpoint_paths": [str(path) for path in checkpoints],
    }
    output_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _parse_fasta_header(header: str) -> dict[str, str]:
    fields = header.split("|")
    metadata: dict[str, str] = {"id": fields[0]}
    for field in fields[1:]:
        if "=" in field:
            key, value = field.split("=", 1)
            metadata[key] = value
    return metadata


def _seed_everything(seed: int, *, random_module: Any, numpy_module: Any, torch_module: Any) -> None:
    random_module.seed(seed)
    numpy_module.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def _resolve_device(device: str, torch_module: Any) -> Any:
    if device == "auto":
        device = "cuda" if torch_module.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested for iPro-MP inference, but no CUDA device is available")
    return torch_module.device(device)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the official iPro-MP five-fold prediction ensemble")
    parser.add_argument("--input-fasta", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--dnabert-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--species-id", type=int, default=10)
    parser.add_argument("--kmer-size", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    metadata = run_ipromp_ensemble(
        input_fasta=args.input_fasta,
        output_csv=args.output_csv,
        split=args.split,
        dnabert_dir=args.dnabert_dir,
        model_dir=args.model_dir,
        species_id=args.species_id,
        kmer_size=args.kmer_size,
        max_length=args.max_length,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
