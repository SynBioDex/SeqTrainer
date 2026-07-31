"""Generate held-out DNA continuations and compile sequence-realism diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import subprocess
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from seqtrainer.data.bacteria_titan import TokenStreamDataset

from .config import MemoryMode, StageCModelConfig
from .memory_trace_cli import _taxonomy_labels
from .model import BlockStates, StageCPaperMACForCausalLM, detach_stream_states
from .study import StudyProtocol
from .tokenizers import SixMerTokenizer


DNA = frozenset("ACGT")
START_CODONS = frozenset(("ATG", "GTG", "TTG"))
STOP_CODONS = frozenset(("TAA", "TAG", "TGA"))


def _optional_positive_int(value: str) -> int | None:
    if value.lower() in {"none", "unrestricted"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive or 'none'")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--taxonomy-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--species", default="Escherichia coli")
    parser.add_argument("--prompts", type=int, default=4)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--new-tokens", type=int, default=1024)
    parser.add_argument("--temperatures", default="0.8,1.0,1.2")
    parser.add_argument("--top-k", type=_optional_positive_int, default=128)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260781)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--memory-mode", choices=[item.value for item in MemoryMode], default="adaptive")
    parser.add_argument("--prodigal", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--protocol-amendment", type=Path, action="append", default=[])
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if bool(args.protocol) != bool(args.run_id):
        parser.error("--protocol and --run-id must be supplied together")
    if args.protocol_amendment and not args.protocol:
        parser.error("--protocol-amendment requires --protocol and --run-id")
    if args.prompts <= 0 or args.new_tokens <= 0:
        parser.error("--prompts and --new-tokens must be positive")
    if not 1 <= args.prompt_tokens <= 32:
        parser.error("--prompt-tokens must be between 1 and 32")
    if not 0.0 < args.top_p <= 1.0:
        parser.error("--top-p must be in (0, 1]")
    try:
        args.temperatures = tuple(float(item) for item in args.temperatures.split(","))
    except ValueError as error:
        parser.error(f"--temperatures must be comma-separated numbers: {error}")
    if not args.temperatures or any(value <= 0 for value in args.temperatures):
        parser.error("every temperature must be positive")
    return args


def _load_checkpoint(path: Path, device: torch.device) -> Mapping[str, object]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload is invalid")
    return payload


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def _kmer_counts(sequences: Iterable[str], k: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    for sequence in sequences:
        normalized = sequence.upper()
        for index in range(max(0, len(normalized) - k + 1)):
            kmer = normalized[index : index + k]
            if set(kmer).issubset(DNA):
                counts[kmer] += 1
    return counts


def _jensen_shannon(left: Counter[str], right: Counter[str]) -> float:
    keys = sorted(set(left) | set(right))
    if not keys or not sum(left.values()) or not sum(right.values()):
        return 0.0
    p = np.asarray([left[key] for key in keys], dtype=np.float64)
    q = np.asarray([right[key] for key in keys], dtype=np.float64)
    p /= p.sum()
    q /= q.sum()
    midpoint = 0.5 * (p + q)

    def divergence(values: np.ndarray) -> float:
        active = values > 0
        return float(np.sum(values[active] * np.log2(values[active] / midpoint[active])))

    return 0.5 * (divergence(p) + divergence(q))


def _base_entropy(sequence: str) -> float:
    counts = Counter(base for base in sequence if base in DNA)
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _max_homopolymer(sequence: str) -> int:
    longest = current = 0
    previous = ""
    for base in sequence:
        current = current + 1 if base == previous else 1
        previous = base
        longest = max(longest, current)
    return longest


def _sequence_metrics(sequence_id: str, sequence: str, group: str) -> dict[str, object]:
    normalized = sequence.upper()
    canonical = sum(base in DNA for base in normalized)
    gc = normalized.count("G") + normalized.count("C")
    sixmers = _kmer_counts((normalized,), 6)
    possible = max(1, len(normalized) - 5)
    aligned_sixmers = [
        normalized[offset : offset + 6]
        for offset in range(0, len(normalized) - 5, 6)
        if set(normalized[offset : offset + 6]).issubset(DNA)
    ]
    orfs = _find_orfs(normalized)
    return {
        "sequence_id": sequence_id,
        "group": group,
        "bases": len(normalized),
        "gc_fraction": gc / max(canonical, 1),
        "n_fraction": normalized.count("N") / max(len(normalized), 1),
        "base_entropy_bits": _base_entropy(normalized),
        "max_homopolymer": _max_homopolymer(normalized),
        "unique_6mer_fraction": len(sixmers) / possible,
        "aligned_unique_6mer_fraction": (
            len(set(aligned_sixmers)) / max(len(aligned_sixmers), 1)
        ),
        "heuristic_orfs_at_least_90bp": len(orfs),
        "heuristic_longest_orf_bases": max(
            (int(record["length"]) for record in orfs),
            default=0,
        ),
    }


def _find_orfs(sequence: str, minimum_bases: int = 90) -> list[dict[str, object]]:
    """Return a transparent six-frame ORF heuristic; this is not a gene caller."""

    records: list[dict[str, object]] = []
    for strand, oriented in (("+", sequence.upper()), ("-", _reverse_complement(sequence.upper()))):
        for frame in range(3):
            start: int | None = None
            start_codon = ""
            for position in range(frame, len(oriented) - 2, 3):
                codon = oriented[position : position + 3]
                if start is None and codon in START_CODONS:
                    start, start_codon = position, codon
                elif start is not None and codon in STOP_CODONS:
                    length = position + 3 - start
                    if length >= minimum_bases:
                        records.append(
                            {
                                "strand": strand,
                                "frame": frame,
                                "start": start,
                                "end": position + 3,
                                "length": length,
                                "start_codon": start_codon,
                                "stop_codon": codon,
                            }
                        )
                    start = None
                    start_codon = ""
    return records


def _sample_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    top_p: float,
    generator: torch.Generator,
    forbidden_ids: Sequence[int],
) -> int:
    values = logits.detach().float() / temperature
    for token_id in forbidden_ids:
        if 0 <= token_id < values.numel():
            values[token_id] = -torch.inf
    if top_k is not None and top_k < values.numel():
        threshold = torch.topk(values, top_k).values[-1]
        values = values.masked_fill(values < threshold, -torch.inf)
    if top_p < 1.0:
        ordered, indices = torch.sort(values, descending=True)
        probabilities = torch.softmax(ordered, dim=-1)
        cumulative = torch.cumsum(probabilities, dim=-1)
        remove = cumulative - probabilities > top_p
        ordered = ordered.masked_fill(remove, -torch.inf)
        filtered = torch.full_like(values, -torch.inf)
        filtered.scatter_(0, indices, ordered)
        values = filtered
    probabilities = torch.softmax(values, dim=-1)
    if not torch.isfinite(probabilities).all() or float(probabilities.sum()) <= 0:
        raise RuntimeError("generation produced an invalid sampling distribution")
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def _forward_prefix(
    model: StageCPaperMACForCausalLM,
    states: BlockStates,
    tokens: Sequence[int],
    *,
    device: torch.device,
    memory_mode: str,
) -> object:
    width = model.config.segment_length
    if not 1 <= len(tokens) <= width:
        raise ValueError("generation prefix must fit one Stage C segment")
    padding = width - len(tokens)
    input_ids = torch.tensor(
        [list(tokens) + [model.config.pad_token_id] * padding],
        dtype=torch.long,
        device=device,
    )
    valid_mask = torch.tensor(
        [[True] * len(tokens) + [False] * padding],
        dtype=torch.bool,
        device=device,
    )
    with torch.enable_grad():
        return model.forward_segment(
            (states,),
            input_ids,
            valid_mask=valid_mask,
            memory_mode=memory_mode,
        )


def generate_continuation(
    model: StageCPaperMACForCausalLM,
    prompt_tokens: Sequence[int],
    *,
    new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float,
    seed: int,
    device: torch.device,
    memory_mode: str = "adaptive",
) -> tuple[list[int], dict[str, float]]:
    """Generate without writing overlapping prefixes repeatedly into memory."""

    if not prompt_tokens or len(prompt_tokens) > model.config.segment_length:
        raise ValueError("prompt_tokens must contain one partial or complete segment")
    generator = torch.Generator(device=device.type).manual_seed(seed)
    states = model.initial_states(f"generation-{seed}")
    current = list(map(int, prompt_tokens))
    generated: list[int] = []
    diagnostics: dict[str, list[float]] = defaultdict(list)
    while len(generated) < new_tokens:
        output = _forward_prefix(
            model,
            states,
            current,
            device=device,
            memory_mode=memory_mode,
        )
        next_token = _sample_token(
            output.logits[0, len(current) - 1],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
            forbidden_ids=(model.config.pad_token_id, 1),
        )
        generated.append(next_token)
        for name in (
            "retrieval_norm",
            "memory_update_norm",
            "surprise_norm",
            "state_drift_norm",
        ):
            diagnostics[name].append(float(getattr(output, name)))
        if len(current) == model.config.segment_length:
            states = detach_stream_states(output.states[0])
            current = [next_token]
        else:
            current.append(next_token)
        del output
    return generated, {
        f"mean_{name}": float(statistics.fmean(values))
        for name, values in diagnostics.items()
    }


def _stream_tokens(stream: Sequence[object], limit: int) -> list[int]:
    tokens: list[int] = []
    for segment in stream:
        tokens.extend(
            int(token)
            for token, valid in zip(segment.input_ids, segment.valid_mask)
            if valid
        )
        if len(tokens) >= limit:
            break
    return tokens[:limit]


def _select_prompt_streams(
    dataset: TokenStreamDataset,
    taxonomy: Mapping[str, str],
    *,
    split: str,
    species: str,
    count: int,
    minimum_tokens: int,
    seed: int,
) -> list[tuple[str, Sequence[object]]]:
    by_accession: dict[str, list[tuple[str, Sequence[object]]]] = defaultdict(list)
    for stream_id, stream in dataset.streams(split=split).items():
        first = stream[0]
        if taxonomy.get(first.accession) == species and len(stream) * 32 >= minimum_tokens:
            by_accession[first.accession].append((stream_id, stream))
    accessions = sorted(by_accession)
    random.Random(seed).shuffle(accessions)
    selected: list[tuple[str, Sequence[object]]] = []
    for accession in accessions[:count]:
        selected.append(max(by_accession[accession], key=lambda item: len(item[1])))
    if len(selected) < count:
        raise ValueError(
            f"requested {count} distinct {species} prompts, found {len(selected)} eligible accessions"
        )
    return selected


def _write_fasta(path: Path, records: Mapping[str, str]) -> None:
    with path.open("w", encoding="ascii") as handle:
        for name, sequence in records.items():
            handle.write(f">{name}\n")
            for offset in range(0, len(sequence), 80):
                handle.write(sequence[offset : offset + 80] + "\n")


def _parse_gff(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 9 or fields[2] != "CDS":
            continue
        attributes = dict(
            item.split("=", 1)
            for item in fields[8].split(";")
            if "=" in item
        )
        start, end = int(fields[3]), int(fields[4])
        rows.append(
            {
                "sequence_id": fields[0],
                "start": start,
                "end": end,
                "length": end - start + 1,
                "strand": fields[6],
                "partial": attributes.get("partial"),
            }
        )
    return rows


def _prodigal_summary(
    executable: Path,
    fasta: Path,
    output_dir: Path,
    sequence_groups: Mapping[str, str],
    sequence_lengths: Mapping[str, int],
) -> dict[str, object]:
    gff = output_dir / f"{fasta.stem}.prodigal.gff"
    proteins = output_dir / f"{fasta.stem}.prodigal.faa"
    genes = output_dir / f"{fasta.stem}.prodigal.fna"
    log = output_dir / f"{fasta.stem}.prodigal.log"
    result = subprocess.run(
        [
            str(executable),
            "-i",
            str(fasta),
            "-o",
            str(gff),
            "-a",
            str(proteins),
            "-d",
            str(genes),
            "-f",
            "gff",
            "-p",
            "meta",
            "-q",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Prodigal failed for {fasta.name}; see {log}")
    calls = _parse_gff(gff)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for call in calls:
        grouped[sequence_groups[str(call["sequence_id"])]].append(call)
    summaries: dict[str, object] = {}
    for group in sorted(set(sequence_groups.values())):
        group_ids = [key for key, value in sequence_groups.items() if value == group]
        group_calls = grouped[group]
        total_bases = sum(sequence_lengths[key] for key in group_ids)
        intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for call in group_calls:
            intervals[str(call["sequence_id"])].append((int(call["start"]), int(call["end"])))
        coding_bases = 0
        intergenic: list[int] = []
        for sequence_id in group_ids:
            merged: list[list[int]] = []
            for start, end in sorted(intervals.get(sequence_id, [])):
                if merged and start <= merged[-1][1] + 1:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
            coding_bases += sum(end - start + 1 for start, end in merged)
            cursor = 1
            for start, end in merged:
                if start > cursor:
                    intergenic.append(start - cursor)
                cursor = end + 1
            if cursor <= sequence_lengths[sequence_id]:
                intergenic.append(sequence_lengths[sequence_id] - cursor + 1)
        lengths = [int(call["length"]) for call in group_calls]
        summaries[group] = {
            "sequences": len(group_ids),
            "genes": len(group_calls),
            "genes_per_10kb": len(group_calls) * 10_000 / max(total_bases, 1),
            "coding_density": coding_bases / max(total_bases, 1),
            "median_gene_bases": statistics.median(lengths) if lengths else 0.0,
            "median_intergenic_bases": statistics.median(intergenic) if intergenic else 0.0,
            "complete_gene_fraction": (
                sum(call["partial"] == "00" for call in group_calls) / len(group_calls)
                if group_calls
                else 0.0
            ),
        }
    return {"status": "completed", "groups": summaries}


def _mean_metrics(rows: Sequence[Mapping[str, object]], group: str) -> dict[str, float]:
    selected = [row for row in rows if row["group"] == group]
    return {
        key: float(statistics.fmean(float(row[key]) for row in selected))
        for key in (
            "bases",
            "gc_fraction",
            "n_fraction",
            "base_entropy_bits",
            "max_homopolymer",
            "unique_6mer_fraction",
            "aligned_unique_6mer_fraction",
            "heuristic_orfs_at_least_90bp",
            "heuristic_longest_orf_bases",
        )
    }


def _bar_svg(labels: Sequence[str], values: Sequence[float], *, title: str, y_label: str) -> str:
    width, height, margin = 900, 520, 80
    maximum = max(max(values, default=0.0), 1e-12)
    bar_width = (width - 2 * margin) / max(len(labels), 1)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="34" font-family="sans-serif" font-size="20">{title}</text>',
        f'<text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle" font-family="sans-serif">{y_label}</text>',
    ]
    for index, (label, value) in enumerate(zip(labels, values)):
        x = margin + index * bar_width + 8
        bar_height = value / maximum * (height - 2 * margin)
        y = height - margin - bar_height
        elements.extend(
            (
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(4.0, bar_width - 16):.1f}" height="{bar_height:.1f}" fill="#2563eb"/>',
                f'<text x="{x + (bar_width - 16) / 2:.1f}" y="{height - margin + 18}" text-anchor="middle" font-family="sans-serif" font-size="11">{label}</text>',
                f'<text x="{x + (bar_width - 16) / 2:.1f}" y="{max(50.0, y - 5):.1f}" text-anchor="middle" font-family="sans-serif" font-size="11">{value:.4f}</text>',
            )
        )
    elements.append("</svg>")
    return "\n".join(elements)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.protocol:
        StudyProtocol.from_path(args.protocol).validate_run_config(
            args.run_id,
            {"phase": "analysis"},
            amendment_paths=args.protocol_amendment,
        )
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    dataset = TokenStreamDataset(args.dataset_dir, verify_checksums=True)
    payload = _load_checkpoint(args.checkpoint, device)
    dataset_fingerprint = hashlib.sha256(
        (args.dataset_dir / "token_stream_manifest.json").read_bytes()
    ).hexdigest()
    if payload.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("checkpoint and generation dataset fingerprints differ")
    raw_config = payload.get("model_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("checkpoint is missing model_config")
    config = StageCModelConfig.from_dict(raw_config)
    if config.tokenizer_name != "nonoverlap_6mer_v1":
        raise ValueError("this generation study currently requires nonoverlap_6mer_v1")
    tokenizer = SixMerTokenizer()
    if tokenizer.spec.checksum != config.tokenizer_checksum:
        raise ValueError("checkpoint tokenizer checksum differs from SixMerTokenizer")
    model = StageCPaperMACForCausalLM(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    # The paper recurrence defines surprise as the gradient of its associative
    # loss with respect to the functional memory weights. Keep autograd enabled
    # even in evaluation mode; generation never calls an outer backward pass.

    taxonomy = _taxonomy_labels(args.taxonomy_manifest, "species")
    required_tokens = args.prompt_tokens + args.new_tokens
    prompts = _select_prompt_streams(
        dataset,
        taxonomy,
        split=args.split,
        species=args.species,
        count=args.prompts,
        minimum_tokens=required_tokens,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_records: dict[str, str] = {}
    reference_records: dict[str, str] = {}
    generated_records: dict[str, str] = {}
    generation_rows: list[dict[str, object]] = []
    sequence_groups: dict[str, str] = {}
    sequence_lengths: dict[str, int] = {}
    sequence_rows: list[dict[str, object]] = []

    for prompt_index, (stream_id, stream) in enumerate(prompts):
        tokens = _stream_tokens(stream, required_tokens)
        prompt_tokens = tokens[: args.prompt_tokens]
        reference_tokens = tokens[args.prompt_tokens : required_tokens]
        prompt_id = f"prompt_{prompt_index:03d}"
        reference_id = f"reference_{prompt_index:03d}"
        prompt_records[prompt_id] = tokenizer.decode(prompt_tokens)
        reference_records[reference_id] = tokenizer.decode(reference_tokens)
        sequence_groups[reference_id] = "reference"
        sequence_lengths[reference_id] = len(reference_records[reference_id])
        sequence_rows.append(
            _sequence_metrics(reference_id, reference_records[reference_id], "reference")
        )
        for temperature_index, temperature in enumerate(args.temperatures):
            policy = f"temperature_{temperature:g}"
            sequence_id = f"generated_t{temperature_index}_{prompt_index:03d}"
            generated_tokens, diagnostics = generate_continuation(
                model,
                prompt_tokens,
                new_tokens=args.new_tokens,
                temperature=temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=args.seed + 10_000 * temperature_index + prompt_index,
                device=device,
                memory_mode=args.memory_mode,
            )
            sequence = tokenizer.decode(generated_tokens)
            generated_records[sequence_id] = sequence
            sequence_groups[sequence_id] = policy
            sequence_lengths[sequence_id] = len(sequence)
            sequence_rows.append(_sequence_metrics(sequence_id, sequence, policy))
            heuristic_orfs = _find_orfs(sequence)
            generation_rows.append(
                {
                    "sequence_id": sequence_id,
                    "policy": policy,
                    "temperature": temperature,
                    "prompt_id": prompt_id,
                    "source_stream_id": stream_id,
                    "source_accession": stream[0].accession,
                    "tokens": len(generated_tokens),
                    "bases": len(sequence),
                    "heuristic_orfs_at_least_90bp": len(heuristic_orfs),
                    "heuristic_longest_orf_bases": max(
                        (int(row["length"]) for row in heuristic_orfs),
                        default=0,
                    ),
                    **diagnostics,
                }
            )
            print(
                json.dumps(
                    {
                        "sequence_id": sequence_id,
                        "policy": policy,
                        "bases": len(sequence),
                        "completed": len(generation_rows),
                        "total": len(prompts) * len(args.temperatures),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    generated_fasta = args.output_dir / "generated_sequences.fasta"
    reference_fasta = args.output_dir / "heldout_reference_continuations.fasta"
    _write_fasta(args.output_dir / "heldout_prompts.fasta", prompt_records)
    _write_fasta(generated_fasta, generated_records)
    _write_fasta(reference_fasta, reference_records)
    groups = ["reference", *[f"temperature_{value:g}" for value in args.temperatures]]
    distribution_summary = {group: _mean_metrics(sequence_rows, group) for group in groups}
    kmer_jsd: dict[str, dict[str, float]] = {}
    reference_sequences = list(reference_records.values())
    for group in groups[1:]:
        generated = [
            sequence
            for sequence_id, sequence in generated_records.items()
            if sequence_groups[sequence_id] == group
        ]
        kmer_jsd[group] = {
            str(k): _jensen_shannon(
                _kmer_counts(generated, k),
                _kmer_counts(reference_sequences, k),
            )
            for k in range(1, 7)
        }
    prodigal = args.prodigal or (
        Path(found) if (found := shutil.which("prodigal")) else None
    )
    prodigal_summary: dict[str, object] = {
        "status": "unavailable",
        "reason": "install Prodigal or pass --prodigal; heuristic ORFs remain available",
    }
    if prodigal is not None:
        generated_prodigal = _prodigal_summary(
            prodigal,
            generated_fasta,
            args.output_dir,
            sequence_groups,
            sequence_lengths,
        )
        reference_prodigal = _prodigal_summary(
            prodigal,
            reference_fasta,
            args.output_dir,
            {key: "reference" for key in reference_records},
            sequence_lengths,
        )
        prodigal_summary = {
            "status": "completed",
            "groups": {
                **generated_prodigal["groups"],
                **reference_prodigal["groups"],
            },
        }

    report = {
        "format_version": 1,
        "classification": "exploratory_conditional_generation_diagnostics",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "dataset_fingerprint": dataset_fingerprint,
        "taxonomy_manifest": str(args.taxonomy_manifest),
        "taxonomy_manifest_sha256": hashlib.sha256(
            args.taxonomy_manifest.read_bytes()
        ).hexdigest(),
        "split": args.split,
        "species": args.species,
        "memory_mode": args.memory_mode,
        "prompt_tokens": args.prompt_tokens,
        "new_tokens": args.new_tokens,
        "temperatures": args.temperatures,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "seed": args.seed,
        "model_config": config.to_dict(),
        "generation_rows": generation_rows,
        "distribution_summary": distribution_summary,
        "kmer_jsd_to_heldout_reference": kmer_jsd,
        "prodigal": prodigal_summary,
        "claim_limits": [
            "Conditional sequence realism is not de novo genome generation.",
            "Prodigal calls are computational predictions, not evidence of expressed or functional genes.",
            "ORF, intergenic, k-mer, and GC similarity do not establish promoter activity, fitness, or safety.",
            "Generation quality does not establish an adaptive-memory benefit without matched controls.",
        ],
    }
    (args.output_dir / "generation_evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "sequence_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sequence_rows[0]))
        writer.writeheader()
        writer.writerows(sequence_rows)
    labels = list(distribution_summary)
    (args.output_dir / "generation_gc.svg").write_text(
        _bar_svg(
            labels,
            [distribution_summary[label]["gc_fraction"] for label in labels],
            title="Generated versus held-out GC fraction",
            y_label="GC fraction",
        ),
        encoding="utf-8",
    )
    (args.output_dir / "generation_kmer_jsd.svg").write_text(
        _bar_svg(
            list(kmer_jsd),
            [kmer_jsd[label]["6"] for label in kmer_jsd],
            title="6-mer divergence from held-out continuations",
            y_label="Jensen-Shannon divergence (bits)",
        ),
        encoding="utf-8",
    )
    lines = [
        "# Stage C conditional-generation diagnostics",
        "",
        f"- Checkpoint SHA-256: `{report['checkpoint_sha256']}`",
        f"- Held-out split/species: `{args.split}` / `{args.species}`",
        f"- Distinct prompt accessions: `{len(prompts)}`",
        f"- Generated continuations: `{len(generated_records)}`",
        f"- Tokens per continuation: `{args.new_tokens}`",
        f"- Prodigal: `{prodigal_summary['status']}`",
        "",
        "## Distribution summary",
        "",
        "| Group | Mean bases | GC | entropy | max homopolymer | overlapping 6-mer diversity | aligned 6-mer diversity | ORFs >=90 bp | longest ORF |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *[
            "| {group} | {bases:.1f} | {gc_fraction:.4f} | "
            "{base_entropy_bits:.4f} | {max_homopolymer:.2f} | "
            "{unique_6mer_fraction:.4f} | {aligned_unique_6mer_fraction:.4f} | "
            "{heuristic_orfs_at_least_90bp:.2f} | "
            "{heuristic_longest_orf_bases:.2f} |".format(
                group=group, **distribution_summary[group]
            )
            for group in labels
        ],
        "",
        "## Interpretation limits",
        "",
        *[f"- {item}" for item in report["claim_limits"]],
        "",
    ]
    (args.output_dir / "GENERATION_REPORT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(args.output_dir / "GENERATION_REPORT.md"),
                "manifest": str(args.output_dir / "generation_evaluation.json"),
                "generated_fasta": str(generated_fasta),
                "prodigal": prodigal_summary["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
