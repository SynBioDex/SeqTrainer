"""Local and accelerator entrypoints for Stage C evidence gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
from typing import Iterable, Sequence

import torch

from seqtrainer.data.bacteria_titan import build_stream_segments
from seqtrainer.data.bacteria_titan.token_shards import iter_fasta
from .baselines import run_statistical_baselines
from .config import StageCModelConfig
from .model import StageCPaperMACForCausalLM, detach_stream_states
from .reporting import write_cpu_pilot_report, write_tokenizer_report, write_training_history
from .tokenizers import (
    Evo2CharTokenizer,
    GenomeTokenizer,
    HuggingFaceBPETokenizer,
    SeqTrainerBaseTokenizer,
    SixMerTokenizer,
    TokenizerMetrics,
    TrainOnlyBPETokenizer,
    evaluate_tokenizer,
)
from .trainer import StageCTrainer, StreamBatchScheduler


def _read_sequences(paths: Iterable[Path], *, max_bases: int | None = None) -> tuple[str, ...]:
    sequences: list[str] = []
    bases = 0
    for path in paths:
        for _, sequence in iter_fasta(path):
            if max_bases is not None and bases >= max_bases:
                return tuple(sequences)
            value = sequence if max_bases is None else sequence[: max_bases - bases]
            if value:
                sequences.append(value)
                bases += len(value)
    if not sequences:
        raise ValueError("no FASTA sequences were found")
    return tuple(sequences)


def _unavailable_metric(name: str, family: str, reason: str) -> TokenizerMetrics:
    return TokenizerMetrics(
        name=name,
        family=family,
        sequence_count=0,
        base_count=0,
        token_count=0,
        bases_per_token=0.0,
        vocabulary_used=0,
        unknown_rate=1.0,
        rare_token_fraction=0.0,
        token_entropy_bits=0.0,
        vocabulary_utilization=0.0,
        mean_token_length=0.0,
        maximum_token_length=0,
        reverse_complement_token_ratio=0.0,
        start_offset_compression_cv=0.0,
        estimated_bytes_per_base=0.0,
        tokenization_bases_per_second=0.0,
        gc_bin_bases_per_token={},
        round_trip_passed=False,
        eligible=False,
        verification=f"unavailable:{reason}",
    )


def _candidate_tokenizers(
    train_sequences: Sequence[str],
    output_dir: Path,
    *,
    dnabert2_path: Path | None,
    skip_train_bpe: bool,
) -> tuple[list[GenomeTokenizer], list[TokenizerMetrics]]:
    tokenizers: list[GenomeTokenizer] = [
        SeqTrainerBaseTokenizer(),
        Evo2CharTokenizer(),
        SixMerTokenizer(),
    ]
    unavailable: list[TokenizerMetrics] = []
    if dnabert2_path is not None:
        try:
            tokenizers.append(HuggingFaceBPETokenizer(dnabert2_path))
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            unavailable.append(_unavailable_metric("dnabert2_official_bpe", "multi_nucleotide", str(error)))
    else:
        unavailable.append(
            _unavailable_metric(
                "dnabert2_official_bpe",
                "multi_nucleotide",
                "provide --dnabert2-path containing frozen tokenizer files",
            )
        )
    if not skip_train_bpe:
        try:
            tokenizers.append(
                TrainOnlyBPETokenizer.train(
                    train_sequences,
                    output_dir / "bacterial_train_only_bpe",
                    vocab_size=4096,
                )
            )
        except RuntimeError as error:
            unavailable.append(_unavailable_metric("bacterial_train_only_bpe", "multi_nucleotide", str(error)))
    return tokenizers, unavailable


def tokenizer_study_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Stage C single- and multi-nucleotide tokenizers")
    parser.add_argument("--train-fasta", type=Path, action="append", required=True)
    parser.add_argument("--validation-fasta", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dnabert2-path", type=Path)
    parser.add_argument("--max-train-bases", type=int, default=1_000_000)
    parser.add_argument("--max-validation-bases", type=int, default=250_000)
    parser.add_argument("--skip-train-bpe", action="store_true")
    args = parser.parse_args(argv)
    train = _read_sequences(args.train_fasta, max_bases=args.max_train_bases)
    validation = _read_sequences(args.validation_fasta, max_bases=args.max_validation_bases)
    candidates, unavailable = _candidate_tokenizers(
        train,
        args.output_dir,
        dnabert2_path=args.dnabert2_path,
        skip_train_bpe=args.skip_train_bpe,
    )
    metrics = [evaluate_tokenizer(tokenizer, validation) for tokenizer in candidates] + unavailable
    paths = write_tokenizer_report(metrics, args.output_dir)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2, sort_keys=True))
    return 0


def _streams_for_sequences(
    sequences: Sequence[str],
    tokenizer: GenomeTokenizer,
    *,
    split: str,
) -> dict[str, tuple[object, ...]]:
    streams = {}
    for index, sequence in enumerate(sequences):
        segments = build_stream_segments(
            sequence=sequence,
            accession=f"local-{split}-{index:06d}",
            contig_id="contig",
            split=split,
            clade_group=f"local-{split}-{index:06d}",
            tokenizer=tokenizer,
        )
        if segments:
            streams[segments[0].stream_id] = segments
    if not streams:
        raise ValueError(f"{tokenizer.spec.name} produced no trainable streams")
    return streams


def _evaluate_model(
    model: StageCPaperMACForCausalLM,
    streams: dict[str, tuple[object, ...]],
) -> tuple[float, int]:
    model.eval()
    total_nll = 0.0
    total_bases = 0
    for stream_id in sorted(streams):
        states = model.initial_states(stream_id)
        for segment in streams[stream_id]:
            tensors = StageCTrainer._batch_tensors((segment,), torch.device("cpu"))
            with torch.enable_grad():
                output = model.forward_segment((states,), **tensors)
            assert output.loss_sum is not None
            total_nll += float(output.loss_sum.detach())
            total_bases += output.valid_bases
            states = detach_stream_states(output.states[0])
    if not total_bases:
        raise ValueError("validation produced no bases")
    return total_nll / (total_bases * __import__("math").log(2.0)), total_bases


def _select_tokenizer(
    intrinsic: Sequence[TokenizerMetrics],
    runs: Sequence[dict[str, object]],
    tokenizers: Sequence[GenomeTokenizer],
    *,
    minimum_improvement: float = 0.01,
) -> dict[str, object]:
    intrinsic_by_name = {item.name: item for item in intrinsic}
    tokenizer_by_name = {item.spec.name: item for item in tokenizers}
    equal_base_runs = [item for item in runs if item["regime"] == "equal_bases"]
    base_run = next(
        item for item in equal_base_runs if item["tokenizer"] == "seqtrainer_base_v1"
    )
    eligible_runs = [
        item
        for item in equal_base_runs
        if intrinsic_by_name[str(item["tokenizer"])].eligible
    ]
    best = min(
        eligible_runs,
        key=lambda item: (
            float(item["validation_bpb"]),
            intrinsic_by_name[str(item["tokenizer"])].estimated_bytes_per_base,
            tokenizer_by_name[str(item["tokenizer"])].spec.vocab_size,
            str(item["tokenizer"]),
        ),
    )
    base_bpb = float(base_run["validation_bpb"])
    best_improvement = base_bpb - float(best["validation_bpb"])
    selected = (
        str(best["tokenizer"])
        if best["tokenizer"] == "seqtrainer_base_v1" or best_improvement >= minimum_improvement
        else "seqtrainer_base_v1"
    )
    return {
        "format_version": 1,
        "selected_tokenizer": selected,
        "decision_regime": "equal_bases",
        "metric": "held_out_bits_per_base",
        "minimum_bpb_improvement": minimum_improvement,
        "base_tokenizer": "seqtrainer_base_v1",
        "base_validation_bpb": base_bpb,
        "best_eligible_candidate": str(best["tokenizer"]),
        "best_candidate_validation_bpb": float(best["validation_bpb"]),
        "best_candidate_improvement_bpb": best_improvement,
        "fallback_applied": selected != str(best["tokenizer"]),
        "eligible_tokenizers": sorted(str(item["tokenizer"]) for item in eligible_runs),
        "selected_tokenizer_spec": tokenizer_by_name[selected].spec.to_dict(),
        "rule": (
            "Choose the lowest equal-base held-out BPB among intrinsically eligible tokenizers; "
            "a non-base tokenizer must improve by at least 0.01 BPB or fall back to "
            "seqtrainer_base_v1. Exact BPB ties use lower estimated storage, then smaller "
            "vocabulary, then lexical name."
        ),
    }


def cpu_pilot_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local Stage C basal capability and tokenizer LM study")
    parser.add_argument("--train-fasta", type=Path, action="append", required=True)
    parser.add_argument("--validation-fasta", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dnabert2-path", type=Path)
    parser.add_argument("--max-train-bases", type=int, default=100_000)
    parser.add_argument("--max-validation-bases", type=int, default=25_000)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--equal-base-budget",
        type=int,
        default=4_096,
        help="Raw valid bases per tokenizer in the matched equal-base regime",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=2, choices=(1, 2, 3, 4))
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--skip-train-bpe", action="store_true")
    args = parser.parse_args(argv)
    torch.manual_seed(20260741)
    train = _read_sequences(args.train_fasta, max_bases=args.max_train_bases)
    validation = _read_sequences(args.validation_fasta, max_bases=args.max_validation_bases)
    baselines = run_statistical_baselines(train, validation)
    tokenizers, unavailable = _candidate_tokenizers(
        train,
        args.output_dir,
        dnabert2_path=args.dnabert2_path,
        skip_train_bpe=args.skip_train_bpe,
    )
    intrinsic = [evaluate_tokenizer(tokenizer, validation) for tokenizer in tokenizers] + unavailable
    write_tokenizer_report(intrinsic, args.output_dir)
    runs: list[dict[str, object]] = []
    for tokenizer_index, tokenizer in enumerate(tokenizers):
        train_streams = _streams_for_sequences(train, tokenizer, split="train")
        validation_streams = _streams_for_sequences(validation, tokenizer, split="val")
        for regime in ("equal_steps", "equal_bases"):
            torch.manual_seed(20260741 + tokenizer_index)
            config = StageCModelConfig.cpu_basal(
                vocab_size=tokenizer.spec.vocab_size,
                pad_token_id=tokenizer.spec.pad_token_id,
                tokenizer_name=tokenizer.spec.name,
                tokenizer_checksum=tokenizer.spec.checksum,
                gradient_horizon=args.horizon,
            )
            model = StageCPaperMACForCausalLM(config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
            scheduler = StreamBatchScheduler(
                train_streams,
                batch_size=args.batch_size,
                seed=20260741,
            )
            trainer = StageCTrainer(model, optimizer)
            train_kwargs = (
                {"max_optimizer_steps": args.steps}
                if regime == "equal_steps"
                else {"max_valid_bases": args.equal_base_budget}
            )
            history = trainer.train(scheduler, **train_kwargs)
            run_dir = args.output_dir / tokenizer.spec.name / regime
            write_training_history(history, run_dir)
            validation_bpb, validation_bases = _evaluate_model(model, validation_streams)
            runs.append(
                {
                    "tokenizer": tokenizer.spec.name,
                    "family": tokenizer.spec.family,
                    "regime": regime,
                    "validation_bpb": validation_bpb,
                    "validation_bases": validation_bases,
                    "parameter_count": model.count_parameters(),
                    "train_steps": trainer.optimizer_step,
                    "train_bases": trainer.processed_bases,
                    "requested_steps": args.steps if regime == "equal_steps" else None,
                    "requested_base_budget": (
                        args.equal_base_budget if regime == "equal_bases" else None
                    ),
                }
            )
    selection = _select_tokenizer(intrinsic, runs, tokenizers)
    paths = write_cpu_pilot_report(
        baselines=baselines,
        tokenizer_runs=runs,
        selection=selection,
        output_dir=args.output_dir,
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2, sort_keys=True))
    return 0


def hardware_preflight_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assert Colab GPU identity before Stage C spending")
    parser.add_argument("--require", choices=("T4", "A100"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if available else "unavailable"
    passed = available and args.require.upper() in device_name.upper()
    payload = {
        "format_version": 1,
        "required": args.require,
        "passed": passed,
        "cuda_available": available,
        "device_name": device_name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(f"required {args.require} GPU, found {device_name}")
    return 0
