"""Lossless genomic tokenizers and comparable intrinsic measurements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Iterable, Mapping, Sequence


SUPPORTED_DNA = frozenset("ACGTN")


def normalize_dna(sequence: str) -> str:
    """Normalize case and map unsupported IUPAC/input symbols to ``N``."""

    return "".join(base if base in SUPPORTED_DNA else "N" for base in str(sequence).upper())


@dataclass(frozen=True)
class EncodedDNA:
    """Token IDs together with half-open source-base spans."""

    ids: tuple[int, ...]
    base_spans: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if len(self.ids) != len(self.base_spans):
            raise ValueError("every token must have one base span")
        previous_end = 0
        for start, end in self.base_spans:
            if start != previous_end or end <= start:
                raise ValueError("token spans must form a contiguous non-empty partition")
            previous_end = end

    @property
    def base_count(self) -> int:
        return self.base_spans[-1][1] if self.base_spans else 0


@dataclass(frozen=True)
class TokenizerSpec:
    """Checkpoint-safe public tokenizer identity."""

    name: str
    family: str
    vocab_size: int
    pad_token_id: int
    unk_token_id: int
    source: str
    checksum: str
    unk_represents_n: bool = False
    verification: str = "native"
    format_version: int = 1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class GenomeTokenizer(ABC):
    """Minimal tokenizer contract needed by ordered Stage C streams."""

    @property
    @abstractmethod
    def spec(self) -> TokenizerSpec: ...

    @abstractmethod
    def encode(self, sequence: str) -> EncodedDNA: ...

    @abstractmethod
    def decode(self, token_ids: Sequence[int]) -> str: ...


def _checksum(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class SeqTrainerBaseTokenizer(GenomeTokenizer):
    """The compact, literal next-base tokenizer already used by SeqTrainer."""

    TOKEN_TO_ID = {"N": 1, "A": 2, "C": 3, "G": 4, "T": 5}
    ID_TO_TOKEN = {value: key for key, value in TOKEN_TO_ID.items()}

    def __init__(self) -> None:
        contract = {"PAD": 0, **self.TOKEN_TO_ID}
        self._spec = TokenizerSpec(
            name="seqtrainer_base_v1",
            family="single_nucleotide",
            vocab_size=6,
            pad_token_id=0,
            unk_token_id=1,
            source="SeqTrainer",
            checksum=_checksum(contract),
            unk_represents_n=True,
        )

    @property
    def spec(self) -> TokenizerSpec:
        return self._spec

    def encode(self, sequence: str) -> EncodedDNA:
        normalized = normalize_dna(sequence)
        return EncodedDNA(
            ids=tuple(self.TOKEN_TO_ID[base] for base in normalized),
            base_spans=tuple((index, index + 1) for index in range(len(normalized))),
        )

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(self.ID_TO_TOKEN[int(token)] for token in token_ids if int(token) != 0)


class Evo2CharTokenizer(GenomeTokenizer):
    """Single-character Evo 2 adapter with optional official Vortex parity.

    Evo 2 constructs ``vortex.model.tokenizer.CharLevelTokenizer(512)``.  When
    Vortex is installed this adapter delegates to it.  The dependency-free
    fallback uses ASCII byte IDs and is explicitly marked unverified so it
    cannot silently win the Stage C selection gate.
    """

    def __init__(self, *, require_official: bool = False) -> None:
        self._official = None
        try:
            from vortex.model.tokenizer import CharLevelTokenizer  # type: ignore

            self._official = CharLevelTokenizer(512)
        except ImportError:
            if require_official:
                raise RuntimeError("the official Vortex CharLevelTokenizer is unavailable")
        verification = "official_vortex" if self._official is not None else "ascii_fallback_unverified"
        self._spec = TokenizerSpec(
            name="evo2_charlevel_512",
            family="single_nucleotide",
            vocab_size=512,
            pad_token_id=0,
            unk_token_id=ord("N"),
            source="ArcInstitute/evo2:vortex.CharLevelTokenizer(512)",
            checksum=_checksum({"type": "CharLevelTokenizer", "vocab_size": 512, "verification": verification}),
            unk_represents_n=True,
            verification=verification,
        )

    @property
    def spec(self) -> TokenizerSpec:
        return self._spec

    def encode(self, sequence: str) -> EncodedDNA:
        normalized = normalize_dna(sequence)
        ids = (
            tuple(int(value) for value in self._official.tokenize(normalized))
            if self._official is not None
            else tuple(ord(base) for base in normalized)
        )
        if len(ids) != len(normalized):
            raise ValueError("Evo 2 character tokenizer must emit one token per nucleotide")
        return EncodedDNA(ids=ids, base_spans=tuple((i, i + 1) for i in range(len(ids))))

    def decode(self, token_ids: Sequence[int]) -> str:
        values = [int(token) for token in token_ids if int(token) != self.spec.pad_token_id]
        if self._official is not None and hasattr(self._official, "detokenize"):
            decoded = self._official.detokenize(values)
            return decoded.decode("ascii") if isinstance(decoded, bytes) else str(decoded)
        return "".join(chr(token) for token in values)


class SixMerTokenizer(GenomeTokenizer):
    """Non-overlapping canonical 6-mers with lossless single-base fallback."""

    BASE_TO_VALUE = {"A": 0, "C": 1, "G": 2, "T": 3}
    VALUE_TO_BASE = "ACGT"
    SINGLE_TO_ID = {"N": 2, "A": 3, "C": 4, "G": 5, "T": 6}
    ID_TO_SINGLE = {value: key for key, value in SINGLE_TO_ID.items()}
    KMER_OFFSET = 7

    def __init__(self) -> None:
        self._spec = TokenizerSpec(
            name="nonoverlap_6mer_v1",
            family="multi_nucleotide",
            vocab_size=self.KMER_OFFSET + 4**6,
            pad_token_id=0,
            unk_token_id=1,
            source="SeqTrainer train-independent fixed vocabulary",
            checksum=_checksum({"k": 6, "alphabet": self.VALUE_TO_BASE, "offset": self.KMER_OFFSET}),
        )

    @property
    def spec(self) -> TokenizerSpec:
        return self._spec

    @classmethod
    def _kmer_id(cls, kmer: str) -> int:
        value = 0
        for base in kmer:
            value = value * 4 + cls.BASE_TO_VALUE[base]
        return cls.KMER_OFFSET + value

    @classmethod
    def _decode_kmer(cls, token_id: int) -> str:
        value = token_id - cls.KMER_OFFSET
        bases = ["A"] * 6
        for index in range(5, -1, -1):
            bases[index] = cls.VALUE_TO_BASE[value % 4]
            value //= 4
        return "".join(bases)

    def encode(self, sequence: str) -> EncodedDNA:
        normalized = normalize_dna(sequence)
        ids: list[int] = []
        spans: list[tuple[int, int]] = []
        position = 0
        while position < len(normalized):
            chunk = normalized[position : position + 6]
            if len(chunk) == 6 and set(chunk).issubset(self.BASE_TO_VALUE):
                ids.append(self._kmer_id(chunk))
                spans.append((position, position + 6))
                position += 6
            else:
                ids.append(self.SINGLE_TO_ID[normalized[position]])
                spans.append((position, position + 1))
                position += 1
        return EncodedDNA(ids=tuple(ids), base_spans=tuple(spans))

    def decode(self, token_ids: Sequence[int]) -> str:
        pieces: list[str] = []
        for raw in token_ids:
            token = int(raw)
            if token == self.spec.pad_token_id:
                continue
            if token in self.ID_TO_SINGLE:
                pieces.append(self.ID_TO_SINGLE[token])
            elif self.KMER_OFFSET <= token < self.spec.vocab_size:
                pieces.append(self._decode_kmer(token))
            else:
                pieces.append("N")
        return "".join(pieces)


class HuggingFaceBPETokenizer(GenomeTokenizer):
    """Local-files-only adapter for the frozen official DNABERT-2 tokenizer."""

    def __init__(self, path: str | Path, *, name: str = "dnabert2_official_bpe") -> None:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required for DNABERT-2 tokenization") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(source), local_files_only=True, trust_remote_code=True, use_fast=True
        )
        files = sorted(file for file in source.rglob("*") if file.is_file())
        digest = hashlib.sha256()
        for file in files:
            digest.update(str(file.relative_to(source)).encode("utf-8"))
            digest.update(file.read_bytes())
        pad = self._tokenizer.pad_token_id
        unk = self._tokenizer.unk_token_id
        self._spec = TokenizerSpec(
            name=name,
            family="multi_nucleotide",
            vocab_size=len(self._tokenizer),
            pad_token_id=0 if pad is None else int(pad),
            unk_token_id=0 if unk is None else int(unk),
            source=str(source),
            checksum=digest.hexdigest(),
            verification="frozen_local_artifact",
        )

    @property
    def spec(self) -> TokenizerSpec:
        return self._spec

    def encode(self, sequence: str) -> EncodedDNA:
        normalized = normalize_dna(sequence)
        result = self._tokenizer(
            normalized,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        spans = tuple((int(start), int(end)) for start, end in result["offset_mapping"])
        encoded = EncodedDNA(ids=tuple(int(value) for value in result["input_ids"]), base_spans=spans)
        if encoded.base_count != len(normalized):
            raise ValueError("DNABERT-2 offsets do not cover the complete normalized sequence")
        return encoded

    def decode(self, token_ids: Sequence[int]) -> str:
        return normalize_dna(self._tokenizer.decode(list(token_ids), skip_special_tokens=True).replace(" ", ""))


class TrainOnlyBPETokenizer(HuggingFaceBPETokenizer):
    """A BPE tokenizer trained solely from supplied Stage C training streams."""

    @classmethod
    def train(
        cls,
        sequences: Iterable[str],
        output_dir: str | Path,
        *,
        vocab_size: int = 4096,
    ) -> "TrainOnlyBPETokenizer":
        try:
            from tokenizers import Tokenizer, decoders, models, trainers
            from transformers import PreTrainedTokenizerFast
        except ImportError as exc:
            raise RuntimeError("tokenizers and transformers are required to train bacterial BPE") from exc
        if vocab_size < 16:
            raise ValueError("BPE vocabulary must contain at least 16 entries")
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.decoder = decoders.BPEDecoder()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[PAD]", "[UNK]"],
            initial_alphabet=list("ACGTN"),
            show_progress=False,
        )
        tokenizer.train_from_iterator((normalize_dna(value) for value in sequences), trainer=trainer)
        wrapped = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            pad_token="[PAD]",
            unk_token="[UNK]",
        )
        wrapped.save_pretrained(destination)
        return cls(destination, name="bacterial_train_only_bpe")


@dataclass(frozen=True)
class TokenizerMetrics:
    name: str
    family: str
    sequence_count: int
    base_count: int
    token_count: int
    bases_per_token: float
    vocabulary_used: int
    unknown_rate: float
    rare_token_fraction: float
    token_entropy_bits: float
    vocabulary_utilization: float
    mean_token_length: float
    maximum_token_length: int
    reverse_complement_token_ratio: float
    start_offset_compression_cv: float
    estimated_bytes_per_base: float
    tokenization_bases_per_second: float
    gc_bin_bases_per_token: Mapping[str, float]
    round_trip_passed: bool
    eligible: bool
    verification: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_tokenizer(tokenizer: GenomeTokenizer, sequences: Iterable[str]) -> TokenizerMetrics:
    """Compute reproducible intrinsic evidence without looking at model quality."""

    counts: Counter[int] = Counter()
    base_count = 0
    sequence_count = 0
    round_trip = True
    token_lengths: list[int] = []
    reverse_tokens = 0
    offset_rates: list[float] = []
    gc_totals = {"gc_0_40": [0, 0], "gc_40_60": [0, 0], "gc_60_100": [0, 0]}
    started = time.perf_counter()
    for sequence in sequences:
        normalized = normalize_dna(sequence)
        encoded = tokenizer.encode(normalized)
        decoded = tokenizer.decode(encoded.ids)
        sequence_count += 1
        base_count += len(normalized)
        counts.update(encoded.ids)
        token_lengths.extend(end - start for start, end in encoded.base_spans)
        reverse = normalized.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]
        reverse_tokens += len(tokenizer.encode(reverse).ids)
        for offset in range(min(6, len(normalized))):
            shifted = tokenizer.encode(normalized[offset:])
            if shifted.ids:
                offset_rates.append(shifted.base_count / len(shifted.ids))
        gc_fraction = (
            (normalized.count("G") + normalized.count("C")) / len(normalized)
            if normalized
            else 0.0
        )
        gc_key = "gc_0_40" if gc_fraction < 0.4 else "gc_60_100" if gc_fraction >= 0.6 else "gc_40_60"
        gc_totals[gc_key][0] += len(normalized)
        gc_totals[gc_key][1] += len(encoded.ids)
        round_trip = round_trip and decoded == normalized and encoded.base_count == len(normalized)
    token_count = sum(counts.values())
    unknown = 0 if tokenizer.spec.unk_represents_n else counts[tokenizer.spec.unk_token_id]
    rare = sum(count for count in counts.values() if count < 2)
    entropy = 0.0
    if token_count:
        entropy = -sum((count / token_count) * math.log2(count / token_count) for count in counts.values())
    eligible = (
        round_trip
        and token_count > 0
        and unknown == 0
        and "unverified" not in tokenizer.spec.verification
    )
    elapsed = time.perf_counter() - started
    offset_mean = statistics.fmean(offset_rates) if offset_rates else 0.0
    offset_cv = (
        statistics.pstdev(offset_rates) / offset_mean
        if len(offset_rates) > 1 and offset_mean
        else 0.0
    )
    return TokenizerMetrics(
        name=tokenizer.spec.name,
        family=tokenizer.spec.family,
        sequence_count=sequence_count,
        base_count=base_count,
        token_count=token_count,
        bases_per_token=base_count / token_count if token_count else 0.0,
        vocabulary_used=len(counts),
        unknown_rate=unknown / token_count if token_count else 0.0,
        rare_token_fraction=rare / token_count if token_count else 0.0,
        token_entropy_bits=entropy,
        vocabulary_utilization=len(counts) / tokenizer.spec.vocab_size,
        mean_token_length=statistics.fmean(token_lengths) if token_lengths else 0.0,
        maximum_token_length=max(token_lengths, default=0),
        reverse_complement_token_ratio=reverse_tokens / token_count if token_count else 0.0,
        start_offset_compression_cv=offset_cv,
        estimated_bytes_per_base=(token_count * 6) / base_count if base_count else 0.0,
        tokenization_bases_per_second=base_count / elapsed if elapsed else 0.0,
        gc_bin_bases_per_token={
            key: bases / tokens if tokens else 0.0
            for key, (bases, tokens) in gc_totals.items()
        },
        round_trip_passed=round_trip,
        eligible=eligible,
        verification=tokenizer.spec.verification,
    )
