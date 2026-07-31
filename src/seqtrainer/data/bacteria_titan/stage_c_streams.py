"""Clade-safe splits and ordered token streams for Stage C paper-MAC."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .token_shards import iter_fasta

if TYPE_CHECKING:
    from seqtrainer.torch.titans_paper_mac_stage_c.tokenizers import GenomeTokenizer


STAGE_C_SPLIT_FRACTIONS = {"train": 0.90, "val": 0.05, "test": 0.05}
ACCESSION_PATTERN = re.compile(r"(?:GCF|GCA)_\d+\.\d+")


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        self.parent[larger] = smaller


def _accession_from_path(value: object) -> str:
    match = ACCESSION_PATTERN.search(str(value))
    if match is None:
        raise ValueError(f"cannot recover accession from skani path: {value!r}")
    return match.group(0)


def normalize_stage_c_source_manifest(accessions: pd.DataFrame) -> pd.DataFrame:
    """Adapt supported source-manifest schemas to the Stage C data contract.

    The shared E. coli source dataset is produced from GTDB metadata.  It
    identifies its NCBI assembly as ``assembly_accession`` and labels the
    sampling tier ``ecoli_related_scope``.  Stage C FASTA shards and Skani use
    the assembly accession in their paths, so it must be the canonical ID used
    for clustering, splitting, and shard regeneration.
    """

    frame = accessions.copy()
    accession_column = "assembly_accession" if "assembly_accession" in frame else "accession"
    if accession_column not in frame:
        raise ValueError("source manifest requires accession or assembly_accession")
    if "accession" in frame and accession_column != "accession":
        frame["source_accession"] = frame["accession"]
    canonical = frame[accession_column].map(_accession_from_path)
    if canonical.duplicated().any():
        duplicates = canonical.loc[canonical.duplicated(keep=False)].unique()[:5]
        raise ValueError(f"source manifest has duplicate assembly accessions: {duplicates.tolist()}")
    frame["accession"] = canonical

    if "scope" not in frame:
        if "ecoli_related_scope" not in frame:
            raise ValueError(
                "source manifest requires scope or ecoli_related_scope for Stage C stratification"
            )
        frame["scope"] = frame["ecoli_related_scope"]
    frame["scope"] = frame["scope"].astype("string").str.strip()
    if frame["scope"].isna().any() or frame["scope"].eq("").any():
        raise ValueError("source manifest has missing Stage C scope values")
    return frame


def cluster_ani_pairs(
    accessions: Iterable[str],
    pairs: pd.DataFrame,
    *,
    threshold: float = 99.0,
) -> pd.DataFrame:
    """Create deterministic single-linkage clusters from skani pair evidence."""

    values = sorted(set(map(str, accessions)))
    if not values:
        raise ValueError("ANI clustering requires accessions")
    required = {"Ref_file", "Query_file", "ANI"}
    missing = required - set(pairs)
    if missing:
        raise ValueError(f"skani pair table is missing columns: {sorted(missing)}")
    disjoint = _DisjointSet(values)
    evidence_rows = 0
    for row in pairs.itertuples(index=False):
        ani = float(getattr(row, "ANI"))
        if ani < threshold:
            continue
        left = _accession_from_path(getattr(row, "Ref_file"))
        right = _accession_from_path(getattr(row, "Query_file"))
        if left not in disjoint.parent or right not in disjoint.parent:
            raise ValueError("skani output contains an accession outside the selected manifest")
        disjoint.union(left, right)
        evidence_rows += 1
    roots = {accession: disjoint.find(accession) for accession in values}
    ordered_roots = {root: index for index, root in enumerate(sorted(set(roots.values())))}
    return pd.DataFrame(
        {
            "accession": values,
            "ani_cluster_99": [f"ani99_{ordered_roots[roots[value]]:06d}" for value in values],
            "ani_threshold": threshold,
            "ani_linkage": "single",
            "skani_edges_at_or_above_threshold": evidence_rows,
        }
    )


def run_skani_triangle(
    fasta_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    threads: int = 4,
    command: str = "skani",
) -> Path:
    """Capture extended skani all-to-all evidence with sensitivity below 99%.

    Screening is set to 95 rather than 99 so the approximate screen does not
    silently discard pairs near the clustering boundary.
    """

    paths = [Path(path) for path in fasta_paths]
    if not paths or any(not path.exists() for path in paths):
        raise ValueError("every skani FASTA input must exist")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    input_list = destination.with_suffix(destination.suffix + ".inputs.txt")
    input_list.write_text(
        "".join(f"{path.resolve()}\n" for path in paths),
        encoding="utf-8",
    )
    partial = destination.with_suffix(destination.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        subprocess.run(
            [
                command,
                "triangle",
                "-l",
                str(input_list),
                "-s",
                "95",
                "-E",
                "-t",
                str(threads),
            ],
            stdout=handle,
            check=True,
        )
    partial.replace(destination)
    return destination


def assign_hybrid_clade_groups(
    accessions: pd.DataFrame,
    ani_membership: pd.DataFrame,
) -> pd.DataFrame:
    """Use ANI99 within E. coli and GTDB species clusters elsewhere."""

    if "scope" not in accessions or "accession" not in accessions:
        raise ValueError("hybrid grouping requires accession and scope")
    if not {"accession", "ani_cluster_99"}.issubset(ani_membership):
        raise ValueError("ANI membership requires accession and ani_cluster_99")
    frame = accessions.merge(
        ani_membership[["accession", "ani_cluster_99"]],
        on="accession",
        how="left",
        validate="one_to_one",
    )
    dense = frame["scope"].eq("ecoli_species")
    if frame.loc[dense, "ani_cluster_99"].isna().any():
        raise ValueError("every E. coli accession requires ANI99 membership")
    representative_column = next(
        (
            column
            for column in ("gtdb_genome_representative", "species_cluster", "species")
            if column in frame
        ),
        None,
    )
    if representative_column is None or frame.loc[~dense, representative_column].isna().any():
        raise ValueError("non-E. coli accessions require a GTDB species-cluster identifier")
    frame["clade_group"] = ""
    frame.loc[dense, "clade_group"] = "ani99:" + frame.loc[dense, "ani_cluster_99"].astype(str)
    frame.loc[~dense, "clade_group"] = (
        "gtdb_species:" + frame.loc[~dense, representative_column].astype(str)
    )
    return frame


def _group_key(group: str, seed: int) -> str:
    return hashlib.sha256(f"stage-c:{seed}:{group}".encode("utf-8")).hexdigest()


def split_clade_groups(
    accessions: pd.DataFrame,
    *,
    group_column: str = "clade_group",
    weight_column: str = "genome_size",
    stratify_column: str = "scope",
    fractions: Mapping[str, float] = STAGE_C_SPLIT_FRACTIONS,
    seed: int = 17,
) -> pd.DataFrame:
    """Assign complete clade/ANI groups to deterministic weighted splits."""

    if set(fractions) != {"train", "val", "test"} or abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("fractions must define train/val/test and sum to one")
    required = {"accession", group_column, weight_column}
    missing = required - set(accessions)
    if missing:
        raise ValueError(f"clade split input is missing columns: {sorted(missing)}")
    if accessions["accession"].duplicated().any() or accessions[group_column].isna().any():
        raise ValueError("accessions must be unique and every row must have a clade group")
    frame = accessions.copy()
    frame["split"] = ""
    strata = frame.groupby(stratify_column, dropna=False, sort=True) if stratify_column in frame else [("all", frame)]
    for _, stratum in strata:
        groups = (
            stratum.groupby(group_column, sort=True)[weight_column]
            .sum()
            .sort_index()
        )
        ordered = sorted(groups.index.astype(str), key=lambda group: _group_key(group, seed))
        total = float(groups.sum())
        test_target = total * fractions["test"]
        val_target = total * fractions["val"]
        assigned = {"test": 0.0, "val": 0.0, "train": 0.0}
        assignments: dict[str, str] = {}
        for group in ordered:
            weight = float(groups.loc[group])
            if assigned["test"] < test_target:
                split = "test"
            elif assigned["val"] < val_target:
                split = "val"
            else:
                split = "train"
            assignments[group] = split
            assigned[split] += weight
        frame.loc[stratum.index, "split"] = stratum[group_column].astype(str).map(assignments)
    assert_no_clade_leakage(frame, group_column=group_column)
    return frame.sort_values(["split", group_column, "accession"], kind="stable").reset_index(drop=True)


def assert_no_clade_leakage(frame: pd.DataFrame, *, group_column: str = "clade_group") -> None:
    """Reject any accession or clade group assigned to multiple splits."""

    required = {"accession", group_column, "split"}
    if required - set(frame):
        raise ValueError("clade leakage audit is missing required columns")
    if not set(frame["split"]).issubset({"train", "val", "test"}):
        raise ValueError("every row must have a recognized split")
    accession_counts = frame.groupby("accession")["split"].nunique()
    group_counts = frame.groupby(group_column)["split"].nunique()
    if (accession_counts > 1).any():
        raise ValueError("accession leakage detected across Stage C splits")
    if (group_counts > 1).any():
        raise ValueError("clade group leakage detected across Stage C splits")


@dataclass(frozen=True)
class StreamSegment:
    """One ordered 32-token next-token example from exactly one contig."""

    stream_id: str
    accession: str
    contig_id: str
    split: str
    clade_group: str
    gc_fraction: float
    segment_index: int
    token_offset: int
    base_start: int
    base_end: int
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    valid_mask: tuple[bool, ...]
    loss_mask: tuple[bool, ...]
    represented_base_counts: tuple[int, ...]
    start_of_stream: bool
    end_of_stream: bool

    def __post_init__(self) -> None:
        widths = {
            len(self.input_ids),
            len(self.labels),
            len(self.valid_mask),
            len(self.loss_mask),
            len(self.represented_base_counts),
        }
        if widths != {32}:
            raise ValueError("Stage C stream segments must have width 32")
        if not self.stream_id or self.segment_index < 0 or self.token_offset < 0:
            raise ValueError("stream identity and offsets must be valid")
        if not 0.0 <= self.gc_fraction <= 1.0:
            raise ValueError("gc_fraction must be between zero and one")
        if self.base_end <= self.base_start:
            raise ValueError("segment must represent a non-empty base interval")
        for valid, loss, bases in zip(self.valid_mask, self.loss_mask, self.represented_base_counts):
            if loss and (not valid or bases <= 0):
                raise ValueError("loss positions must be valid and represent bases")
            if not loss and bases:
                raise ValueError("masked loss positions cannot count bases")

    @property
    def valid_tokens(self) -> int:
        return sum(self.loss_mask)

    @property
    def valid_bases(self) -> int:
        return sum(self.represented_base_counts)

    def metadata_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in ("input_ids", "labels", "valid_mask", "loss_mask", "represented_base_counts"):
            payload.pop(key)
        payload["valid_tokens"] = self.valid_tokens
        payload["valid_bases"] = self.valid_bases
        return payload


def build_stream_segments(
    *,
    sequence: str,
    accession: str,
    contig_id: str,
    split: str,
    clade_group: str,
    tokenizer: GenomeTokenizer,
    segment_length: int = 32,
) -> tuple[StreamSegment, ...]:
    """Tokenize one complete contig and preserve causal cross-segment labels."""

    if segment_length != 32:
        raise ValueError("Stage C paper-MAC segments must contain 32 tokens")
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    encoded = tokenizer.encode(sequence)
    if len(encoded.ids) < 2:
        return ()
    stream_id = f"{accession}:{contig_id}"
    normalized = str(sequence).upper()
    gc_fraction = (
        (normalized.count("G") + normalized.count("C")) / len(normalized)
        if normalized
        else 0.0
    )
    examples = len(encoded.ids) - 1
    segments: list[StreamSegment] = []
    for segment_index, offset in enumerate(range(0, examples, segment_length)):
        count = min(segment_length, examples - offset)
        inputs = list(encoded.ids[offset : offset + count])
        labels = list(encoded.ids[offset + 1 : offset + count + 1])
        base_counts = [encoded.base_spans[offset + position + 1][1] - encoded.base_spans[offset + position + 1][0] for position in range(count)]
        base_start = encoded.base_spans[offset][0]
        base_end = encoded.base_spans[offset + count][1]
        padding = segment_length - count
        inputs.extend([tokenizer.spec.pad_token_id] * padding)
        labels.extend([tokenizer.spec.pad_token_id] * padding)
        base_counts.extend([0] * padding)
        segments.append(
            StreamSegment(
                stream_id=stream_id,
                accession=accession,
                contig_id=contig_id,
                split=split,
                clade_group=clade_group,
                gc_fraction=gc_fraction,
                segment_index=segment_index,
                token_offset=offset,
                base_start=base_start,
                base_end=base_end,
                input_ids=tuple(inputs),
                labels=tuple(labels),
                valid_mask=tuple([True] * count + [False] * padding),
                loss_mask=tuple([True] * count + [False] * padding),
                represented_base_counts=tuple(base_counts),
                start_of_stream=segment_index == 0,
                end_of_stream=offset + count == examples,
            )
        )
    return tuple(segments)


def materialize_stream_dataset(
    records: pd.DataFrame | Iterable[Mapping[str, object]],
    tokenizer: GenomeTokenizer,
    output_dir: str | Path,
    *,
    segments_per_shard: int = 100_000,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, Path]:
    """Write deterministic sharded tensors and JSONL metadata for ordered contigs."""

    required = {"accession", "contig_id", "sequence", "split", "clade_group"}
    if isinstance(records, pd.DataFrame):
        missing = required - set(records)
        if missing:
            raise ValueError(f"stream records are missing columns: {sorted(missing)}")
        ordered_records: Iterable[Mapping[str, object]] = (
            row._asdict()
            for row in records.sort_values(
                ["split", "accession", "contig_id"], kind="stable"
            ).itertuples(index=False)
        )
    else:
        ordered_records = records
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if segments_per_shard <= 0:
        raise ValueError("segments_per_shard must be positive")
    metadata = destination / "stream_segments.jsonl"
    buffer: list[StreamSegment] = []
    shard_rows: list[dict[str, object]] = []
    stream_ids: set[str] = set()
    segment_count = 0

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        shard_index = len(shard_rows)
        path = destination / f"stream_segments_{shard_index:05d}.npz"
        np.savez(
            path,
            input_ids=np.asarray([segment.input_ids for segment in buffer], dtype=np.int32),
            labels=np.asarray([segment.labels for segment in buffer], dtype=np.int32),
            valid_mask=np.asarray([segment.valid_mask for segment in buffer], dtype=np.bool_),
            loss_mask=np.asarray([segment.loss_mask for segment in buffer], dtype=np.bool_),
            represented_base_counts=np.asarray(
                [segment.represented_base_counts for segment in buffer], dtype=np.int32
            ),
        )
        shard_rows.append(
            {
                "path": path.name,
                "segments": len(buffer),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        buffer = []

    with metadata.open("w", encoding="utf-8") as metadata_handle:
        for row in ordered_records:
            missing = required - set(row)
            if missing:
                raise ValueError(f"stream record is missing columns: {sorted(missing)}")
            for segment in build_stream_segments(
                sequence=str(row["sequence"]),
                accession=str(row["accession"]),
                contig_id=str(row["contig_id"]),
                split=str(row["split"]),
                clade_group=str(row["clade_group"]),
                tokenizer=tokenizer,
            ):
                metadata_handle.write(json.dumps(segment.metadata_dict(), sort_keys=True) + "\n")
                stream_ids.add(segment.stream_id)
                segment_count += 1
                buffer.append(segment)
                if len(buffer) == segments_per_shard:
                    flush()
    flush()
    tokenizer_path = destination / "tokenizer_spec.json"
    tokenizer_path.write_text(json.dumps(tokenizer.spec.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = destination / "stream_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format_version": 1,
                "segment_length": 32,
                "segments": segment_count,
                "streams": len(stream_ids),
                "tokenizer": tokenizer.spec.to_dict(),
                "shards": shard_rows,
                "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
                "provenance": dict(provenance or {}),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    paths = {"metadata": metadata, "tokenizer": tokenizer_path, "manifest": manifest}
    if shard_rows:
        paths["tensors"] = destination / str(shard_rows[0]["path"])
    return paths


def iter_stage_c_fasta_records(
    fasta_manifest: pd.DataFrame,
    accession_manifest: pd.DataFrame,
) -> Iterable[dict[str, object]]:
    """Stream contigs from deterministic FASTA shards with clade metadata."""

    required_fasta = {"split", "path"}
    required_accession = {"accession", "split", "clade_group"}
    if required_fasta - set(fasta_manifest) or required_accession - set(accession_manifest):
        raise ValueError("FASTA/accession manifests lack Stage C fields")
    lookup = accession_manifest.set_index("accession")[["split", "clade_group"]].to_dict("index")
    seen_contigs: dict[tuple[str, str], int] = {}
    for row in fasta_manifest.sort_values(["split", "path"], kind="stable").itertuples(index=False):
        for header, sequence in iter_fasta(str(row.path)):
            if not header.startswith("accession=") or "|" not in header:
                raise ValueError("Stage C FASTA headers must preserve accession=<id>|<contig>")
            prefix, raw_contig = header.split("|", 1)
            accession = prefix.split("=", 1)[1]
            if accession not in lookup:
                raise ValueError(f"FASTA accession {accession!r} is absent from the manifest")
            metadata = lookup[accession]
            if str(row.split) != str(metadata["split"]):
                raise ValueError("FASTA shard split disagrees with clade-safe accession split")
            base_id = raw_contig.split()[0] or "contig"
            key = (accession, base_id)
            occurrence = seen_contigs.get(key, 0)
            seen_contigs[key] = occurrence + 1
            contig_id = base_id if occurrence == 0 else f"{base_id}#{occurrence}"
            yield {
                "accession": accession,
                "contig_id": contig_id,
                "sequence": sequence,
                "split": str(metadata["split"]),
                "clade_group": str(metadata["clade_group"]),
            }


def load_stream_dataset(
    dataset_dir: str | Path,
    *,
    split: str | None = None,
    verify_checksums: bool = True,
) -> tuple[StreamSegment, ...]:
    """Load and optionally checksum-verify a materialized Stage C dataset."""

    root = Path(dataset_dir)
    manifest = json.loads((root / "stream_manifest.json").read_text(encoding="utf-8"))
    metadata_path = root / "stream_segments.jsonl"
    if verify_checksums and hashlib.sha256(metadata_path.read_bytes()).hexdigest() != manifest["metadata_sha256"]:
        raise ValueError("Stage C stream metadata checksum mismatch")
    metadata_rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    segments: list[StreamSegment] = []
    metadata_offset = 0
    for shard in manifest["shards"]:
        path = root / shard["path"]
        if verify_checksums and hashlib.sha256(path.read_bytes()).hexdigest() != shard["sha256"]:
            raise ValueError(f"Stage C tensor checksum mismatch: {path.name}")
        arrays = np.load(path, mmap_mode="r", allow_pickle=False)
        count = int(shard["segments"])
        for index in range(count):
            row = dict(metadata_rows[metadata_offset + index])
            if split is not None and row["split"] != split:
                continue
            row.pop("valid_tokens", None)
            row.pop("valid_bases", None)
            segments.append(
                StreamSegment(
                    **row,
                    input_ids=tuple(int(value) for value in arrays["input_ids"][index]),
                    labels=tuple(int(value) for value in arrays["labels"][index]),
                    valid_mask=tuple(bool(value) for value in arrays["valid_mask"][index]),
                    loss_mask=tuple(bool(value) for value in arrays["loss_mask"][index]),
                    represented_base_counts=tuple(
                        int(value) for value in arrays["represented_base_counts"][index]
                    ),
                )
            )
        metadata_offset += count
    if metadata_offset != len(metadata_rows):
        raise ValueError("Stage C tensor and metadata row counts disagree")
    return tuple(segments)


@dataclass(frozen=True)
class TokenStreamIndex:
    """Compact contig-level index into memory-mapped production token shards."""

    stream_id: str
    accession: str
    contig_id: str
    split: str
    clade_group: str
    gc_fraction: float
    shard_index: int
    token_offset: int
    token_count: int
    base_count: int

    def __post_init__(self) -> None:
        if self.split not in {"train", "val", "test"}:
            raise ValueError("token stream split is invalid")
        if self.shard_index < 0 or self.token_offset < 0 or self.token_count < 2:
            raise ValueError("token stream index values are invalid")
        if not 0.0 <= self.gc_fraction <= 1.0:
            raise ValueError("token stream GC fraction is invalid")


def materialize_token_stream_dataset(
    records: Iterable[Mapping[str, object]],
    tokenizer: GenomeTokenizer,
    output_dir: str | Path,
    *,
    tokens_per_shard: int = 50_000_000,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, Path]:
    """Write production-scale contig token streams without segment explosion."""

    if tokens_per_shard <= 0:
        raise ValueError("tokens_per_shard must be positive")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "token_stream_index.jsonl"
    tokenizer_path = root / "tokenizer_spec.json"
    tokenizer_path.write_text(
        json.dumps(tokenizer.spec.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shard_rows: list[dict[str, object]] = []
    buffered_ids: list[np.ndarray] = []
    buffered_lengths: list[np.ndarray] = []
    buffered_tokens = 0
    stream_count = 0
    total_tokens = 0
    total_bases = 0

    def flush() -> None:
        nonlocal buffered_ids, buffered_lengths, buffered_tokens
        if not buffered_ids:
            return
        shard_index = len(shard_rows)
        token_path = root / f"tokens_{shard_index:05d}.npy"
        length_path = root / f"base_lengths_{shard_index:05d}.npy"
        np.save(token_path, np.concatenate(buffered_ids).astype(np.int32, copy=False), allow_pickle=False)
        np.save(length_path, np.concatenate(buffered_lengths).astype(np.uint16, copy=False), allow_pickle=False)
        shard_rows.append(
            {
                "shard_index": shard_index,
                "tokens": token_path.name,
                "base_lengths": length_path.name,
                "token_count": buffered_tokens,
                "tokens_sha256": hashlib.sha256(token_path.read_bytes()).hexdigest(),
                "base_lengths_sha256": hashlib.sha256(length_path.read_bytes()).hexdigest(),
            }
        )
        buffered_ids = []
        buffered_lengths = []
        buffered_tokens = 0

    required = {"accession", "contig_id", "sequence", "split", "clade_group"}
    with index_path.open("w", encoding="utf-8") as index_handle:
        for record in records:
            missing = required - set(record)
            if missing:
                raise ValueError(f"token stream record is missing columns: {sorted(missing)}")
            encoded = tokenizer.encode(str(record["sequence"]))
            if len(encoded.ids) < 2:
                continue
            if buffered_ids and buffered_tokens + len(encoded.ids) > tokens_per_shard:
                flush()
            lengths = np.asarray([end - start for start, end in encoded.base_spans], dtype=np.int64)
            if int(lengths.max(initial=0)) > np.iinfo(np.uint16).max:
                raise ValueError("one tokenizer token spans more than uint16 bases")
            shard_index = len(shard_rows)
            token_offset = buffered_tokens
            stream_id = f"{record['accession']}:{record['contig_id']}"
            normalized = str(record["sequence"]).upper()
            gc_fraction = (
                (normalized.count("G") + normalized.count("C")) / len(normalized)
                if normalized
                else 0.0
            )
            row = TokenStreamIndex(
                stream_id=stream_id,
                accession=str(record["accession"]),
                contig_id=str(record["contig_id"]),
                split=str(record["split"]),
                clade_group=str(record["clade_group"]),
                gc_fraction=gc_fraction,
                shard_index=shard_index,
                token_offset=token_offset,
                token_count=len(encoded.ids),
                base_count=encoded.base_count,
            )
            index_handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")
            buffered_ids.append(np.asarray(encoded.ids, dtype=np.int32))
            buffered_lengths.append(lengths.astype(np.uint16))
            buffered_tokens += len(encoded.ids)
            stream_count += 1
            total_tokens += len(encoded.ids)
            total_bases += encoded.base_count
    flush()
    manifest_path = root / "token_stream_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "layout": "contig_indexed_lazy_segments",
                "segment_length": 32,
                "streams": stream_count,
                "tokens": total_tokens,
                "bases": total_bases,
                "index": index_path.name,
                "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
                "tokenizer": tokenizer.spec.to_dict(),
                "shards": shard_rows,
                "provenance": dict(provenance or {}),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"manifest": manifest_path, "index": index_path, "tokenizer": tokenizer_path}


class LazyTokenStream(Sequence[StreamSegment]):
    """Generate ordered 32-token segments lazily from one mmap-backed contig."""

    def __init__(
        self,
        index: TokenStreamIndex,
        token_ids: np.ndarray,
        base_lengths: np.ndarray,
        *,
        pad_token_id: int,
    ) -> None:
        self.index = index
        self.token_ids = token_ids
        self.base_lengths = base_lengths
        self.pad_token_id = pad_token_id
        self._cached_index = -1
        self._cached_next_base = 0

    def __len__(self) -> int:
        return (self.index.token_count - 2) // 32 + 1

    def _base_start(self, segment_index: int) -> int:
        offset = segment_index * 32
        source_start = self.index.token_offset
        if segment_index == self._cached_index + 1:
            return self._cached_next_base
        return int(self.base_lengths[source_start : source_start + offset].sum(dtype=np.int64))

    def __getitem__(self, segment_index: int) -> StreamSegment:
        if segment_index < 0:
            segment_index += len(self)
        if not 0 <= segment_index < len(self):
            raise IndexError(segment_index)
        local_offset = segment_index * 32
        count = min(32, self.index.token_count - 1 - local_offset)
        source = self.index.token_offset + local_offset
        inputs = self.token_ids[source : source + count].astype(np.int64).tolist()
        labels = self.token_ids[source + 1 : source + count + 1].astype(np.int64).tolist()
        label_lengths = self.base_lengths[source + 1 : source + count + 1].astype(np.int64).tolist()
        base_start = self._base_start(segment_index)
        base_end = base_start + int(self.base_lengths[source : source + count + 1].sum(dtype=np.int64))
        self._cached_index = segment_index
        self._cached_next_base = base_start + int(
            self.base_lengths[source : source + count].sum(dtype=np.int64)
        )
        padding = 32 - count
        return StreamSegment(
            stream_id=self.index.stream_id,
            accession=self.index.accession,
            contig_id=self.index.contig_id,
            split=self.index.split,
            clade_group=self.index.clade_group,
            gc_fraction=self.index.gc_fraction,
            segment_index=segment_index,
            token_offset=local_offset,
            base_start=base_start,
            base_end=base_end,
            input_ids=tuple(inputs + [self.pad_token_id] * padding),
            labels=tuple(labels + [self.pad_token_id] * padding),
            valid_mask=tuple([True] * count + [False] * padding),
            loss_mask=tuple([True] * count + [False] * padding),
            represented_base_counts=tuple(label_lengths + [0] * padding),
            start_of_stream=segment_index == 0,
            end_of_stream=segment_index + 1 == len(self),
        )


class TokenStreamDataset:
    """Memory-mapped production dataset with a small contig-level index."""

    def __init__(
        self,
        root: str | Path,
        *,
        verify_checksums: bool = False,
    ) -> None:
        self.root = Path(root)
        self.manifest = json.loads((self.root / "token_stream_manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("format_version") != 1:
            raise ValueError("unsupported Stage C token stream format")
        index_path = self.root / str(self.manifest["index"])
        if verify_checksums and hashlib.sha256(index_path.read_bytes()).hexdigest() != self.manifest["index_sha256"]:
            raise ValueError("Stage C token stream index checksum mismatch")
        self.index = tuple(
            TokenStreamIndex(**json.loads(line))
            for line in index_path.read_text(encoding="utf-8").splitlines()
        )
        self.tokens: list[np.ndarray] = []
        self.base_lengths: list[np.ndarray] = []
        for shard in self.manifest["shards"]:
            token_path = self.root / shard["tokens"]
            length_path = self.root / shard["base_lengths"]
            if verify_checksums:
                if hashlib.sha256(token_path.read_bytes()).hexdigest() != shard["tokens_sha256"]:
                    raise ValueError(f"token shard checksum mismatch: {token_path.name}")
                if hashlib.sha256(length_path.read_bytes()).hexdigest() != shard["base_lengths_sha256"]:
                    raise ValueError(f"base-length shard checksum mismatch: {length_path.name}")
            self.tokens.append(np.load(token_path, mmap_mode="r", allow_pickle=False))
            self.base_lengths.append(np.load(length_path, mmap_mode="r", allow_pickle=False))

    def streams(
        self,
        *,
        split: str,
        stream_ids: Iterable[str] | None = None,
    ) -> dict[str, LazyTokenStream]:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        selected = set(map(str, stream_ids)) if stream_ids is not None else None
        pad_token_id = int(self.manifest["tokenizer"]["pad_token_id"])
        streams = {
            item.stream_id: LazyTokenStream(
                item,
                self.tokens[item.shard_index],
                self.base_lengths[item.shard_index],
                pad_token_id=pad_token_id,
            )
            for item in self.index
            if item.split == split and (selected is None or item.stream_id in selected)
        }
        if selected is not None:
            missing = selected - set(streams)
            if missing:
                raise ValueError(
                    f"selected stream IDs are absent from split {split!r}: "
                    + ", ".join(sorted(missing)[:5])
                )
        return streams
