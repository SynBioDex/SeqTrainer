"""Frozen, nested whole-replicon panels for Stage C E. coli studies."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence
import zipfile

import pandas as pd

from .stage_c_streams import TokenStreamDataset


ACCESSION_PATTERN = re.compile(r"(?:GCF|GCA)_\d+\.\d+")
PANEL_FORMAT_VERSION = 1


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _accession(value: object) -> str:
    match = ACCESSION_PATTERN.search(str(value))
    if match is None:
        raise ValueError(f"cannot recover assembly accession from {value!r}")
    return match.group(0)


@dataclass(frozen=True)
class StageCPanelManifest:
    """Validated stream selection linked to one immutable token dataset."""

    payload: Mapping[str, object]
    path: Path | None = None

    REQUIRED = {
        "format_version",
        "panel_id",
        "role",
        "split",
        "parent_dataset_fingerprint",
        "stream_ids",
        "accessions",
        "predictable_bases",
        "selection_order",
    }

    def __post_init__(self) -> None:
        missing = sorted(self.REQUIRED - set(self.payload))
        if missing:
            raise ValueError(f"panel manifest is missing fields: {missing}")
        if self.payload["format_version"] != PANEL_FORMAT_VERSION:
            raise ValueError("unsupported Stage C panel format")
        if self.payload["role"] not in {"train", "validation", "test", "outgroup"}:
            raise ValueError("panel role is invalid")
        if self.payload["split"] not in {"train", "val", "test"}:
            raise ValueError("panel split is invalid")
        streams = self.payload["stream_ids"]
        accessions = self.payload["accessions"]
        if (
            not isinstance(streams, list)
            or not streams
            or len(streams) != len(set(map(str, streams)))
        ):
            raise ValueError("panel stream_ids must be unique and non-empty")
        if (
            not isinstance(accessions, list)
            or not accessions
            or len(accessions) != len(set(map(str, accessions)))
        ):
            raise ValueError("panel accessions must be unique and non-empty")
        if int(self.payload["predictable_bases"]) <= 0:
            raise ValueError("panel predictable_bases must be positive")

    @classmethod
    def from_path(cls, path: str | Path) -> "StageCPanelManifest":
        source = Path(path)
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("panel manifest must contain a JSON object")
        return cls(value, source)

    @property
    def hash(self) -> str:
        return hashlib.sha256(canonical_json(self.payload).encode("utf-8")).hexdigest()

    @property
    def stream_ids(self) -> frozenset[str]:
        return frozenset(map(str, self.payload["stream_ids"]))


def dataset_fingerprint(dataset_dir: str | Path) -> str:
    return sha256_file(Path(dataset_dir) / "token_stream_manifest.json")


def _assembly_levels_from_zips(zip_dir: Path) -> dict[str, str]:
    """Read NCBI Datasets assembly reports without extracting archives."""

    levels: dict[str, str] = {}
    for archive_path in sorted(zip_dir.glob("*.zip")):
        with zipfile.ZipFile(archive_path) as archive:
            reports = sorted(
                name
                for name in archive.namelist()
                if name.endswith("assembly_data_report.jsonl")
            )
            for report in reports:
                for line in archive.read(report).decode("utf-8", errors="replace").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    accession = row.get("accession") or row.get("assemblyAccession")
                    level = (
                        row.get("assemblyInfo", {}).get("assemblyLevel")
                        if isinstance(row.get("assemblyInfo"), Mapping)
                        else row.get("assembly_level")
                    )
                    if accession and level:
                        levels[_accession(accession)] = str(level)
    return levels


def _normalized_accessions(
    frame: pd.DataFrame,
    *,
    ncbi_zip_dir: Path | None,
) -> pd.DataFrame:
    result = frame.copy()
    accession_column = "assembly_accession" if "assembly_accession" in result else "accession"
    if accession_column not in result:
        raise ValueError("accession manifest has no assembly accession")
    result["accession"] = result[accession_column].map(_accession)
    aliases = {
        "ecoli_related_scope": "scope",
        "checkm_completeness": "completeness",
        "checkm_contamination": "contamination",
    }
    for source, target in aliases.items():
        if target not in result and source in result:
            result[target] = result[source]
    if "assembly_level" not in result:
        if ncbi_zip_dir is None:
            raise ValueError(
                "complete-genome selection requires assembly_level or --ncbi-zip-dir"
            )
        levels = _assembly_levels_from_zips(ncbi_zip_dir)
        result["assembly_level"] = result["accession"].map(levels)
    for column in ("scope", "completeness", "contamination", "assembly_level"):
        if column not in result:
            raise ValueError(f"accession manifest is missing {column}")
    result["completeness"] = pd.to_numeric(result["completeness"], errors="coerce")
    result["contamination"] = pd.to_numeric(result["contamination"], errors="coerce")
    return result


def _ani_lookup(pairs: pd.DataFrame) -> dict[tuple[str, str], float]:
    required = {"Ref_file", "Query_file", "ANI"}
    if missing := required - set(pairs):
        raise ValueError(f"ANI pair table is missing columns: {sorted(missing)}")
    values: dict[tuple[str, str], float] = {}
    for row in pairs.itertuples(index=False):
        left = _accession(getattr(row, "Ref_file"))
        right = _accession(getattr(row, "Query_file"))
        values[tuple(sorted((left, right)))] = float(getattr(row, "ANI"))
    return values


def _predictable_bases(dataset: TokenStreamDataset, stream_ids: Iterable[str]) -> int:
    selected = set(stream_ids)
    total = 0
    for item in dataset.index:
        if item.stream_id not in selected:
            continue
        first = int(dataset.base_lengths[item.shard_index][item.token_offset])
        total += item.base_count - first
    return total


def _representatives(
    candidates: pd.DataFrame,
    membership: pd.DataFrame,
    dataset: TokenStreamDataset,
    *,
    split: str,
) -> pd.DataFrame:
    if not {"accession", "ani_cluster_99"}.issubset(membership):
        raise ValueError("ANI membership requires accession and ani_cluster_99")
    membership = membership.copy()
    membership["accession"] = membership["accession"].map(_accession)
    indexed = pd.DataFrame(
        {
            "accession": [item.accession for item in dataset.index if item.split == split],
            "stream_id": [item.stream_id for item in dataset.index if item.split == split],
            "base_count": [item.base_count for item in dataset.index if item.split == split],
            "gc_fraction": [item.gc_fraction for item in dataset.index if item.split == split],
        }
    )
    assembly = (
        indexed.groupby("accession", sort=True)
        .agg(
            stream_count=("stream_id", "nunique"),
            indexed_bases=("base_count", "sum"),
            gc_fraction=("gc_fraction", "mean"),
        )
        .reset_index()
    )
    frame = candidates.merge(membership[["accession", "ani_cluster_99"]], on="accession")
    frame = frame.merge(assembly, on="accession")
    if frame.empty:
        raise ValueError("no complete E. coli train assemblies overlap the token dataset")
    source = frame["accession"].str.startswith("GCF_").astype(int)
    representative = frame.get(
        "gtdb_representative",
        frame.get("is_representative", pd.Series(False, index=frame.index)),
    )
    representative = representative.astype(str).str.lower().isin({"t", "true", "1", "yes"})
    frame["rank_refseq"] = source
    frame["rank_representative"] = representative.astype(int)
    frame["quality_score"] = frame["completeness"] - 5.0 * frame["contamination"]
    frame = frame.sort_values(
        [
            "ani_cluster_99",
            "rank_refseq",
            "rank_representative",
            "quality_score",
            "stream_count",
            "accession",
        ],
        ascending=[True, False, False, False, True, True],
        kind="stable",
    )
    return frame.groupby("ani_cluster_99", sort=True, as_index=False).first()


def _group_order(
    representatives: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    seed: int,
) -> list[str]:
    """Deterministic maximin order with phylogroup/GC diversity tie-breaks."""

    ani = _ani_lookup(pairs)
    frame = representatives.set_index("accession", drop=False)
    accessions = sorted(frame.index.astype(str))
    if not accessions:
        raise ValueError("panel selection requires representatives")
    gc_values = frame["gc_fraction"].astype(float)
    gc_bins = pd.qcut(gc_values, q=min(5, len(gc_values)), labels=False, duplicates="drop")
    phylogroup_column = next(
        (name for name in ("phylogroup", "clermont_phylogroup") if name in frame),
        None,
    )

    def hash_key(accession: str) -> str:
        return hashlib.sha256(f"stage-c-v3:{seed}:{accession}".encode()).hexdigest()

    chosen = [min(accessions, key=hash_key)]
    remaining = set(accessions) - set(chosen)
    seen_gc = {int(gc_bins.loc[chosen[0]])}
    seen_phylogroups = (
        {str(frame.loc[chosen[0], phylogroup_column])} if phylogroup_column else set()
    )
    while remaining:
        scored = []
        for accession in remaining:
            distances = []
            for prior in chosen:
                key = tuple(sorted((accession, prior)))
                if key not in ani:
                    raise ValueError(
                        "ANI evidence is incomplete for complete-genome representatives: "
                        f"{accession}, {prior}"
                    )
                distances.append(100.0 - ani[key])
            gc_novel = int(int(gc_bins.loc[accession]) not in seen_gc)
            phylogroup_novel = (
                int(str(frame.loc[accession, phylogroup_column]) not in seen_phylogroups)
                if phylogroup_column
                else 0
            )
            scored.append(
                (
                    min(distances),
                    phylogroup_novel,
                    gc_novel,
                    hash_key(accession),
                    accession,
                )
            )
        selected = max(scored)[-1]
        chosen.append(selected)
        remaining.remove(selected)
        seen_gc.add(int(gc_bins.loc[selected]))
        if phylogroup_column:
            seen_phylogroups.add(str(frame.loc[selected, phylogroup_column]))
    return chosen


def _panel_payload(
    *,
    panel_id: str,
    role: str,
    split: str,
    dataset: TokenStreamDataset,
    parent_fingerprint: str,
    accessions: Sequence[str],
    representatives: pd.DataFrame,
    target_bases: int,
    seed: int,
    source_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    selected = set(accessions)
    rows = [
        item
        for item in dataset.index
        if item.split == split and item.accession in selected
    ]
    stream_ids = [item.stream_id for item in rows]
    lookup = representatives.set_index("accession")
    order = [
        {
            "accession": accession,
            "ani_cluster_99": str(lookup.loc[accession, "ani_cluster_99"]),
            "gc_fraction": float(lookup.loc[accession, "gc_fraction"]),
            "streams": sum(item.accession == accession for item in rows),
            "bases": sum(item.base_count for item in rows if item.accession == accession),
        }
        for accession in accessions
    ]
    return {
        "format_version": PANEL_FORMAT_VERSION,
        "panel_id": panel_id,
        "role": role,
        "split": split,
        "parent_dataset_fingerprint": parent_fingerprint,
        "selection_seed": seed,
        "target_predictable_bases": target_bases,
        "predictable_bases": _predictable_bases(dataset, stream_ids),
        "accessions": list(accessions),
        "ani99_groups": [row["ani_cluster_99"] for row in order],
        "stream_ids": stream_ids,
        "selection_order": order,
        "source_provenance": dict(source_provenance or {}),
        "eligibility": {
            "scope": "ecoli_species",
            "assembly_level": "Complete Genome",
            "minimum_completeness": 95.0,
            "maximum_contamination": 2.0,
            "one_representative_per_ani99_before_repeats": True,
        },
    }


def freeze_ecoli_panels(
    *,
    dataset_dir: str | Path,
    accession_manifest: pd.DataFrame,
    ani_membership: pd.DataFrame,
    ani_pairs: pd.DataFrame,
    output_dir: str | Path,
    ncbi_zip_dir: str | Path | None = None,
    targets: Mapping[str, int] | None = None,
    seed: int = 20260751,
    source_provenance: Mapping[str, object] | None = None,
) -> dict[str, Path]:
    """Freeze nested complete-genome E. coli panels without copying token shards."""

    targets = dict(targets or {"e25": 25_000_000, "e100": 100_000_000, "e250": 250_000_000})
    if not targets or any(int(value) <= 0 for value in targets.values()):
        raise ValueError("panel targets must be positive")
    ordered_targets = sorted(targets.items(), key=lambda item: item[1])
    if len({value for _, value in ordered_targets}) != len(ordered_targets):
        raise ValueError("panel targets must be distinct")
    dataset = TokenStreamDataset(dataset_dir, verify_checksums=True)
    normalized = _normalized_accessions(
        accession_manifest,
        ncbi_zip_dir=Path(ncbi_zip_dir) if ncbi_zip_dir else None,
    )
    candidates = normalized.loc[
        normalized["scope"].eq("ecoli_species")
        & normalized["completeness"].ge(95.0)
        & normalized["contamination"].le(2.0)
        & normalized["assembly_level"].astype(str).str.casefold().eq("complete genome")
    ].copy()
    representatives = _representatives(
        candidates,
        ani_membership,
        dataset,
        split="train",
    )
    selection = _group_order(representatives, ani_pairs, seed=seed)
    parent_fingerprint = dataset_fingerprint(dataset_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    selected: list[str] = []
    panel_accessions: dict[str, list[str]] = {}
    for name, target in ordered_targets:
        while _predictable_bases(
            dataset,
            [
                item.stream_id
                for item in dataset.index
                if item.accession in set(selected) and item.split == "train"
            ],
        ) < target:
            if len(selected) >= len(selection):
                raise ValueError(
                    f"complete E. coli representatives provide insufficient bases for {name}"
                )
            selected.append(selection[len(selected)])
        payload = _panel_payload(
            panel_id=f"stage_c_v3_{name}",
            role="train",
            split="train",
            dataset=dataset,
            parent_fingerprint=parent_fingerprint,
            accessions=selected,
            representatives=representatives,
            target_bases=target,
            seed=seed,
            source_provenance=source_provenance,
        )
        path = output / f"{name}.json"
        path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        paths[name] = path
        panel_accessions[name] = list(selected)
    if "e25" in panel_accessions and "e100" in panel_accessions:
        additions = [
            accession
            for accession in panel_accessions["e100"]
            if accession not in set(panel_accessions["e25"])
        ]
        payload = _panel_payload(
            panel_id="stage_c_v3_e100_additions",
            role="train",
            split="train",
            dataset=dataset,
            parent_fingerprint=parent_fingerprint,
            accessions=additions,
            representatives=representatives,
            target_bases=max(
                1, targets["e100"] - targets["e25"]
            ),
            seed=seed,
            source_provenance=source_provenance,
        )
        path = output / "e100_additions.json"
        path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        paths["e100_additions"] = path
    normalized_membership = ani_membership.copy()
    normalized_membership["accession"] = normalized_membership["accession"].map(
        _accession
    )
    for split, role in (("val", "validation"), ("test", "test")):
        split_accessions = {
            item.accession for item in dataset.index if item.split == split
        }
        split_candidates = candidates.loc[
            candidates["accession"].isin(split_accessions)
        ]
        split_representatives = _representatives(
            split_candidates,
            normalized_membership,
            dataset,
            split=split,
        )
        if len(split_representatives) < 8:
            raise ValueError(
                f"{split} has only {len(split_representatives)} eligible complete "
                "E. coli ANI99 representatives; at least 8 are required"
            )
        split_order = _group_order(split_representatives, ani_pairs, seed=seed + 1)
        selected_heldout = split_order[: min(12, len(split_order))]
        payload = _panel_payload(
            panel_id=f"stage_c_v3_{role}",
            role=role,
            split=split,
            dataset=dataset,
            parent_fingerprint=parent_fingerprint,
            accessions=selected_heldout,
            representatives=split_representatives,
            target_bases=sum(
                item.base_count
                for item in dataset.index
                if item.split == split and item.accession in set(selected_heldout)
            ),
            seed=seed,
            source_provenance=source_provenance,
        )
        path = output / f"{role}.json"
        path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        paths[role] = path
    summary = {
        "format_version": 1,
        "parent_dataset_fingerprint": parent_fingerprint,
        "selection_seed": seed,
        "source_provenance": dict(source_provenance or {}),
        "selection_order": selection,
        "panels": {
            name: {
                "path": path.name,
                "sha256": sha256_file(path),
                "predictable_bases": StageCPanelManifest.from_path(path).payload[
                    "predictable_bases"
                ],
            }
            for name, path in paths.items()
        },
    }
    summary_path = output / "panel_summary.json"
    summary_path.write_text(canonical_json(summary) + "\n", encoding="utf-8")
    paths["summary"] = summary_path
    return paths


def validate_panel_against_dataset(
    panel: StageCPanelManifest,
    dataset: TokenStreamDataset,
) -> None:
    parent = hashlib.sha256(
        (dataset.root / "token_stream_manifest.json").read_bytes()
    ).hexdigest()
    if panel.payload["parent_dataset_fingerprint"] != parent:
        raise ValueError("panel parent dataset fingerprint does not match")
    index = {item.stream_id: item for item in dataset.index}
    missing = sorted(panel.stream_ids - set(index))
    if missing:
        raise ValueError(f"panel streams are absent from dataset: {missing[:5]}")
    wrong_split = sorted(
        stream_id
        for stream_id in panel.stream_ids
        if index[stream_id].split != panel.payload["split"]
    )
    if wrong_split:
        raise ValueError(f"panel contains streams from the wrong split: {wrong_split[:5]}")
    observed = _predictable_bases(dataset, panel.stream_ids)
    if observed != int(panel.payload["predictable_bases"]):
        raise ValueError(
            f"panel predictable-base count changed: expected "
            f"{panel.payload['predictable_bases']}, observed {observed}"
        )
