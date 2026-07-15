"""Taxonomy-aware deterministic sampling for E. coli-related genomes."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_SCOPE_FRACTIONS = {
    "ecoli_species": 0.35,
    "escherichia_genus": 0.15,
    "enterobacteriaceae_family": 0.30,
    "enterobacterales_order": 0.20,
}


def _taxonomy_parts(taxonomy: str) -> dict[str, str]:
    ranks = ("domain", "phylum", "class", "order", "family", "genus", "species")
    values = str(taxonomy or "").split(";")
    return {rank: values[index].split("__", 1)[-1] if index < len(values) else "" for index, rank in enumerate(ranks)}


def classify_scope(taxonomy: str) -> str | None:
    parts = _taxonomy_parts(taxonomy)
    species = parts["species"].replace("_", " ")
    if species == "Escherichia coli":
        return "ecoli_species"
    if parts["genus"] == "Escherichia":
        return "escherichia_genus"
    if parts["family"] == "Enterobacteriaceae":
        return "enterobacteriaceae_family"
    if parts["order"] == "Enterobacterales":
        return "enterobacterales_order"
    return None


def prepare_candidates(
    metadata: pd.DataFrame,
    min_completeness: float = 90.0,
    max_contamination: float = 5.0,
    min_genome_size: int = 500_000,
    max_genome_size: int = 12_000_000,
) -> pd.DataFrame:
    """Normalize GTDB metadata, apply quality filters, and assign one scope."""

    frame = metadata.copy()
    aliases = {
        "checkm_completeness": "completeness",
        "checkm_contamination": "contamination",
        "genome_size": "genome_size",
        "gtdb_representative": "gtdb_representative",
    }
    for source, target in aliases.items():
        if target not in frame.columns and source in frame.columns:
            frame[target] = frame[source]
    required = {"accession", "gtdb_taxonomy", "completeness", "contamination", "genome_size"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"GTDB metadata is missing columns: {sorted(missing)}")
    for column in ("completeness", "contamination", "genome_size"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["scope"] = frame["gtdb_taxonomy"].map(classify_scope)
    frame = frame[
        frame["scope"].notna()
        & frame["completeness"].ge(min_completeness)
        & frame["contamination"].le(max_contamination)
        & frame["genome_size"].between(min_genome_size, max_genome_size)
    ].copy()
    taxonomy = frame["gtdb_taxonomy"].map(_taxonomy_parts)
    for rank in ("order", "family", "genus", "species"):
        frame[rank] = taxonomy.map(lambda item, key=rank: item[key])
    representative = frame.get("gtdb_representative", pd.Series("", index=frame.index)).astype(str).str.lower()
    frame["is_representative"] = representative.isin(("t", "true", "1", "yes"))
    frame["quality_score"] = frame["completeness"] - 5.0 * frame["contamination"]
    return frame.sort_values(
        ["scope", "is_representative", "quality_score", "accession"],
        ascending=[True, False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def _round_robin(frame: pd.DataFrame, group_columns: Sequence[str], seed: int) -> list[int]:
    def ordered_indices(group: pd.DataFrame, columns: Sequence[str], level_seed: int) -> list[int]:
        if not columns:
            return group.sort_values(
                ["is_representative", "quality_score", "accession"],
                ascending=[False, False, True],
                kind="stable",
            ).index.tolist()
        column = columns[0]
        child_groups = [(str(key), child) for key, child in group.groupby(column, dropna=False, sort=True)]
        rng = np.random.default_rng(level_seed)
        if child_groups:
            tie_breakers = rng.random(len(child_groups))
            child_groups = [
                item
                for _, item in sorted(
                    zip(
                        (
                            (
                                -int(child["is_representative"].any()),
                                -float(child["quality_score"].max()),
                                tie_breakers[index],
                                key,
                            )
                            for index, (key, child) in enumerate(child_groups)
                        ),
                        child_groups,
                    ),
                    key=lambda pair: pair[0],
                )
            ]
        queues = [ordered_indices(child, columns[1:], level_seed + offset + 1) for offset, (_, child) in enumerate(child_groups)]
        result: list[int] = []
        while queues:
            remaining: list[list[int]] = []
            for queue in queues:
                result.append(queue.pop(0))
                if queue:
                    remaining.append(queue)
            queues = remaining
        return result

    return ordered_indices(frame, group_columns, seed)


def sample_accessions(
    candidates: pd.DataFrame,
    target_bp: int = 15_000_000_000,
    scope_fractions: Mapping[str, float] = DEFAULT_SCOPE_FRACTIONS,
    seed: int = 17,
) -> pd.DataFrame:
    """Select genomes to deterministic per-scope base-pair budgets."""

    if target_bp <= 0:
        raise ValueError("target_bp must be positive")
    if not np.isclose(sum(scope_fractions.values()), 1.0):
        raise ValueError("scope fractions must sum to 1")
    chosen: list[pd.DataFrame] = []
    grouping = {
        "ecoli_species": ("organism_name",) if "organism_name" in candidates.columns else ("accession",),
        "escherichia_genus": ("species",),
        "enterobacteriaceae_family": ("species",),
        "enterobacterales_order": ("family", "genus", "species"),
    }
    for offset, (scope, fraction) in enumerate(scope_fractions.items()):
        available = candidates[candidates["scope"].eq(scope)]
        order = _round_robin(available, grouping[scope], seed + offset)
        budget = int(target_bp * fraction)
        running = 0
        indices: list[int] = []
        for index in order:
            indices.append(index)
            running += int(candidates.at[index, "genome_size"])
            if running >= budget:
                break
        if running < budget:
            raise ValueError(f"scope {scope!r} has only {running:,} bp; required {budget:,} bp")
        chosen.append(candidates.loc[indices])
    result = pd.concat(chosen, ignore_index=True) if chosen else candidates.iloc[:0].copy()
    result["sampling_seed"] = seed
    return result.sort_values(["scope", "accession"], kind="stable").reset_index(drop=True)
