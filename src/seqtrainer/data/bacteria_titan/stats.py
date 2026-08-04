"""Dataset summary statistics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def accession_stats(manifest: pd.DataFrame) -> dict[str, object]:
    by_split = manifest.groupby("split")["genome_size"].agg(["count", "sum"]).to_dict("index")
    by_scope = manifest.groupby("scope")["genome_size"].agg(["count", "sum"]).to_dict("index")
    return {
        "accessions": int(len(manifest)),
        "total_bp": int(manifest["genome_size"].sum()),
        "by_split": by_split,
        "by_scope": by_scope,
    }


def token_shard_stats(paths: list[str | Path]) -> dict[str, object]:
    arrays = [np.load(path, mmap_mode="r", allow_pickle=False) for path in paths]
    counts = np.zeros(6, dtype=np.int64)
    for array in arrays:
        counts += np.bincount(np.asarray(array).reshape(-1), minlength=6)[:6]
    bases = counts[2:].sum()
    return {
        "shards": len(arrays),
        "windows": int(sum(len(array) for array in arrays)),
        "token_counts": counts.tolist(),
        "gc_fraction": float((counts[3] + counts[4]) / bases) if bases else float("nan"),
    }
