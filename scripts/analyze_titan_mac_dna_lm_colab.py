#!/usr/bin/env python3
"""Create PCA/t-SNE and memory diagnostics for a trained Titan MAC DNA LM."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from seqtrainer.data.bacteria_titan import TokenShardDataset
from seqtrainer.torch.titans_mac import TitansMACForCausalLM, TitansMACLMConfig, load_training_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--dataset-class", default="bacteria_titan_v1_ecoli_related_15gbp")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tsne", action="store_true")
    parser.add_argument("--metadata-csv", type=Path)
    return parser.parse_args()


def scatter(
    frame: pd.DataFrame,
    dimensions: int,
    color: str,
    path: Path,
    x: str = "PC1",
    y: str = "PC2",
) -> None:
    import matplotlib.pyplot as plt

    values = frame[color]
    categorical = not pd.api.types.is_numeric_dtype(values)
    plotted_values = pd.Categorical(values).codes if categorical else values
    figure = plt.figure(figsize=(8, 6))
    if dimensions == 3:
        axis = figure.add_subplot(111, projection="3d")
        points = axis.scatter(frame[x], frame[y], frame["PC3"], c=plotted_values, s=8, cmap="viridis")
        axis.set_zlabel("PC3")
    else:
        axis = figure.add_subplot(111)
        points = axis.scatter(frame[x], frame[y], c=plotted_values, s=8, cmap="viridis")
    axis.set(xlabel=x, ylabel=y)
    colorbar = figure.colorbar(points, ax=axis, label=color)
    if categorical:
        categories = pd.Categorical(values).categories
        colorbar.set_ticks(np.arange(len(categories)))
        colorbar.set_ticklabels(categories)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_projection(values: np.ndarray, labels: pd.DataFrame, prefix: Path, tsne: bool) -> None:
    count = min(len(values), len(labels))
    components = min(3, values.shape[1], count)
    coordinates = PCA(n_components=components).fit_transform(values[:count])
    frame = labels.iloc[:count].reset_index(drop=True).copy()
    for index in range(components):
        frame[f"PC{index + 1}"] = coordinates[:, index]
    frame.to_csv(prefix.with_name(prefix.name + "_pca_coordinates.csv"), index=False)
    color_columns = [
        column
        for column in ("gc_fraction", "split", "scope", "top_memory_slot", "accession", "genus", "family")
        if column in frame and frame[column].notna().any()
    ]
    for color in color_columns:
        scatter(frame, 2, color, prefix.with_name(prefix.name + f"_pca_2d_by_{color}.png"))
        if components >= 3:
            scatter(frame, 3, color, prefix.with_name(prefix.name + f"_pca_3d_by_{color}.png"))
    if tsne and count >= 5:
        perplexity = min(30, max(2, count // 10), count - 1)
        embedding = TSNE(n_components=2, perplexity=perplexity, random_state=17, init="pca").fit_transform(values[:count])
        frame["TSNE1"], frame["TSNE2"] = embedding[:, 0], embedding[:, 1]
        frame.to_csv(prefix.with_name(prefix.name + "_tsne_coordinates.csv"), index=False)
        for color in color_columns:
            scatter(
                frame,
                2,
                color,
                prefix.with_name(prefix.name + f"_tsne_by_{color}.png"),
                x="TSNE1",
                y="TSNE2",
            )


def main() -> None:
    args = parse_args()
    run_dir = args.drive_root / "runs" / args.run_name
    checkpoint_path = run_dir / "best_overall.pt"
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / "latest.pt"
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = TitansMACLMConfig.from_dict(raw["config"])
    model = TitansMACForCausalLM(config)
    load_training_checkpoint(checkpoint_path, model, map_location="cpu", trusted=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    token_root = (
        args.drive_root / args.dataset_class / "tokenized" / args.dataset_class / f"ctx{args.context_length}"
    )
    records: list[dict[str, object]] = []
    inputs: list[np.ndarray] = []
    for split in ("train", "val", "test"):
        dataset = TokenShardDataset(token_root / split)
        take = min(len(dataset), max(1, args.samples // 3))
        for index in np.linspace(0, len(dataset) - 1, take, dtype=int):
            tokens, _ = dataset[int(index)]
            valid = tokens[tokens >= 2]
            gc = float(np.isin(valid, (3, 4)).mean()) if len(valid) else float("nan")
            inputs.append(tokens)
            records.append({"split": split, "dataset_index": int(index), "gc_fraction": gc})
    labels = pd.DataFrame(records)
    if args.metadata_csv:
        metadata = pd.read_csv(args.metadata_csv)
        if len(metadata) == len(labels):
            labels = pd.concat((labels, metadata.reset_index(drop=True)), axis=1)

    sequence_vectors: list[np.ndarray] = []
    top_slots: list[int] = []
    usage = np.zeros(config.memory_slots, dtype=np.int64)
    for start in range(0, len(inputs), args.batch_size):
        batch = torch.tensor(np.stack(inputs[start : start + args.batch_size]), dtype=torch.long, device=device)
        with torch.inference_mode():
            sequence_vectors.append(model.extract_sequence_embeddings(batch).float().cpu().numpy())
            diagnostics = model.get_memory_diagnostics(batch)
        indices = diagnostics["slot_indices"].cpu().numpy()
        top_slots.extend(indices[:, 0].tolist())
        usage += np.bincount(indices.reshape(-1), minlength=config.memory_slots)
    labels["top_memory_slot"] = top_slots
    output = args.output_dir or run_dir / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    save_projection(np.concatenate(sequence_vectors), labels, output / "sequence_embeddings", args.tsne)

    token_values = model.token_embeddings.weight.detach().float().cpu().numpy()
    token_labels = pd.DataFrame({"token": ["PAD", "N", "A", "C", "G", "T"], "gc_fraction": [0, 0, 0, 1, 1, 0]})
    save_projection(token_values, token_labels, output / "token_embeddings", False)
    position_values = model.position_embeddings.weight.detach().float().cpu().numpy()
    position_labels = pd.DataFrame({"position": np.arange(len(position_values)), "gc_fraction": np.zeros(len(position_values))})
    save_projection(position_values, position_labels, output / "position_embeddings", False)

    import matplotlib.pyplot as plt

    final_diagnostics = model.get_memory_diagnostics(torch.tensor(np.stack(inputs[:1]), dtype=torch.long, device=device))
    cosine = final_diagnostics["slot_cosine_similarity"].float().cpu().numpy()
    pd.DataFrame(cosine).to_csv(output / "memory_slot_cosine.csv", index=False)
    pd.DataFrame({"slot": np.arange(len(usage)), "retrieval_count": usage}).to_csv(
        output / "memory_slot_usage.csv", index=False
    )
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.bar(np.arange(len(usage)), usage)
    axis.set(xlabel="memory slot", ylabel="retrieval count")
    figure.tight_layout()
    figure.savefig(output / "memory_slot_usage.png", dpi=180)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(cosine, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set(xlabel="memory slot", ylabel="memory slot")
    figure.colorbar(image, ax=axis, label="cosine similarity")
    figure.tight_layout()
    figure.savefig(output / "memory_slot_cosine_heatmap.png", dpi=180)
    plt.close(figure)

    (output / "REPORT.md").write_text(
        """# Titan MAC Analysis Report

## Embedding projections

The 2D and 3D PCA plots show dominant linear structure in learned token,
position, and sequence hidden embeddings. Sequence points are colored by GC
fraction; the coordinate CSV also includes split and top retrieved memory slot.
Optional accession, genus, family, and scope columns are retained when a
row-aligned metadata CSV is supplied. t-SNE, when requested, emphasizes local
neighborhoods and should not be interpreted as preserving global distance.

## Memory diagnostics

Memory-slot usage measures how often each slot was among the retrieved context
tokens. The cosine heatmap compares learned slot directions. Collapsed usage or
uniformly high cosine similarity can indicate redundant memory; scope-specific
usage with diverse slot directions is evidence that retrieval has specialized.
""",
        encoding="utf-8",
    )
    print(f"Analysis written to {output}")


if __name__ == "__main__":
    main()
