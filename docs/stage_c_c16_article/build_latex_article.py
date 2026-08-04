"""Build and verify the LaTeX c16 research article and PCA diagnostic figure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
OUT = ROOT / "artifacts" / "stage_c_c16_scientific_package"
FIG = OUT / "figures"
TEX = SOURCE / "C16_Titans_DNA_Generation_Research_Article.tex"
PDF = OUT / "C16_Titans_DNA_Generation_Research_Article.pdf"
DRIVE = SOURCE / "drive_artifacts"
ARCHITECTURE_SOURCE = SOURCE / "assets" / "figure_1_model_architecture.png"
EXPECTED_TITLE = (
    "DNA generation by a compact Titan neural-memory model shows promising "
    "genetic features but incomplete genome organization"
)
# Freeze PDF creation metadata so identical sources produce an identical digest.
SOURCE_DATE_EPOCH = "1785456000"  # 2026-07-31 00:00:00 UTC


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_architecture_figure() -> Path:
    """Install the reviewed publication figure into the generated package."""

    if not ARCHITECTURE_SOURCE.is_file():
        raise FileNotFoundError(ARCHITECTURE_SOURCE)
    FIG.mkdir(parents=True, exist_ok=True)
    destination = FIG / "figure_1_model_architecture.png"
    shutil.copyfile(ARCHITECTURE_SOURCE, destination)
    return destination


def build_pca_figure() -> Path:
    memory = json.loads((DRIVE / "memory_trace.json").read_text(encoding="utf-8"))
    embedding = json.loads((DRIVE / "embedding_pca.json").read_text(encoding="utf-8"))
    rows = memory["rows"]
    erows = embedding["rows"]

    pc1 = np.asarray([row["pc1"] for row in rows], dtype=float)
    pc2 = np.asarray([row["pc2"] for row in rows], dtype=float)
    gc = np.asarray([row["gc_fraction"] for row in rows], dtype=float)
    positions = np.asarray([row["segment_index"] for row in rows], dtype=float)
    position_norm = np.zeros_like(positions)
    for stream in sorted({row["stream_id"] for row in rows}):
        mask = np.asarray([row["stream_id"] == stream for row in rows])
        local = positions[mask]
        span = max(float(local.max() - local.min()), 1.0)
        position_norm[mask] = (local - local.min()) / span

    memory_variance = memory["pca_explained_variance"]
    embedding_variance = embedding["pca_explained_variance"]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.7), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("#f7fafc")
        ax.grid(color="#d9e2ea", linewidth=0.6, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    first = axes[0].scatter(pc1, pc2, c=position_norm, cmap="viridis", s=20, alpha=0.82)
    axes[0].set_title("a  Memory trajectory", loc="left", weight="bold", color="#13233f")
    axes[0].set_xlabel(f"PC1 ({100 * memory_variance[0]:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({100 * memory_variance[1]:.1f}%)")
    bar = fig.colorbar(first, ax=axes[0], fraction=0.046, pad=0.03)
    bar.set_label("normalized stream position")
    axes[0].text(0.02, 0.02, "PC1-position r = 0.683", transform=axes[0].transAxes,
                 fontsize=9, color="#5d6975")

    second = axes[1].scatter(pc1, pc2, c=gc, cmap="coolwarm", s=20, alpha=0.82)
    axes[1].set_title("b  Memory and local GC", loc="left", weight="bold", color="#13233f")
    axes[1].set_xlabel(f"PC1 ({100 * memory_variance[0]:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({100 * memory_variance[1]:.1f}%)")
    bar = fig.colorbar(second, ax=axes[1], fraction=0.046, pad=0.03)
    bar.set_label("segment GC fraction")
    axes[1].text(0.02, 0.02, "PC1-GC r = 0.079", transform=axes[1].transAxes,
                 fontsize=9, color="#5d6975")

    epc1 = np.asarray([row["pc1"] for row in erows], dtype=float)
    epc2 = np.asarray([row["pc2"] for row in erows], dtype=float)
    axes[2].scatter(epc1, epc2, color="#2563a6", s=45, edgecolor="white", linewidth=0.7)
    for index, (x, y) in enumerate(zip(epc1, epc2), start=1):
        axes[2].annotate(str(index), (x, y), xytext=(4, 3), textcoords="offset points",
                         fontsize=8, color="#13233f")
    axes[2].set_title("c  Mean hidden embeddings", loc="left", weight="bold", color="#13233f")
    axes[2].set_xlabel(f"PC1 ({100 * embedding_variance[0]:.1f}%)")
    axes[2].set_ylabel(f"PC2 ({100 * embedding_variance[1]:.1f}%)")
    axes[2].text(0.02, 0.02, "8 streams; one accession; one species",
                 transform=axes[2].transAxes, fontsize=9, color="#5d6975")

    path = FIG / "figure_7_pca_diagnostics.png"
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def compiler_path(requested: str | None) -> str:
    candidate = requested or os.environ.get("TECTONIC") or shutil.which("tectonic")
    if not candidate:
        raise RuntimeError("Tectonic was not found; pass --tectonic or set TECTONIC")
    return candidate


def compile_article(tectonic: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [tectonic, "--outdir", str(OUT), TEX.name],
        cwd=SOURCE,
        env={**os.environ, "SOURCE_DATE_EPOCH": SOURCE_DATE_EPOCH},
        check=True,
    )


def verify_pdf() -> int:
    source = TEX.read_text(encoding="utf-8")
    if "\\documentclass[10pt,twocolumn]{article}" not in source:
        raise RuntimeError("article must use the two-column document class")
    if "\\onecolumn" in source:
        raise RuntimeError("article contains a one-column layout switch")
    if "figure_2_memory_math" in source or "fig:memorymath" in source:
        raise RuntimeError("superseded neural-memory Figure 2 remains in the article")
    supporting = source.partition("\\section*{Supporting Information}")[2]
    if not supporting or supporting.count("\\begin{table*}") != 2:
        raise RuntimeError(
            "Supporting Information must contain exactly two double-column tables"
        )

    reader = PdfReader(str(PDF))
    if len(reader.pages) < 7:
        raise RuntimeError(f"article has only {len(reader.pages)} pages")
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    normalized = " ".join(text.split())
    required = [
        EXPECTED_TITLE,
        "convention relative to the paper",
        "Notation summary",
        "Relationship to the Titans",
        "Dataset construction and relation to the parent corpus",
        "Prodigal gene-structure diagnostics",
        "incomplete genome organization",
        "Supporting Information",
        "Detailed correspondence between the Titans architecture",
        "evidence matrix for the c16",
    ]
    missing = [phrase for phrase in required if phrase not in normalized]
    if missing:
        raise RuntimeError(f"compiled PDF is missing required text: {missing}")
    return len(reader.pages)


def update_manifest(
    pages: int,
    architecture_path: Path,
    pca_path: Path,
    tectonic: str,
) -> None:
    existing: dict[str, str] = {}
    manifest_path = OUT / "BUILD_MANIFEST.txt"
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                existing[key] = value
    # The former memory-mathematics graphic is intentionally no longer part
    # of the article; do not preserve its stale package-manifest entry.
    existing.pop("figure_math_sha256", None)
    existing.update(
        {
            "article": PDF.name,
            "article_pages": str(pages),
            "article_sha256": sha256(PDF),
            "article_source": str(TEX.relative_to(ROOT)),
            "article_source_sha256": sha256(TEX),
            "article_compiler": subprocess.run(
                [tectonic, "--version"], capture_output=True, text=True, check=True
            ).stdout.strip(),
            "figures": "6",
            "figure_architecture_sha256": sha256(architecture_path),
            "figure_architecture_source": str(ARCHITECTURE_SOURCE.relative_to(ROOT)),
            "figure_architecture_source_sha256": sha256(ARCHITECTURE_SOURCE),
            "figure_pca_sha256": sha256(pca_path),
            "pca_memory_source_sha256": sha256(DRIVE / "memory_trace.json"),
            "pca_embedding_source_sha256": sha256(DRIVE / "embedding_pca.json"),
        }
    )
    preferred = [
        "article", "article_pages", "article_sha256", "article_source",
        "article_source_sha256", "article_compiler", "slides", "slide_pages",
        "slides_sha256", "figures",
    ]
    keys = preferred + sorted(key for key in existing if key not in preferred)
    manifest_path.write_text(
        "".join(f"{key}={existing[key]}\n" for key in keys if key in existing),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tectonic", help="path to the Tectonic executable")
    args = parser.parse_args()
    architecture_path = install_architecture_figure()
    pca_path = build_pca_figure()
    tectonic = compiler_path(args.tectonic)
    compile_article(tectonic)
    pages = verify_pdf()
    update_manifest(pages, architecture_path, pca_path, tectonic)
    print(f"article={PDF}")
    print(f"pages={pages}")
    print(f"sha256={sha256(PDF)}")


if __name__ == "__main__":
    main()
