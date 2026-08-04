"""Build the c16 exploratory research article, figures, and slide-deck PDFs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "stage_c_c16_scientific_package"
FIG = OUT / "figures"
NAVY = "#13233f"
BLUE = "#2563a6"
CYAN = "#16a3a5"
CORAL = "#ef6a5b"
GOLD = "#e7a83e"
PALE = "#eef4f8"
INK = "#18212b"
MUTED = "#5d6975"


POLICIES = [
    ("T0.6 · k1024 · p.99", 0.6, "1024", 0.99, 0.0682, 0.802, 0.0638, 0.4055, 0.80, 0.471),
    ("T0.8 · k128 · p.99", 0.8, "128", 0.99, 0.0745, 0.285, 0.1352, 0.5450, 1.60, 1.068),
    ("T0.8 · k512 · p.99", 0.8, "512", 0.99, 0.0594, 0.682, 0.0825, 0.4310, 1.20, 1.050),
    ("T0.8 · k1024 · p.99", 0.8, "1024", 0.99, 0.0646, 0.822, 0.0616, 0.4010, 1.00, 0.515),
    ("T1.0 · k1024 · p.99", 1.0, "1024", 0.99, 0.0664, 0.838, 0.0580, 0.3939, 1.00, 0.408),
    ("T1.1 · k1024 · p.99", 1.1, "1024", 0.99, 0.0649, 0.842, 0.0570, 0.3933, 0.80, 0.340),
    ("T0.8 · no-k · p.95", 0.8, "none", 0.95, 0.0524, 0.982, 0.0237, 0.3335, 1.20, 0.362),
    ("T0.8 · no-k · p.99", 0.8, "none", 0.99, 0.0474, 0.986, 0.0200, 0.3270, 1.00, 0.668),
    ("T0.8 · no-k · p1.0", 0.8, "none", 1.00, 0.0470, 0.988, 0.0192, 0.3261, 1.00, 0.668),
]

REFERENCE = {
    "gc": 0.51416015625,
    "entropy": 1.9876506041901352,
    "homopolymer": 6.5,
    "aligned_diversity": 0.921875,
    "overlap_diversity": 0.638408868601239,
    "orfs": 28.0,
    "longest_orf": 864.0,
    "genes_10kb": 8.138020833333334,
    "coding_density": 0.494140625,
    "gene_length": 504.0,
    "intergenic": 72.0,
}

BEST = {
    "gc": 0.46712239583333337,
    "entropy": 1.9955347271647463,
    "homopolymer": 9.0,
    "aligned_diversity": 0.9111328125,
    "overlap_diversity": 0.6571568307792631,
    "orfs": 37.5,
    "longest_orf": 372.0,
    "genes_10kb": 8.138020833333334,
    "coding_density": 0.330078125,
    "gene_length": 345.0,
    "intergenic": 506.0,
    "jsd": [0.0016074, 0.0079547, 0.0192136, 0.0395340, 0.0966383, 0.3261467],
}

BLOCK_IMPROVEMENT = [9.600, 11.312, 10.723, 10.205]
DELAYED_MARGIN = [0.0017356, 0.0017426, 0.0015309, 0.0014727]


def configure_fonts() -> tuple[str, str]:
    sans = "/System/Library/Fonts/Supplemental/Arial.ttf"
    mono = "/System/Library/Fonts/Supplemental/Courier New.ttf"
    if Path(sans).exists():
        pdfmetrics.registerFont(TTFont("ArticleSans", sans))
        sans_name = "ArticleSans"
    else:
        sans_name = "Helvetica"
    if Path(mono).exists():
        pdfmetrics.registerFont(TTFont("ArticleMono", mono))
        mono_name = "ArticleMono"
    else:
        mono_name = "Courier"
    return sans_name, mono_name


SANS, MONO = configure_fonts()


def savefig(name: str) -> Path:
    path = FIG / name
    plt.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


def architecture_figure() -> Path:
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.text(0.35, 6.75, "How the tested model reads, remembers and predicts DNA", fontsize=22, weight="bold", color=NAVY)
    ax.text(0.35, 6.32, "A compact Memory-as-Context (MAC) network; four identical blocks are stacked.", fontsize=12, color=MUTED)

    def box(x, y, w, h, text, color, sub=None):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                               linewidth=1.5, edgecolor=color, facecolor=color + "18")
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.60, text, ha="center", va="center", fontsize=12, weight="bold", color=color)
        if sub:
            ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=9, color=INK)

    def arrow(a, b, color=BLUE):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, linewidth=1.8, color=color))

    box(0.35, 3.05, 1.45, 1.25, "DNA", CORAL, "ACGT… → 6-mer tokens")
    box(2.15, 3.05, 1.55, 1.25, "Embedding", BLUE, "4,103 tokens → 128-D")
    arrow((1.80, 3.68), (2.15, 3.68))
    box(4.10, 1.10, 5.35, 4.55, "Titans MAC block × 4", NAVY)
    arrow((3.70, 3.68), (4.10, 3.68))

    box(4.45, 3.75, 1.45, 1.20, "Query", CYAN, "What should I recall?")
    box(6.20, 3.75, 1.60, 1.20, "Neural memory", CYAN, "2-layer residual MLP")
    box(8.10, 3.75, 1.00, 1.20, "Recall", CYAN, "M(q)")
    arrow((5.90, 4.35), (6.20, 4.35), CYAN)
    arrow((7.80, 4.35), (8.10, 4.35), CYAN)

    box(4.45, 1.75, 2.10, 1.20, "Persistent tokens", GOLD, "4 learned task anchors")
    box(6.85, 1.75, 2.25, 1.20, "Causal attention", BLUE, "current + recall + anchors")
    arrow((6.55, 2.35), (6.85, 2.35), BLUE)
    arrow((8.60, 3.75), (8.60, 2.95), CYAN)

    box(9.85, 3.05, 1.25, 1.25, "Predict", CORAL, "next 6-mer")
    arrow((9.10, 2.35), (9.85, 3.30), BLUE)
    box(9.85, 1.30, 2.30, 1.05, "Write after reading", GOLD, "surprise updates fast weights")
    arrow((9.10, 2.15), (9.85, 1.82), GOLD)
    arrow((11.00, 2.35), (7.20, 3.75), GOLD)
    ax.text(10.12, 0.68, "Memory is stream-specific and changes during generation;\nordinary model weights stay fixed.", fontsize=9.5, color=MUTED, ha="center")
    return savefig("figure_1_model_architecture.png")


def recurrence_figure() -> Path:
    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.8)
    ax.text(0.35, 5.35, "Neural memory update: prediction error becomes “surprise”", fontsize=22, weight="bold", color=NAVY)
    steps = [
        (0.45, "Current token xₜ", "project to qₜ, kₜ, vₜ", BLUE),
        (3.05, "Memory predicts value", "Mₜ₋₁(kₜ) ≈ vₜ", CYAN),
        (5.75, "Momentary surprise", "gₜ = ∇M ½‖M(kₜ)−vₜ‖²", CORAL),
        (8.55, "Surprise momentum", "Sₜ = ηₜSₜ₋₁ − θₜgₜ", GOLD),
        (11.05, "Updated memory", "Mₜ=(1−αₜ)Mₜ₋₁+Sₜ", NAVY),
    ]
    for x, title, sub, color in steps:
        w = 2.05 if x < 11 else 1.55
        ax.add_patch(FancyBboxPatch((x, 2.15), w, 1.55, boxstyle="round,pad=.04,rounding_size=.13",
                                    facecolor=color + "18", edgecolor=color, linewidth=1.5))
        ax.text(x + w / 2, 3.18, title, ha="center", fontsize=11, weight="bold", color=color)
        ax.text(x + w / 2, 2.60, sub, ha="center", fontsize=9.5, color=INK)
    for x1, x2 in [(2.50, 3.05), (5.10, 5.75), (7.80, 8.55), (10.60, 11.05)]:
        ax.add_patch(FancyArrowPatch((x1, 2.92), (x2, 2.92), arrowstyle="-|>", mutation_scale=14, color=BLUE))
    ax.text(0.55, 1.25, "η: how long past surprise persists", color=GOLD, fontsize=11, weight="bold")
    ax.text(4.25, 1.25, "θ: strength of the new write", color=CORAL, fontsize=11, weight="bold")
    ax.text(8.15, 1.25, "α: fraction of old memory forgotten", color=NAVY, fontsize=11, weight="bold")
    ax.text(0.55, 0.55, "c16 initializes η=0.9, θ=0.001 and α=0.001, learns them by layer/channel, and applies no surprise cap.", fontsize=10.5, color=MUTED)
    return savefig("figure_2_memory_math.png")


def topk_figure() -> Path:
    rows = [p for p in POLICIES if p[1] == 0.8 and p[3] == 0.99 and p[2] in {"128", "512", "1024", "none"}]
    rows.sort(key=lambda r: 9999 if r[2] == "none" else int(r[2]))
    labels = ["128", "512", "1,024", "unrestricted"]
    diversity = [r[5] for r in rows]
    jsd = [r[7] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    axes[0].bar(labels, diversity, color=[CORAL, GOLD, BLUE, CYAN])
    axes[0].axhline(1, color=NAVY, ls="--", lw=1.2)
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel("Aligned 6-mer diversity / reference")
    axes[0].set_title("Vocabulary diversity recovers")
    axes[1].bar(labels, jsd, color=[CORAL, GOLD, BLUE, CYAN])
    axes[1].set_ylim(0, 0.60)
    axes[1].set_ylabel("6-mer Jensen–Shannon divergence (bits)")
    axes[1].set_title("Distributional mismatch decreases")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("top-k")
    fig.suptitle("The apparent generation collapse was largely decoder-induced", fontsize=18, weight="bold", color=NAVY)
    fig.tight_layout()
    return savefig("figure_3_topk_ablation.png")


def fidelity_figure() -> Path:
    names = ["GC", "aligned\n6-mer diversity", "ORF count", "longest ORF", "genes / 10 kb",
             "coding density", "median gene", "median intergenic"]
    ratios = [
        BEST["gc"] / REFERENCE["gc"],
        BEST["aligned_diversity"] / REFERENCE["aligned_diversity"],
        BEST["orfs"] / REFERENCE["orfs"],
        BEST["longest_orf"] / REFERENCE["longest_orf"],
        BEST["genes_10kb"] / REFERENCE["genes_10kb"],
        BEST["coding_density"] / REFERENCE["coding_density"],
        BEST["gene_length"] / REFERENCE["gene_length"],
        BEST["intergenic"] / REFERENCE["intergenic"],
    ]
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    x = range(len(names))
    ax.bar(x, ratios, color=[BLUE, CYAN, GOLD, CORAL, CYAN, BLUE, GOLD, CORAL])
    ax.axhline(1, color=NAVY, ls="--", label="held-out reference")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 9)
    ax.set_xticks(list(x), names)
    ax.set_ylabel("Generated / reference ratio (log scale)")
    ax.set_title("Unrestricted sampling restores diversity, not genomic organization", fontsize=18, weight="bold", color=NAVY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    for i, value in enumerate(ratios):
        ax.text(i, value * (1.08 if value >= 1 else 0.91), f"{value:.2f}×", ha="center",
                va="bottom" if value >= 1 else "top", fontsize=9)
    fig.tight_layout()
    return savefig("figure_4_biological_calibration.png")


def memory_probe_figure() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    blocks = [1, 2, 3, 4]
    axes[0].bar(blocks, BLOCK_IMPROVEMENT, color=CYAN)
    axes[0].set_ylabel("Immediate associative-MSE improvement (%)")
    axes[0].set_ylim(0, 13)
    axes[0].set_title("One write improves recall in every block")
    axes[1].bar(blocks, DELAYED_MARGIN, color=GOLD)
    axes[1].axhline(0, color=NAVY, lw=1)
    axes[1].set_ylabel("Shuffled MSE − aligned MSE")
    axes[1].set_title("Pair-specific recall persists after interference")
    for ax in axes:
        ax.set_xlabel("MAC block")
        ax.set_xticks(blocks)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Controlled probes show active associative memory—not biological meaning", fontsize=17, weight="bold", color=NAVY)
    fig.tight_layout()
    return savefig("figure_5_memory_probe.png")


def claim_figure() -> Path:
    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    ax.axis("off")
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 5)
    ax.text(0.3, 4.55, "Evidence ladder: where the present study stops", fontsize=20, weight="bold", color=NAVY)
    levels = [
        ("Established", "finite training · exact resume · active memory · broad decoded diversity", CYAN),
        ("Supported hint", "short-range bacterial statistics · coding-like sequence", BLUE),
        ("Not established", "adaptive-memory benefit · taxonomy · anomaly utility", GOLD),
        ("Prohibited inference", "functional genes · viability · promoter activity · safety", CORAL),
    ]
    for i, (title, text, color) in enumerate(levels):
        y = 3.55 - i * 0.88
        ax.add_patch(FancyBboxPatch((0.45 + i * 0.4, y), 11.1 - i * 0.8, 0.62,
                                    boxstyle="round,pad=.03,rounding_size=.08",
                                    facecolor=color + "20", edgecolor=color))
        ax.text(0.75 + i * 0.4, y + 0.31, title, va="center", weight="bold", color=color)
        ax.text(3.05 + i * 0.4, y + 0.31, text, va="center", color=INK, fontsize=10)
    return savefig("figure_6_evidence_ladder.png")


def make_figures() -> dict[str, Path]:
    FIG.mkdir(parents=True, exist_ok=True)
    return {
        "architecture": architecture_figure(),
        "math": recurrence_figure(),
        "topk": topk_figure(),
        "fidelity": fidelity_figure(),
        "memory": memory_probe_figure(),
        "claims": claim_figure(),
    }


def styles():
    return {
        "title": ParagraphStyle("title", fontName=SANS, fontSize=24, leading=28, textColor=colors.HexColor(NAVY), spaceAfter=12),
        "subtitle": ParagraphStyle("subtitle", fontName=SANS, fontSize=11, leading=15, textColor=colors.HexColor(MUTED), spaceAfter=14),
        "h1": ParagraphStyle("h1", fontName=SANS, fontSize=17, leading=21, textColor=colors.HexColor(NAVY), spaceBefore=14, spaceAfter=7),
        "h2": ParagraphStyle("h2", fontName=SANS, fontSize=12.5, leading=16, textColor=colors.HexColor(BLUE), spaceBefore=9, spaceAfter=4),
        "body": ParagraphStyle("body", fontName=SANS, fontSize=9.4, leading=13.2, textColor=colors.HexColor(INK), alignment=TA_LEFT, spaceAfter=6),
        "small": ParagraphStyle("small", fontName=SANS, fontSize=7.6, leading=10, textColor=colors.HexColor(MUTED), spaceAfter=4),
        "caption": ParagraphStyle("caption", fontName=SANS, fontSize=7.6, leading=10, textColor=colors.HexColor(MUTED), spaceAfter=9),
        "box": ParagraphStyle("box", fontName=SANS, fontSize=9.5, leading=13, textColor=colors.HexColor(NAVY), borderColor=colors.HexColor(CYAN),
                              borderWidth=1, borderPadding=8, backColor=colors.HexColor(PALE), spaceBefore=6, spaceAfter=8),
        "eq": ParagraphStyle("eq", fontName=MONO, fontSize=8.2, leading=12, leftIndent=12, textColor=colors.HexColor(NAVY), spaceAfter=7),
        "ref": ParagraphStyle("ref", fontName=SANS, fontSize=7.5, leading=10, textColor=colors.HexColor(INK), spaceAfter=3),
    }


ST = styles()


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, ST[style])


def T(data, widths=None, header=True, font=7.4):
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), SANS),
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("LEADING", (0, 0), (-1, -1), font + 2),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5df")),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#f6f9fb")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), SANS),
        ]
    table.setStyle(TableStyle(commands))
    return table


def article_header(canvas_obj, doc):
    canvas_obj.saveState()
    canvas_obj.setFont(SANS, 7)
    canvas_obj.setFillColor(colors.HexColor(MUTED))
    canvas_obj.drawString(1.6 * cm, 1.0 * cm, "SeqTrainer Stage C • exploratory preprint • 31 July 2026")
    canvas_obj.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, str(doc.page))
    canvas_obj.restoreState()


def article_story(figs: dict[str, Path]):
    s = []
    s += [
        P("Conditional DNA generation by a compact Titans neural-memory model reveals decoder-hidden diversity but incomplete genomic organization", "title"),
        P("<b>Gonzalo Vidal</b> • SeqTrainer Stage C Study<br/>Exploratory research article; not peer reviewed", "subtitle"),
        P("<b>Abstract.</b> Neural-memory models update a learned function while processing a sequence, potentially combining short-range attention with an adaptive summary of long-past context. We implemented a paper-traceable Memory-as-Context (MAC) Titans model for bacterial DNA and studied a compact 2.52-million-parameter checkpoint trained on 5.0 million valid bases. The exact two-layer functional memory remained finite, improved a controlled associative objective in all four blocks, and achieved held-out 1.9828 bits per base. Conditional generation initially appeared vocabulary-collapsed under top-k 128. A preregistered nine-policy decoding ablation showed that this collapse was largely decoder-induced: removing top-k increased tokenizer-aligned six-mer diversity from 0.285 to 0.988 of reference and reduced six-mer Jensen–Shannon divergence from 0.545 to 0.326 bits. Nevertheless, unrestricted samples remained 4.70 percentage points GC-poor, had 34% more heuristic ORFs, 57% shorter longest ORFs, 33% lower predicted coding density, 32% shorter median genes and sevenfold longer median intergenic spans. Thus the model learned broad short-range bacterial sequence support and stable online memory dynamics, but not yet realistic gene-scale organization. These exploratory results justify a controlled scale-up, not claims of functional sequence generation or adaptive-memory benefit.", "box"),
        Image(str(figs["architecture"]), width=17.5 * cm, height=9.7 * cm),
        P("<b>Figure 1 | Accessible view of the tested architecture.</b> Six-base tokens are embedded, passed through four MAC blocks, and decoded autoregressively. Each block retrieves from stream-specific fast weights before attention and writes once after reading the segment.", "caption"),
        P("Introduction", "h1"),
        P("Transformers retain precise short-term context through attention but incur quadratic cost in context length. Behrouz, Zhong and Mirrokni introduced Titans as a family combining a limited-window attention core, a neural long-term memory and persistent learned tokens. The long-term memory is unusual: its own weights are updated online from an associative objective, so the model can compress past context into parameters while it is running [1]."),
        P("DNA is a stringent test because local composition, codon structure, genes, regulatory intervals and genome-scale organization coexist over different length scales. A generator can look plausible by GC content while failing at higher-order motifs or gene organization. We therefore treat generation as a hierarchy of diagnostics rather than as evidence of biological function."),
        P("Study question", "h2"),
        P("Does a compact, paper-deep Titans MAC implementation (i) train and retrieve stably, (ii) retain broad conditional DNA diversity, and (iii) generate sequence statistics resembling held-out <i>Escherichia coli</i>? The present work is explicitly exploratory and adaptive-only; it cannot estimate the causal benefit of memory."),
        P("Architecture and mathematical correspondence", "h1"),
        Image(str(figs["math"]), width=17.5 * cm, height=7.8 * cm),
        P("<b>Figure 2 | Associative neural-memory recurrence.</b> Prediction error in the inner memory objective produces momentary surprise; momentum carries recent surprise and adaptive forgetting manages finite capacity.", "caption"),
        P("Paper-defined mechanism", "h2"),
        P("For representation x<sub>t</sub>, the Titans paper projects keys, values and queries and trains memory M to associate keys with values:"),
        P(
            "k_t = x_t W_K,   v_t = x_t W_V,   q_t = x_t W_Q<br/>"
            "L_paper(M_(t-1);x_t) = sum_j [M_(t-1)(k_t)_j - v_(t,j)]^2<br/>"
            "S_t = eta_t * S_(t-1) - theta_t * grad_M L_paper(M_(t-1);x_t)<br/>"
            "M_t = (1 - alpha_t) * M_(t-1) + S_t,   retrieval y_t = M_(t-1)(q_t)",
            "eq",
        ),
        P("Here * denotes the paper's element-wise, layer/channel-gated multiplication."),
        P("Here η controls persistence of past surprise, θ controls the present write, and α controls forgetting. The paper’s MAC layout concatenates persistent parameters, the current segment and retrieved history before attention; attention output then writes the long-term memory [1]."),
        P("SeqTrainer c16 realization", "h2"),
        P("The implementation preserves the same key–value objective and recurrence but makes explicit choices where the paper does not uniquely specify an executable path. Its loss is ½‖M(k)−v‖² summed over 128 output channels. The factor ½ leaves the optimum unchanged and removes the factor two from the gradient; learned θ absorbs the scale. Unlike the paper’s chunk approximation, c16 computes exact evolving nonlinear gradients token by token with higher-order autograd and truncates the outer graph at horizon three."),
        T([
            ["Element", "Titans paper", "c16 implementation"],
            ["Memory function", "MLP, depth ≥1", "Residual 2-linear-layer MLP: 128→512→128, GELU + LayerNorm"],
            ["Associative loss", "sum_j [M(k)_j-v_j]^2", "0.5 sum_(j=1)^128 [M(k)_j-v_j]^2"],
            ["Recurrence", "momentum + forgetting", "same recurrence; per-layer/output-channel α, η, θ"],
            ["Initialization", "data-dependent gates", "α=.001, η=.9, θ=.001; learned thereafter"],
            ["Projection context", "learned Q/K/V", "independent Q/K/V + causal depthwise convolution, kernel 4"],
            ["Normalization", "implementation-dependent", "L2-normalized q and k"],
            ["Inner update", "chunkwise acceleration described", "exact evolving nonlinear gradient; no scan approximation"],
            ["Numerics", "not a biological prescription", "FP32 memory; no RMS conditioner; no norm-4 surprise cap"],
        ], widths=[3.1*cm, 5.6*cm, 8.3*cm], font=6.8),
        Spacer(1, 8),
        P("Model tested", "h2"),
        T([
            ["Property", "c16 value"],
            ["Parameters", "2,518,144 trainable"],
            ["Blocks / width / heads", "4 / 128 / 4"],
            ["Memory", "two-layer residual MLP in each block; expansion 4"],
            ["Tokenizer", "non-overlapping six-mer; vocabulary 4,103"],
            ["Segment / gradient horizon", "32 tokens (192 bases) / 3 segments"],
            ["Persistent tokens", "4 per block"],
            ["Functional memory state", "4,235,264 bytes per active stream"],
            ["Backend", "exact accelerated memory; SDPA; FP32"],
            ["Training", "5,000,060 bases, 8,839 optimizer steps, T4 GPU"],
            ["Corpus exposure", "0.000388 passes of 12.87-billion-base corpus"],
        ], widths=[6.1*cm, 10.9*cm]),
        P("Methods", "h1"),
        P("Data and training", "h2"),
        P("Training streams were restricted to the Stage C <i>E. coli</i> plus <i>Escherichia</i> eligibility set, tokenized as non-overlapping six-mers. The model was optimized at learning rate 3×10⁻⁵ with gradient clipping at 0.5 and weight decay 0.1. The 5M budget is only 0.0388% of one predictable-corpus pass; it is a scale gate, not saturation training."),
        P("Held-out and memory analyses", "h2"),
        P("Held-out evaluation covered 38,926 bases in 206 segments from eight streams, but all eight streams belonged to one accession and one ANI99 clade. A controlled 64-pair probe measured immediate reduction in associative reconstruction error and delayed aligned-versus-shuffled recall. Memory-state and hidden-state PCA were exploratory; taxonomic separation could not be evaluated because all plotted streams were <i>E. coli</i>."),
        P("Conditional generation and decoding ablation", "h2"),
        P("Generation used held-out <i>E. coli</i> prompts. The first study generated four 6,144-base continuations per temperature under top-k 128. The follow-up held prompt identity, seed and checkpoint constant while changing one decoding dimension at a time: temperature (0.6–1.1) at k=1,024, top-k (128–unrestricted) at T=0.8, and nucleus p (0.95–1.0) without top-k. Each screen policy used two 3,072-base continuations. Scientific ranking prioritized lower six-mer Jensen–Shannon divergence, then higher aligned-token diversity, then lower GC error."),
        P("Metrics", "h2"),
        P("<b>Bits per base (BPB)</b> is cross-entropy normalized by predictable DNA bases; lower is better. <b>Aligned six-mer diversity</b> is the fraction of distinct non-overlapping tokenizer tokens, reported relative to real continuation diversity. <b>Jensen–Shannon divergence</b> compares k-mer frequency distributions, JSD(P,Q)=½KL(P‖R)+½KL(Q‖R), R=(P+Q)/2; zero is identical and one bit is maximally disjoint under base-2 logs [3]. <b>Prodigal</b> calls coding sequences computationally; its calls are not functional assays [2]. ORF scans are transparent six-frame start-to-stop heuristics."),
        P("Results", "h1"),
        P("Stable learning and active associative memory", "h2"),
        T([
            ["Outcome", "Observed value", "Interpretation"],
            ["Held-out BPB", "1.98278", "finite learning signal; narrow one-accession panel"],
            ["Memory update / surprise", "1.4556 / 0.04434", "non-zero online writes"],
            ["Retrieval / state drift", "110.76 / 21.78", "active finite state; scale-dependent"],
            ["Gradient interventions", "0%", "paper-exact recurrence did not invoke a repair"],
            ["Gate means α / η / θ", ".00118 / .89542 / .00124", "near intended retention/momentum/write regime"],
            ["Immediate probe", "9.60–11.31% better, all blocks", "one write reduces declared associative loss"],
            ["Delayed probe", "positive in 4/4 blocks", "pair-specific recall persists after interference"],
        ], widths=[4.0*cm, 4.0*cm, 9.0*cm], font=7.0),
        Spacer(1, 7),
        Image(str(figs["memory"]), width=17.5 * cm, height=6.45 * cm),
        P("<b>Figure 3 | Controlled memory behavior.</b> Every block improves the inner associative target immediately and retains a small positive aligned-versus-shuffled margin. This validates mechanism, not biological utility.", "caption"),
        P("Decoding, not the model alone, caused the apparent diversity collapse", "h2"),
        Image(str(figs["topk"]), width=17.5 * cm, height=6.6 * cm),
        P("<b>Figure 4 | Top-k ablation.</b> At fixed T=0.8 and p=.99, removing the hard vocabulary cutoff restores almost all reference-level token diversity and reduces six-mer mismatch by 40%.", "caption"),
        T([
            ["top-k", "Aligned diversity / ref", "3-mer JSD", "6-mer JSD", "GC absolute error"],
            ["128", "0.285", "0.1352", "0.5450", "0.0745"],
            ["512", "0.682", "0.0825", "0.4310", "0.0594"],
            ["1,024", "0.822", "0.0616", "0.4010", "0.0646"],
            ["unrestricted", "0.986", "0.0200", "0.3270", "0.0474"],
        ], widths=[2.6*cm, 4.2*cm, 3.1*cm, 3.1*cm, 4.0*cm]),
        P("At k=128, each step retained at most ~3% of the six-mer vocabulary. Unrestricted p=.99 recovered 98.6% of reference aligned diversity. Therefore the 03h collapse cannot be attributed solely to model training. The probability tail beyond rank 1,024 carries substantial, context-dependent sequence diversity."),
        P("Temperature and nucleus sampling", "h2"),
        P("At fixed k=1,024, increasing temperature from 0.6 to 1.1 improved diversity only from 0.802 to 0.842 of reference and six-mer JSD from 0.4055 to 0.3933. Temperature 0.6 was therefore not beneficial. With no top-k, p=.95, .99 and 1.0 yielded diversity ratios .982, .986 and .988 and six-mer JSD .3335, .3270 and .3261. Full p=1.0 sampling is the least biased scientific diagnostic; p=.99 is a nearly equivalent practical policy."),
        P("Unrestricted samples remain genomically miscalibrated", "h2"),
        Image(str(figs["fidelity"]), width=17.5 * cm, height=7.25 * cm),
        P("<b>Figure 5 | Biological calibration of the best decoding policy.</b> Ratios are generated/reference and use a logarithmic axis. Token diversity and gene-call rate match; gene architecture does not.", "caption"),
        T([
            ["Metric", "Reference", "T=.8, no-k, p=1", "Meaning"],
            ["GC fraction", "0.5142", "0.4671", "4.70 percentage points AT-biased"],
            ["Aligned diversity", "0.9219", "0.9111", "0.988× reference; no vocabulary collapse"],
            ["Overlapping diversity", "0.6384", "0.6572", "1.029×; diversity ≠ correctness"],
            ["ORFs ≥90 bp", "28.0", "37.5", "1.34×; excess short coding-like spans"],
            ["Longest ORF", "864 bp", "372 bp", "0.43×; poor long coding coherence"],
            ["Genes / 10 kb", "8.14", "8.14", "same count, only five calls/group"],
            ["Coding density", "0.494", "0.330", "0.67×"],
            ["Median gene", "504 bp", "345 bp", "0.68×"],
            ["Median intergenic", "72 bp", "506 bp", "7.03×"],
        ], widths=[4.0*cm, 3.0*cm, 3.8*cm, 6.2*cm], font=6.9),
        P("Short k-mer distributions were close (JSD₁=.0016, JSD₂=.0080, JSD₃=.0192), but mismatch rose with motif length (JSD₆=.3261). A reference-versus-reference finite-sample baseline is absent, so the absolute excess over natural variation is unknown. Prodigal gene counts matched, but predicted genes were shorter and more widely separated. The generator learned coding-like local patterns, not convincing gene-scale organization."),
        P("Discussion", "h1"),
        P("What the findings demonstrate", "h2"),
        P("The study demonstrates a technically functional deep neural memory: exact nonlinear surprise gradients remain finite, online writes are non-zero, all blocks improve a controlled associative target, and broad decoding modestly increases surprise without destabilizing the state. It also demonstrates that the c16 static model distribution contains broad six-mer support and comparatively accurate one- to four-mer statistics. These are meaningful engineering and exploratory learning results."),
        P("What they do not demonstrate", "h2"),
        Image(str(figs["claims"]), width=17.5 * cm, height=7.0 * cm),
        P("<b>Figure 6 | Claim boundary.</b> The current evidence ends at stable mechanism and conditional sequence diagnostics.", "caption"),
        P("No matched no-memory checkpoint was trained at 5M, so neither BPB nor generation can be causally attributed to adaptive memory. Prodigal calls and ORFs do not establish expression, protein function, promoters, operons, viability, fitness or safety. PCA from one accession cannot establish taxonomic structure. Two-prompt decoding screens do not support confidence intervals or generalization claims. The generated fragments are conditional continuations, not de novo chromosomes."),
        P("Advantages and liabilities", "h2"),
        T([
            ["Advantages observed", "Liabilities observed"],
            ["Expressive two-layer memory with exact, stable updates", "Higher-order gradients and per-stream fast state are computationally costly"],
            ["Memory remains active over long generation", "Benefit over static/no-memory architecture is unmeasured"],
            ["Unrestricted distribution is broad, not collapsed", "Hard top-k can severely and misleadingly suppress diversity"],
            ["Short-range bacterial statistics are learned", "GC and gene/intergenic organization remain miscalibrated"],
            ["Append-only protocol and artifact hashing", "Validation panel is narrow; ledger recorder has a commit-context defect"],
        ], widths=[8.5*cm, 8.5*cm]),
        P("Future work and decision path", "h1"),
        P("First, confirm T=.8, no top-k, p=1.0 with four accessions and 1,024 tokens, adding p=.99 as a practical secondary policy. Compute split-half reference JSD, GC-matched random and six-mer Markov baselines, codon usage, frame periodicity, stop-codon frequency and full gene/intergenic distributions. Second, freeze a clade/accession-stratified held-out panel. Third, qualify the 24.8M-parameter Medium model (12 blocks, d=256, eight heads) on A100 with FP32 functional memory and an empirically selected batch. Fourth, train one adaptive-only 25M discovery run with gates at 5M, 10M and 25M. Extend to 50M only if diverse-panel BPB, GC, k-mer JSD and gene organization improve while memory remains finite. Finally, only after the adaptive model is useful should matched no-memory, frozen-memory, shallow-memory and second-seed controls be run; only those controls can support a memory-benefit claim."),
        P("Data and code availability", "h1"),
        P("The checkpoint SHA-256 is <font name='ArticleMono'>21898362291f4fd1e6aafcfbe47e8b05dbe69e5c8036e6ae7927a6ac24ac4541</font>. Drive-backed ledger events record the 5M run, first generation study and decoding ablation. Generation code was pinned to commits fc36e9f287240b3ad3367a092a7983f9376bff5b and 6d08cd8a3128d4d61a7e67ab20eebaa0dbff48b5. The article package is generated from source under <font name='ArticleMono'>docs/stage_c_c16_article</font>."),
        P("References", "h1"),
        P("[1] Behrouz A, Zhong P, Mirrokni V. Titans: Learning to Memorize at Test Time. <i>Advances in Neural Information Processing Systems</i> 39 (2025). https://proceedings.neurips.cc/paper_files/paper/2025/file/a4ca07aa108036f80cbb5b82285fd4b1-Paper-Conference.pdf", "ref"),
        P("[2] Hyatt D et al. Prodigal: prokaryotic gene recognition and translation initiation site identification. <i>BMC Bioinformatics</i> 11, 119 (2010). doi:10.1186/1471-2105-11-119.", "ref"),
        P("[3] Lin J. Divergence measures based on the Shannon entropy. <i>IEEE Transactions on Information Theory</i> 37, 145–151 (1991).", "ref"),
        P("[4] Vaswani A et al. Attention Is All You Need. <i>Advances in Neural Information Processing Systems</i> 30 (2017).", "ref"),
        P("[5] SeqTrainer Stage C paper-deep implementation decision and frozen protocol, repository evidence dated 26–31 July 2026.", "ref"),
        P("Reporting note", "h1"),
        P("This manuscript follows an evidence-tiered reporting policy. Exact engineering, held-out prediction, controlled associative behavior, representation exploration and generation diagnostics are separated. No p-values are reported because the decoding screen contains only two paired prompts and was not designed for inferential testing."),
    ]
    return s


def build_article(figs):
    path = OUT / "C16_Titans_DNA_Generation_Research_Article.pdf"
    doc = BaseDocTemplate(str(path), pagesize=A4, leftMargin=1.6*cm, rightMargin=1.6*cm,
                          topMargin=1.5*cm, bottomMargin=1.5*cm,
                          title="Conditional DNA generation by a compact Titans neural-memory model",
                          author="Gonzalo Vidal")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="article", frames=[frame], onPage=article_header)])
    doc.build(article_story(figs))
    return path


def slide_text(c, text, x, y, width, size=20, color=INK, bold=False):
    if not color.startswith("#"):
        color = "#" + color
    style = ParagraphStyle("slide", fontName=SANS, fontSize=size, leading=size*1.22,
                           textColor=colors.HexColor(color), alignment=TA_LEFT)
    if bold:
        text = f"<b>{text}</b>"
    p = Paragraph(text, style)
    _, h = p.wrap(width, 1000)
    p.drawOn(c, x, y-h)
    return h


def slide_base(c, title, number):
    W, H = landscape(A4)
    c.setFillColor(colors.white)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(colors.HexColor(NAVY))
    c.rect(0, H-1.55*cm, W, 1.55*cm, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont(SANS, 22)
    c.drawString(0.7*cm, H-1.02*cm, title)
    c.setFont(SANS, 8)
    c.setFillColor(colors.HexColor(MUTED))
    c.drawRightString(W-0.6*cm, 0.35*cm, f"SeqTrainer Stage C • {number}")
    return W, H


def add_image(c, path, x, y, w, h):
    c.drawImage(str(path), x, y, w, h, preserveAspectRatio=True, anchor="c", mask="auto")


def build_slides(figs):
    path = OUT / "C16_Titans_DNA_Generation_Presentation.pdf"
    W, H = landscape(A4)
    c = canvas.Canvas(str(path), pagesize=(W, H), pageCompression=1)
    c.setTitle("C16 Titans DNA generation findings")
    # 1
    c.setFillColor(colors.HexColor(NAVY))
    c.rect(0, 0, W, H, fill=1, stroke=0)
    slide_text(c, "A compact Titans neural-memory model for DNA", 1.2*cm, H-2.3*cm, W-2.4*cm, 31, "ffffff", True)
    slide_text(c, "Decoder-hidden diversity, stable online memory—and incomplete genomic organization", 1.2*cm, H-4.9*cm, W-2.4*cm, 20, "b9dbe8")
    slide_text(c, "c16 exploratory study • 5M training bases • 2.52M parameters • July 2026", 1.2*cm, 2.0*cm, W-2.4*cm, 12, "ffffff")
    c.showPage()
    # 2
    slide_base(c, "The question", 2)
    slide_text(c, "Can this compact paper-deep Titans model:", 1.0*cm, H-2.4*cm, 12.2*cm, 18, NAVY, True)
    bullets = ["train with exact online neural-memory updates?", "retain broad DNA sequence diversity?", "generate E. coli-like local and gene-scale organization?", "justify a larger adaptive-only discovery run?"]
    y = H-4.2*cm
    for b in bullets:
        slide_text(c, "• "+b, 1.2*cm, y, 11.5*cm, 17, INK)
        y -= 1.35*cm
    slide_text(c, "Boundary: no matched 5M no-memory control → no memory-benefit claim.", 14.2*cm, H-4.0*cm, 12*cm, 18, CORAL, True)
    c.showPage()
    # 3
    slide_base(c, "Architecture in one picture", 3)
    add_image(c, figs["architecture"], 0.7*cm, 0.9*cm, W-1.4*cm, H-2.7*cm)
    c.showPage()
    # 4
    slide_base(c, "The mathematics: remember what was surprising", 4)
    add_image(c, figs["math"], 0.8*cm, 2.0*cm, W-1.6*cm, H-4.0*cm)
    slide_text(c, "Paper recurrence preserved; c16 uses exact evolving nonlinear gradients, per-channel gates, and no surprise clipping.", 1.0*cm, 1.5*cm, W-2*cm, 13, MUTED)
    c.showPage()
    # 5
    slide_base(c, "The model tested", 5)
    items = [("2.52M", "parameters"), ("4 ×", "MAC blocks"), ("128", "hidden width"), ("2-layer", "residual MLP memory"),
             ("5.0M", "training bases"), ("0.0388%", "of one corpus pass"), ("1.9828", "held-out BPB"), ("T4", "training GPU")]
    for i, (big, small) in enumerate(items):
        col, row = i % 4, i // 4
        x, y = 1.0*cm + col*6.8*cm, H-3.0*cm-row*5.1*cm
        c.setFillColor(colors.HexColor(PALE))
        c.roundRect(x, y-2.5*cm, 5.7*cm, 3.2*cm, 8, fill=1, stroke=0)
        slide_text(c, big, x+0.35*cm, y, 5.0*cm, 24, BLUE, True)
        slide_text(c, small, x+0.35*cm, y-1.25*cm, 5.0*cm, 13, MUTED)
    c.showPage()
    # 6
    slide_base(c, "Neural memory is active and stable", 6)
    add_image(c, figs["memory"], 0.8*cm, 1.2*cm, 17.2*cm, H-3.0*cm)
    slide_text(c, "0%", 19.2*cm, H-3.0*cm, 7*cm, 32, CYAN, True)
    slide_text(c, "gradient interventions", 19.2*cm, H-4.3*cm, 7*cm, 15, MUTED)
    slide_text(c, "4 / 4", 19.2*cm, H-6.4*cm, 7*cm, 32, GOLD, True)
    slide_text(c, "blocks retain delayed association", 19.2*cm, H-7.7*cm, 7*cm, 15, MUTED)
    slide_text(c, "Mechanism works ≠ memory improves genomics.", 19.2*cm, H-10.3*cm, 7*cm, 15, CORAL, True)
    c.showPage()
    # 7
    slide_base(c, "Key result: top-k created the apparent collapse", 7)
    add_image(c, figs["topk"], 0.8*cm, 1.2*cm, W-1.6*cm, H-3.0*cm)
    c.showPage()
    # 8
    slide_base(c, "Temperature mattered less than vocabulary truncation", 8)
    slide_text(c, "At k=1,024 and p=.99", 1.0*cm, H-2.5*cm, 10*cm, 18, NAVY, True)
    temp_data = [["T", "diversity/ref", "6-mer JSD"], [".6", ".802", ".4055"], [".8", ".822", ".4010"], ["1.0", ".838", ".3939"], ["1.1", ".842", ".3933"]]
    table = T(temp_data, widths=[2.2*cm, 4.0*cm, 3.5*cm], font=11)
    table.wrapOn(c, 10*cm, 10*cm)
    table.drawOn(c, 1.0*cm, H-10.5*cm)
    slide_text(c, "Temperature 0.6 was not advantageous.", 1.0*cm, 2.3*cm, 10*cm, 16, CORAL, True)
    slide_text(c, "Without top-k", 14.0*cm, H-2.5*cm, 10*cm, 18, NAVY, True)
    slide_text(c, "p=.95 → 6-mer JSD .3335<br/>p=.99 → .3270<br/>p=1.0 → .3261", 14.0*cm, H-4.0*cm, 10*cm, 19, INK)
    slide_text(c, "Freeze scientific diagnostic:<br/><b>T=.8 · no top-k · p=1.0</b>", 14.0*cm, H-8.5*cm, 11*cm, 19, CYAN)
    c.showPage()
    # 9
    slide_base(c, "Diversity recovered; genome organization did not", 9)
    add_image(c, figs["fidelity"], 0.7*cm, 1.0*cm, W-1.4*cm, H-2.7*cm)
    c.showPage()
    # 10
    slide_base(c, "What the best samples look like statistically", 10)
    data = [
        ["Metric", "reference", "generated", "assessment"],
        ["GC", "51.42%", "46.71%", "AT-biased"],
        ["aligned diversity", ".9219", ".9111", "near reference"],
        ["ORFs ≥90 bp", "28", "37.5", "too many"],
        ["longest ORF", "864 bp", "372 bp", "too short"],
        ["Prodigal genes / 10 kb", "8.14", "8.14", "count matches; n=5"],
        ["coding density", ".494", ".330", "33% lower"],
        ["median gene", "504 bp", "345 bp", "32% shorter"],
        ["median intergenic", "72 bp", "506 bp", "7× longer"],
    ]
    table = T(data, widths=[5*cm, 4*cm, 4*cm, 10*cm], font=10)
    table.wrapOn(c, W-2*cm, H)
    table.drawOn(c, 1*cm, 1.5*cm)
    c.showPage()
    # 11
    slide_base(c, "What we may—and may not—claim", 11)
    add_image(c, figs["claims"], 0.7*cm, 1.0*cm, W-1.4*cm, H-2.7*cm)
    c.showPage()
    # 12
    slide_base(c, "Next decision path", 12)
    path_items = [
        ("1", "Confirm decoding", "4 prompts × 1,024 tokens; reference-v-reference baseline"),
        ("2", "Freeze diverse panel", "clade/accession-stratified validation"),
        ("3", "A100 qualification", "24.8M parameters; 12 blocks × d256"),
        ("4", "Adaptive-only 25M", "evaluate at 5M, 10M and 25M"),
        ("5", "Controls only if useful", "no-memory, frozen, shallow, second seed"),
    ]
    for i, (n, title, sub) in enumerate(path_items):
        x = 0.8*cm + i*5.5*cm
        c.setFillColor(colors.HexColor([CYAN, BLUE, GOLD, CORAL, NAVY][i]))
        c.circle(x+0.55*cm, H-3.2*cm, 0.48*cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont(SANS, 15)
        c.drawCentredString(x+0.55*cm, H-3.38*cm, n)
        slide_text(c, title, x, H-4.2*cm, 4.7*cm, 15, NAVY, True)
        slide_text(c, sub, x, H-5.5*cm, 4.7*cm, 11, MUTED)
        if i < 4:
            c.setStrokeColor(colors.HexColor("#aab8c4"))
            c.line(x+1.1*cm, H-3.2*cm, x+5.25*cm, H-3.2*cm)
    slide_text(c, "Scale is justified as a controlled discovery—not as a whole-corpus or functional-generation claim.", 1.0*cm, 2.1*cm, W-2*cm, 18, CORAL, True)
    c.showPage()
    # 13
    c.setFillColor(colors.HexColor(NAVY))
    c.rect(0, 0, W, H, fill=1, stroke=0)
    slide_text(c, "Take-home message", 1.2*cm, H-2.5*cm, W-2.4*cm, 29, "ffffff", True)
    slide_text(c, "The compact model learned broad short-range bacterial sequence support and stable online memory dynamics.", 1.2*cm, H-4.5*cm, W-2.4*cm, 21, "b9dbe8")
    slide_text(c, "It has not yet learned convincing gene-scale organization—and adaptive memory has not yet been shown to cause a benefit.", 1.2*cm, H-7.2*cm, W-2.4*cm, 21, "f4c7bf")
    slide_text(c, "Qualified green light: confirm, stratify, then scale to the 25M Medium discovery gate.", 1.2*cm, 2.0*cm, W-2.4*cm, 18, "ffffff", True)
    c.showPage()
    c.save()
    return path


def write_data_tables():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["policy,temperature,top_k,top_p,gc_abs_error,aligned_diversity_ratio,jsd_3mer,jsd_6mer,genes_10kb_ratio,coding_density_ratio"]
    for row in POLICIES:
        lines.append(",".join(map(str, row)))
    (OUT / "decoding_ablation_table.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "DATA_PROVENANCE.md").write_text(
        """# Data provenance

- Checkpoint SHA-256: `21898362291f4fd1e6aafcfbe47e8b05dbe69e5c8036e6ae7927a6ac24ac4541`
- Dataset fingerprint: `2fbdb870606e6fb1ce0f6750524726d041474be5081da2cb2316948bc12a2ee9`
- Study protocol hash: `41587632b18790f761e9412617f284646098ae843701f6b2113991831ccc439e`
- Ledger run IDs: `adaptive_exploration_5m`, `adaptive_exploration_5m_generation`,
  `adaptive_exploration_5m_decoding_ablation`
- Generation implementation commit: `fc36e9f287240b3ad3367a092a7983f9376bff5b`
- Decoding-ablation implementation commit: `6d08cd8a3128d4d61a7e67ab20eebaa0dbff48b5`

The PDFs are exploratory scientific communication artifacts. Drive files were
verified against ledger SHA-256 values before transcription into the build
script.
""",
        encoding="utf-8",
    )


def validate_pdf(path: Path, minimum_pages: int):
    reader = PdfReader(str(path))
    if len(reader.pages) < minimum_pages:
        raise RuntimeError(f"{path.name} has only {len(reader.pages)} pages")
    for index, page in enumerate(reader.pages):
        box = page.mediabox
        if float(box.width) <= 0 or float(box.height) <= 0:
            raise RuntimeError(f"{path.name} page {index + 1} has invalid geometry")
    return len(reader.pages)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figs = make_figures()
    write_data_tables()
    article = build_article(figs)
    slides = build_slides(figs)
    article_pages = validate_pdf(article, 7)
    slide_pages = validate_pdf(slides, 12)
    manifest = (
        f"article={article.name}\narticle_pages={article_pages}\n"
        f"article_sha256={sha256(article)}\n"
        f"slides={slides.name}\nslide_pages={slide_pages}\n"
        f"slides_sha256={sha256(slides)}\nfigures={len(figs)}\n"
    )
    for name, path in sorted(figs.items()):
        manifest += f"figure_{name}_sha256={sha256(path)}\n"
    (OUT / "BUILD_MANIFEST.txt").write_text(manifest, encoding="utf-8")
    print(manifest)


if __name__ == "__main__":
    main()
