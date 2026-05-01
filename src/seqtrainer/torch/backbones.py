"""Backbone presets for DNA foundation models in PyTorch."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HuggingFaceBackbone:
    """Descriptor for a Hugging Face backbone."""

    name: str
    model_id: str
    tokenizer_id: str
    max_length: int = 512
    trust_remote_code: bool = True


def nucleotide_transformer_v2(*, variant: str = "500m_human_ref") -> HuggingFaceBackbone:
    """Return a Nucleotide Transformer v2 backbone descriptor."""

    model_id = f"InstaDeepAI/nucleotide-transformer-v2-{variant}"
    return HuggingFaceBackbone(
        name=f"nucleotide-transformer-v2:{variant}",
        model_id=model_id,
        tokenizer_id=model_id,
        max_length=1024,
    )


def hyena_dna(*, variant: str = "medium-160k") -> HuggingFaceBackbone:
    """Return a HyenaDNA/Hyena2-style backbone descriptor.

    Parameters
    ----------
    variant:
        Supported aliases:
        - ``"tiny-1k"`` -> ``LongSafari/hyenadna-tiny-1k-seqlen-hf``
        - ``"small-32k"`` -> ``LongSafari/hyenadna-small-32k-seqlen-hf``
        - ``"medium-160k"`` -> ``LongSafari/hyenadna-medium-160k-seqlen-hf``
    """

    variants = {
        "tiny-1k": ("LongSafari/hyenadna-tiny-1k-seqlen-hf", 1024),
        "small-32k": ("LongSafari/hyenadna-small-32k-seqlen-hf", 4096),
        "medium-160k": ("LongSafari/hyenadna-medium-160k-seqlen-hf", 8192),
    }
    if variant not in variants:
        valid = ", ".join(sorted(variants))
        raise ValueError(f"Unknown HyenaDNA variant '{variant}'. Valid options: {valid}")

    model_id, max_length = variants[variant]
    return HuggingFaceBackbone(
        name=f"hyena-dna:{variant}",
        model_id=model_id,
        tokenizer_id=model_id,
        max_length=max_length,
    )
