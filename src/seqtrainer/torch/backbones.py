"""Backbone presets for transformer-based DNA modeling in PyTorch."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HuggingFaceBackbone:
    """Descriptor for a Hugging Face transformer backbone."""

    name: str
    model_id: str
    tokenizer_id: str
    max_length: int = 512
    trust_remote_code: bool = True


def nucleotide_transformer_v2(*, variant: str = "500m_human_ref") -> HuggingFaceBackbone:
    """Return a Nucleotide Transformer v2 backbone descriptor.

    Parameters
    ----------
    variant:
        Model variant suffix hosted by InstaDeepAI, for example
        ``"500m_human_ref"`` or ``"100m_multi_species_v2"``.
    """

    model_id = f"InstaDeepAI/nucleotide-transformer-v2-{variant}"
    return HuggingFaceBackbone(
        name=f"nucleotide-transformer-v2:{variant}",
        model_id=model_id,
        tokenizer_id=model_id,
        max_length=1024,
    )
