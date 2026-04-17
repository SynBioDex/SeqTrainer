"""Promoter regression application wiring."""

from __future__ import annotations

from seqtrainer.data.recipes import DatasetRecipe
from seqtrainer.sparql.recipes import sequence_query


def build_promoter_regression_blueprint(framework: str = "torch") -> dict:
    """Describe the recipe, transforms, and adapter path for promoter regression."""
    recipe = DatasetRecipe(
        name="promoter-regression",
        query=sequence_query(),
        sequence_field="sequence",
        label_field="target",
        metadata_fields=("source",),
        provenance={"task": "promoter_regression", "framework": framework},
    )
    return {
        "recipe": recipe,
        "transforms": [
            "seqtrainer.transforms.dna.normalize_sequence",
            "seqtrainer.transforms.dna.pad_or_trim",
            "seqtrainer.transforms.dna.one_hot_encode",
        ],
        "model_factory": "seqtrainer.models.registry.ModelRegistry",
        "framework_adapter": f"seqtrainer.{framework}.adapters",
    }
