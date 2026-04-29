"""Common SBOL query recipes."""

from .builder import build_select_query


def sequence_query() -> str:
    """Return a recipe query that extracts SBOL sequence elements."""
    return build_select_query(
        fields=["?sequence"],
        where_lines=["?s sbol:elements ?sequence ."],
    )
