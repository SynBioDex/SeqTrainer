"""Simple SPARQL query builder helpers."""

from .prefixes import format_prefixes


def build_select_query(*, fields: list[str], where_lines: list[str], prefixes: dict[str, str] | None = None) -> str:
    """Build a compact SELECT query from fields and where clauses."""
    fields_str = " ".join(fields)
    where = "\n  ".join(where_lines)
    return f"{format_prefixes(prefixes)}\n\nSELECT {fields_str}\nWHERE {{\n  {where}\n}}"
