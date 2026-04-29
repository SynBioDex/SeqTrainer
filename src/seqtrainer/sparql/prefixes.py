"""Reusable SPARQL prefixes for SBOL/SynBioHub queries."""

DEFAULT_PREFIXES = {
    "sbol": "http://sbols.org/v2#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "om": "http://www.ontology-of-units-of-measure.org/resource/om-2/",
}


def format_prefixes(prefixes: dict[str, str] | None = None) -> str:
    """Render SPARQL PREFIX declarations."""
    selected = prefixes or DEFAULT_PREFIXES
    return "\n".join(f"PREFIX {name}: <{uri}>" for name, uri in selected.items())
