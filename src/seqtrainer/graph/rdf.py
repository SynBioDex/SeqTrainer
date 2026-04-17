"""Graph conversion helpers for SBOL/RDF workflows."""

from __future__ import annotations

from pathlib import Path

from rdflib import Graph


def xml_to_nt(input_path: str | Path, output_path: str | Path) -> None:
    """Convert SBOL XML to N-Triples for graph tooling."""
    graph = Graph()
    graph.parse(str(input_path), format="xml")
    graph.serialize(destination=str(output_path), format="nt")
