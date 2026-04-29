"""SynBioHub client abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class SynBioHubClient:
    """Lightweight client for querying SynBioHub-compatible endpoints."""

    base_url: str
    timeout: int = 30
    sparql_path: str = "/sparql"

    def run_sparql(self, query: str) -> dict[str, Any] | list[Any]:
        """Run a SPARQL query and return decoded JSON response."""
        url = f"{self.base_url.rstrip('/')}{self.sparql_path}"
        response = requests.post(
            url,
            data={"query": query},
            headers={"Accept": "application/sparql-results+json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def fetch_sbol(self, uri: str) -> str | bytes:
        """Fetch SBOL XML for a given resource URI."""
        response = requests.get(uri, timeout=self.timeout)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "text" in content_type or "xml" in content_type:
            return response.text
        return response.content
