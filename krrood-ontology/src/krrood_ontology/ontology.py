"""RDFLib-backed loading without changing the asserted graph."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from rdflib import Graph
from rdflib.util import guess_format

from krrood_ontology.config import OntologyInput, RDFFormat


@dataclass(frozen=True)
class OntologyDocument:
    """An asserted RDF graph with its source and content fingerprint."""

    source: OntologyInput
    graph: Graph
    sha256: str
    format: RDFFormat

    @classmethod
    def load(cls, source: OntologyInput) -> OntologyDocument:
        """Parse a local file; retain imports as assertions without fetching them."""
        detected = source.format
        if detected is None:
            syntax = guess_format(source.path.name)
            if syntax is None and source.path.suffix.lower() == ".owl":
                syntax = RDFFormat.XML.value
            if syntax not in {member.value for member in RDFFormat}:
                raise ValueError("Unknown RDF syntax; specify --format explicitly")
            detected = RDFFormat(syntax)
        content = source.path.read_bytes()
        graph = Graph().parse(
            data=content, format=detected.value, publicID=source.path.as_uri()
        )
        return cls(source, graph, hashlib.sha256(content).hexdigest(), detected)
