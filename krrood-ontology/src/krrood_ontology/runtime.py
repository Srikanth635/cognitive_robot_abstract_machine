"""Runtime mappings and the identity registry for a loaded A-Box."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from krrood.entity_query_language.predicate import Symbol
from rdflib import Graph


@dataclass(frozen=True)
class FieldMapping:
    """A generated field's RDF signature and collection representation."""

    iri: str
    name: str
    domain_iri: str
    range_iri: str
    kind: str
    container: str


@dataclass(frozen=True)
class GeneratedModel:
    """Describe the generated classes accepted by an individual loader."""

    classes: dict[str, type[Symbol]]
    properties: dict[str, FieldMapping]
    annotation_properties: frozenset[str] = frozenset()
    bindings: tuple[FieldMapping, ...] = ()
    binding_policy: str = "explicit"
    entity_base: type[Symbol] | None = None

    def mapping_for(
        self, property_iri: str, class_iri: str, *, require_applicable: bool = False
    ) -> FieldMapping | None:
        """Resolve a native field association without inventing an OWL domain."""
        if self.binding_policy == "explicit":
            mapping = self.properties.get(property_iri)
            if (
                require_applicable
                and mapping
                and mapping.domain_iri
                and not issubclass(
                    self.classes[class_iri], self.classes[mapping.domain_iri]
                )
            ):
                return None
            return mapping
        concrete = self.classes[class_iri]
        applicable = [
            b
            for b in self.bindings
            if b.iri == property_iri
            and issubclass(
                concrete,
                (self.classes[b.domain_iri] if b.domain_iri else self.entity_base),
            )
        ]
        return (
            min(
                applicable,
                key=lambda b: concrete.__mro__.index(
                    (self.classes[b.domain_iri] if b.domain_iri else self.entity_base)
                ),
            )
            if applicable
            else None
        )


@dataclass(frozen=True)
class LoadedIndividuals:
    """One object per named individual IRI in a single successful load."""

    by_iri: dict[str, Symbol]
    asserted_types: dict[str, tuple[str, ...]]
    annotations: dict[str, dict[str, tuple[Any, ...]]]
    source_sha256: str
    asserted_graph: Graph
    identity_policy: str = "reject"
    distinct_groups: tuple[frozenset[str], ...] = ()
    source_bytes: bytes | None = None
    source_format: str | None = None

    def __getitem__(self, iri: str) -> Symbol:
        """Resolve an individual without normalizing or merging its IRI."""
        return self.by_iri[iri]


class IndividualLoadingError(ValueError):
    """An unsupported or invalid assertion found before object construction."""

    def __init__(self, code: str, message: str, iri: str | None = None) -> None:
        self.code = code
        self.iri = iri
        super().__init__(f"{code}: {message}" + (f" [{iri}]" if iri else ""))
