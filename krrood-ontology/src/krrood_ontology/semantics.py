"""Plan supported relation axioms for existing KRROOD descriptors."""

from __future__ import annotations

from dataclasses import dataclass, replace
from graphlib import CycleError, TopologicalSorter
from typing import NoReturn

from rdflib import OWL, RDF, RDFS

from krrood_ontology.model import (
    CompilationDiagnostic,
    DescriptorDefinition,
    ModelDefinition,
    OntologyTranslationError,
    PropertyKind,
)
from krrood_ontology.ontology import OntologyDocument


@dataclass
class RelationSemantics:
    """Translate inverse, symmetric, transitive, and single-parent subproperties."""

    def translate(
        self, document: OntologyDocument, model: ModelDefinition
    ) -> ModelDefinition:
        """Select a tested descriptor mapping; do not implement implication rules."""
        graph = document.graph
        properties = {
            prop.iri: prop
            for prop in model.properties
            if prop.kind is PropertyKind.OBJECT
        }
        names = {
            iri: f"_Property{index}" for index, iri in enumerate(sorted(properties))
        }
        inverse: dict[str, str] = {}
        transitive = {
            str(term) for term in graph.subjects(RDF.type, OWL.TransitiveProperty)
        }
        symmetric = {
            str(term) for term in graph.subjects(RDF.type, OWL.SymmetricProperty)
        }
        for iri in sorted(transitive | symmetric):
            if iri not in properties:
                self._fail(
                    "semantic_property_kind",
                    "Relation characteristics require a declared object property",
                    iri,
                )
            prop = properties[iri]
            if prop.domain_iri != prop.range_iri:
                self._fail(
                    "semantic_signature",
                    "Transitive and symmetric properties require the same domain and range in this profile",
                    iri,
                )
        for subject, target in graph.subject_objects(OWL.inverseOf):
            left, right = str(subject), str(target)
            if left not in properties or right not in properties:
                self._fail(
                    "inverse_target",
                    "Both inverse properties must be declared object properties",
                    left,
                )
            first, second = properties[left], properties[right]
            if (first.domain_iri, first.range_iri) != (
                second.range_iri,
                second.domain_iri,
            ):
                self._fail(
                    "inverse_signature",
                    "Inverse properties require exactly swapped domains and ranges",
                    left,
                )
            self._bind_inverse(inverse, left, right)
            self._bind_inverse(inverse, right, left)
        for iri in sorted(symmetric):
            self._bind_inverse(inverse, iri, iri)
        parents: dict[str, str | None] = dict.fromkeys(properties)
        for subject, target in graph.subject_objects(RDFS.subPropertyOf):
            child, parent = str(subject), str(target)
            if child not in properties or parent not in properties:
                self._fail(
                    "subproperty_kind",
                    "Subproperty axioms require declared object properties",
                    child,
                )
            if parents[child] is not None and parents[child] != parent:
                self._fail(
                    "multiple_superproperties",
                    "Only one named superproperty is supported",
                    child,
                )
            first, second = properties[child], properties[parent]
            if (first.domain_iri, first.range_iri) != (
                second.domain_iri,
                second.range_iri,
            ):
                self._fail(
                    "subproperty_signature",
                    "Subproperties require identical explicit domains and ranges in this profile",
                    child,
                )
            parents[child] = parent
        hierarchy = {
            iri: (parents[iri],) if parents[iri] else () for iri in sorted(properties)
        }
        try:
            order = tuple(TopologicalSorter(hierarchy).static_order())
        except CycleError as error:
            raise OntologyTranslationError(
                (
                    CompilationDiagnostic(
                        "subproperty_cycle",
                        "Cyclic subproperty axioms cannot become descriptor inheritance",
                    ),
                )
            ) from error
        for iri in order:
            parent = parents[iri]
            while parent is not None:
                if parent in transitive:
                    self._fail(
                        "transitive_superproperty",
                        "KRROOD descriptor inheritance would over-strengthen this subproperty; a transitive ancestor is unsupported",
                        iri,
                    )
                parent = parents[parent]
        affected = transitive | symmetric | set(inverse)
        for child, parent in parents.items():
            if parent is not None:
                affected.update((child, parent))
        updated_properties = tuple(
            (
                replace(prop, container="set", constructor=False)
                if prop.iri in affected
                else prop
            )
            for prop in model.properties
        )
        updated = {prop.iri: prop for prop in updated_properties}
        classes = tuple(
            replace(
                concept,
                properties=tuple(updated[prop.iri] for prop in concept.properties),
            )
            for concept in model.classes
        )
        descriptors = tuple(
            DescriptorDefinition(
                property=updated[iri],
                name=names[iri],
                parent_name=names[parents[iri]] if parents[iri] else None,
                parent_iri=parents[iri],
                inverse_name=names[inverse[iri]] if iri in inverse else None,
                inverse_iri=inverse.get(iri),
                transitive=iri in transitive,
                symmetric=iri in symmetric,
            )
            for iri in order
        )
        return replace(
            model,
            classes=classes,
            properties=updated_properties,
            descriptors=descriptors,
        )

    def _bind_inverse(self, inverse: dict[str, str], left: str, right: str) -> None:
        if left in inverse and inverse[left] != right:
            self._fail(
                "multiple_inverses",
                "Only one inverse target per property is supported",
                left,
            )
        inverse[left] = right

    @staticmethod
    def _fail(code: str, message: str, iri: str | None = None) -> NoReturn:
        raise OntologyTranslationError((CompilationDiagnostic(code, message, iri),))
