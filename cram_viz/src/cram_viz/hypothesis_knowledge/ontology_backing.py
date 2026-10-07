"""
The ontology terms behind a projected hypothesis: the class of every fragment, the
property stating every link, and where every value a fragment holds comes from.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import StrEnum
from operator import attrgetter
from pathlib import Path

from llm_reasoner.generation.schema import is_inferred
from llm_reasoner.generation.wire import CONCEPT_PREFIXES
from llm_reasoner.hypotheses.graph import HypothesisGraphNode, HypothesisGraphRelation
from llm_reasoner.ontologies import (
    CRAM_AICOR_NAMESPACE,
    DUL_NAMESPACE,
    HYPOTHESIS_CONCEPT_NAMESPACE,
    PREDICTIVE_MANIPULATION_NAMESPACE_ROOT,
    PredictiveManipulationOntologyFile,
)
from llm_reasoner.ontologies.terms import (
    ConceptReference,
    DescriptionEntity,
    PropertyTerm,
)
from typing_extensions import Any, Optional

DRAWN_LINK_NOTE = "(drawn by the viewer)"
"""
The note shown in place of an ontology term on what only the viewer draws.
"""

# %% namespaces


class TermSource(StrEnum):
    """
    The ontology a term or value comes from.
    """

    PREDICTIVE_MANIPULATION = "predictive manipulation"
    """
    One of the predictive manipulation modules.
    """

    CRAM_AICOR = "CRAM-AICOR"
    """
    The CRAM-AICOR ontology.
    """

    DUL = "DUL"
    """
    DOLCE+DnS Ultralite.
    """

    HYPOTHESIS = "introduced by the hypothesis"
    """
    A concept the hypothesis introduced because no linked ontology declares it.
    """

    LITERAL = "literal"
    """
    A data value, such as a number.
    """

    UNKNOWN = "outside the known ontologies"
    """
    A term of no namespace the viewer knows.
    """


@dataclass(frozen=True)
class TermNamespace:
    """
    A namespace and the prefix it is written with.
    """

    prefix: str
    """
    The prefix standing for the namespace.
    """

    iri: str
    """
    The namespace IRI every term of it starts with.
    """

    source: TermSource
    """
    The ontology the namespace belongs to.
    """


def known_namespaces() -> tuple[TermNamespace, ...]:
    """
    :return: The namespace of every predictive manipulation module, prefixed with the
        module's name, and of every ontology a hypothesis may cite a concept from.
    """
    source_of_concept_namespace = {
        CRAM_AICOR_NAMESPACE: TermSource.CRAM_AICOR,
        DUL_NAMESPACE: TermSource.DUL,
        HYPOTHESIS_CONCEPT_NAMESPACE: TermSource.HYPOTHESIS,
    }
    return (
        *(
            TermNamespace(
                prefix=Path(module).stem,
                iri=PREDICTIVE_MANIPULATION_NAMESPACE_ROOT + Path(module).stem + "#",
                source=TermSource.PREDICTIVE_MANIPULATION,
            )
            for module in PredictiveManipulationOntologyFile
        ),
        *(
            TermNamespace(
                prefix=prefix,
                iri=namespace,
                source=source_of_concept_namespace[namespace],
            )
            for prefix, namespace in CONCEPT_PREFIXES.items()
        ),
    )


# %% backing


@dataclass(frozen=True)
class BackedValue:
    """
    A value a fragment holds, written for display, with the ontology it comes from.
    """

    text: str
    """
    The value as displayed.
    """

    source: TermSource
    """
    The ontology the value comes from.
    """


@dataclass(frozen=True)
class OntologyBacking:
    """
    Names the ontology terms behind the nodes and links of a projected hypothesis.
    """

    namespaces: tuple[TermNamespace, ...] = field(default_factory=known_namespaces)
    """
    The namespaces terms are written and attributed with.
    """

    def namespace_of(self, iri: str) -> Optional[TermNamespace]:
        """
        :return: The namespace a term belongs to, or ``None`` when none is known.
        """
        return next(
            (
                namespace
                for namespace in self.namespaces
                if iri.startswith(namespace.iri)
            ),
            None,
        )

    def prefixed(self, iri: str) -> str:
        """
        :return: A term written with its namespace's prefix, or in full without one.
        """
        namespace = self.namespace_of(iri)
        if namespace is None:
            return iri
        return f"{namespace.prefix}:{iri.removeprefix(namespace.iri)}"

    def source_of(self, iri: str) -> TermSource:
        """
        :return: The ontology a term comes from.
        """
        namespace = self.namespace_of(iri)
        if namespace is None:
            return TermSource.UNKNOWN
        return namespace.source

    def node_label(self, node: HypothesisGraphNode) -> str:
        """
        :return: The node's label with the ontology class of its fragment beneath it.
        """
        if not node.ontology_class:
            return node.label
        return f"{node.label}\n{self.prefixed(node.ontology_class)}"

    def relation_label(self, relation: HypothesisGraphRelation) -> str:
        """
        :return: The link's field with the ontology property stating it beneath it, or
            a note that the viewer drew the link.
        """
        if not relation.property_iri:
            return f"{relation.predicate}\n{DRAWN_LINK_NOTE}"
        return f"{relation.predicate}\n{self.prefixed(relation.property_iri)}"

    def backed_value(self, value: Any) -> BackedValue:
        """
        :return: A value a fragment holds, with the ontology it comes from: a fragment
            by its class, a concept or an enumerated individual by its own term.
        """
        if isinstance(value, DescriptionEntity):
            iri = type(value).term.iri
            return BackedValue(
                text=f"{value.label or type(value).__name__} "
                f"[{self.prefixed(iri)}]",
                source=self.source_of(iri),
            )
        if isinstance(value, ConceptReference):
            return BackedValue(
                text=self.prefixed(value.iri), source=self.source_of(value.iri)
            )
        if isinstance(value, StrEnum):
            return BackedValue(
                text=self.prefixed(value.value), source=self.source_of(value.value)
            )
        return BackedValue(text=repr(value), source=TermSource.LITERAL)

    def node_lines(self, node: HypothesisGraphNode) -> list[str]:
        """
        :return: Detail lines naming the node's ontology class, the linked terms it
            specialises, its evidence status and claims, and every value its fragment
            states with the property stating it and the value's ontology.
        """
        fragment = node.source
        if fragment is None:
            return [f"kind: {node.kind.value}", f"no ontology class {DRAWN_LINK_NOTE}"]
        term = type(fragment).term
        lines = [
            f"class: {self.prefixed(term.iri)} ({self.source_of(term.iri).value})",
            f"iri: {term.iri}",
            *(
                f"specialises: {self.prefixed(linked)} "
                f"({self.source_of(linked).value})"
                for linked in term.linked_terms
            ),
        ]
        if term.evidence_status:
            lines.append(f"evidence: {self.prefixed(term.evidence_status)}")
        lines.extend(f"claim: {self.prefixed(claim)}" for claim in term.claims)
        lines.extend(self.value_lines(fragment))
        return lines

    def value_lines(self, fragment: DescriptionEntity) -> list[str]:
        """
        :return: One line per value the fragment states, leaving out properties the
            ontology infers from sub-properties.
        """
        lines = []
        for declared in fields(fragment):
            property_term = PropertyTerm.of_field(type(fragment), declared.name)
            if property_term is None or is_inferred(type(fragment), declared.name):
                continue
            held: Any = attrgetter(declared.name)(fragment)
            for value in held if isinstance(held, list) else [held]:
                if value is None:
                    continue
                backed = self.backed_value(value)
                lines.append(
                    f"{declared.name} · {self.prefixed(property_term.iri)}: "
                    f"{backed.text} ({backed.source.value})"
                )
        return lines
