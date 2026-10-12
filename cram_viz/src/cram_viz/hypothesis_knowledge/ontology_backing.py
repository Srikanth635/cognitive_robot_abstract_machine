"""
Naming ontology terms for display: the prefix each known namespace is written with, and
the ontology each term comes from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from llm_reasoner.knowledge.access import (
    CONCEPT_PREFIXES,
    CRAM_AICOR_NAMESPACE,
    DUL_NAMESPACE,
    HYPOTHESIS_CONCEPT_NAMESPACE,
    PREDICTIVE_MANIPULATION_NAMESPACE_ROOT,
    PredictiveManipulationOntologyFile,
)
from typing_extensions import Optional

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
