"""
References from a declaration to terms of an ontology that is not compiled here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge_updated.ontological.vocabulary import AlignmentRelation

# %% terms


@dataclass(frozen=True)
class Term:
    """
    A named ontology term, identified by its IRI.
    """

    iri: str
    """
    The term's identifier.
    """

    @property
    def local_name(self) -> str:
        """
        The end of the IRI, after its last ``#`` or, when it has none, its last ``/``.
        """
        separator = "#" if "#" in self.iri else "/"
        return self.iri.rsplit(separator, 1)[-1]


# %% concept references


@dataclass(frozen=True)
class ConceptReference(Term):
    """
    A term held as a value, with the terms that subsume it.

    Stands in for a class that is not compiled to Python, so a field can name it without
    holding a ``Type``. Carrying the ancestors makes subsumption a set-membership test
    that needs no reasoner.
    """

    label: Optional[str] = None
    """
    Human-readable name, or ``None`` to show the local name.
    """

    ancestors: frozenset[str] = frozenset()
    """
    IRIs of every term that subsumes this one, transitively and excluding itself.
    """

    def is_subsumed_by(self, other: ConceptReference | str) -> bool:
        """
        Whether this term is ``other`` or a specialisation of it.

        :param other: The subsuming term, or its IRI.
        """
        target = other.iri if isinstance(other, ConceptReference) else other
        return target == self.iri or target in self.ancestors

    def __str__(self) -> str:
        return self.label or self.local_name


# %% alignments


@dataclass(frozen=True)
class Alignment(Term):
    """
    A link from a concept to a term of another ontology.

    Recorded as an annotation: no axiom of the foreign term is imported or reasoned with.
    """

    relation: AlignmentRelation
    """
    How the concept relates to the foreign term.
    """


# %% namespaces


@dataclass
class InvalidNamespace(LLMReasonerError):
    """
    A namespace IRI does not end where a term's name can begin.
    """

    iri: str
    """
    The rejected namespace IRI.
    """

    def __str__(self) -> str:
        return f"namespace '{self.iri}' must end with '#' or '/'"


NAMESPACE_TERMINATORS = ("#", "/")
"""
The characters a namespace IRI may end with.
"""


@dataclass(frozen=True)
class Namespace:
    """
    The IRI prefix under which a module declares its terms.
    """

    iri: str
    """
    The prefix, ending with ``#`` or ``/``.
    """

    def __post_init__(self) -> None:
        """
        :raises InvalidNamespace: when the prefix does not end with ``#`` or ``/``.
        """
        if not self.iri.endswith(NAMESPACE_TERMINATORS):
            raise InvalidNamespace(self.iri)

    def term(self, local_name: str) -> str:
        """
        Return the IRI of a term with this name in the namespace.
        """
        return self.iri + local_name

    def property_term(self, field_name: str) -> str:
        """
        Return the IRI of the property held by a field, named after it in lower camel
        case.
        """
        first, *rest = field_name.split("_")
        return self.term(first + "".join(word.capitalize() for word in rest))
