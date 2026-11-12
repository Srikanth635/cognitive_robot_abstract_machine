"""
One stored hypothesis as instances of the generated vocabulary, and the fragments its
citations name.

Questions a single fragment answers through its own fields are left to the fragment.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from typing_extensions import Optional, TypeVar

from krrood.utils import resolve_class_from_full_name

from llm_reasoner.knowledge.storage.citation import Citation
from llm_reasoner.knowledge.representation.rdf import GraphReader, ReadHypothesis
from llm_reasoner.knowledge.storage.record import HypothesisRecord
from llm_reasoner.knowledge.storage.store import StoredHypothesis
from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge.vocabulary.terms import DescriptionEntity

Fragment = TypeVar("Fragment", bound=DescriptionEntity)
"""
The vocabulary class of the fragments asked for.
"""


@dataclass
class UnknownCitation(LLMReasonerError):
    """
    A citation names a fragment its stored hypothesis does not have.
    """

    citation: Citation
    """
    The citation.
    """

    def __post_init__(self):
        super().__init__(f"the stored hypothesis has no fragment {self.citation.iri}")


@dataclass
class HypothesisView:
    """
    One stored hypothesis, as instances of the generated vocabulary.
    """

    record: HypothesisRecord
    """
    The provenance of the hypothesis.
    """

    fragments: list[DescriptionEntity]
    """
    Every fragment of the hypothesis.
    """

    identifiers: dict[int, str]
    """
    The IRI each fragment is stored under, keyed by the fragment's identity.
    """

    @classmethod
    def of(
        cls, stored: StoredHypothesis, read: Optional[ReadHypothesis] = None
    ) -> HypothesisView:
        """
        :param stored: The stored hypothesis.
        :param read: Its fragments, read with the fragments it builds on; read from its
            graph alone when not given.
        :return: The view of a stored hypothesis.
        """
        read = read or GraphReader().read(stored.graph)
        return cls(
            record=stored.record,
            fragments=read.fragments,
            identifiers=read.identifiers,
        )

    @cached_property
    def root(self) -> DescriptionEntity:
        """
        The fragment the hypothesis was generated for.
        """
        root_type = resolve_class_from_full_name(self.record.root_type)
        (root,) = [
            fragment for fragment in self.fragments if isinstance(fragment, root_type)
        ]
        return root

    def by_iri(self) -> dict[str, DescriptionEntity]:
        """
        :return: Every fragment, by the IRI it is stored under.
        """
        return {self.identifiers[id(fragment)]: fragment for fragment in self.fragments}

    def instances_of(self, kind: type[Fragment]) -> list[Fragment]:
        """
        :param kind: A vocabulary class.
        :return: The hypothesis's fragments that are instances of the class, in order.
        """
        return [fragment for fragment in self.fragments if isinstance(fragment, kind)]

    # %% citations

    def citation_of(self, fragment: DescriptionEntity) -> Citation:
        """
        :return: The citation of a fragment of this hypothesis.
        """
        return Citation.from_iri(self.identifiers[id(fragment)])

    def cites(self, citation: Citation) -> bool:
        """
        :return: Whether a citation names a fragment of this hypothesis.
        """
        return any(
            self.identifiers[id(fragment)] == citation.iri
            for fragment in self.fragments
        )

    def fragment_cited(self, citation: Citation) -> DescriptionEntity:
        """
        :return: The fragment a citation names.
        :raises UnknownCitation: If this hypothesis has no such fragment.
        """
        cited = [
            fragment
            for fragment in self.fragments
            if self.identifiers[id(fragment)] == citation.iri
        ]
        if not cited:
            raise UnknownCitation(citation=citation)
        return cited[0]
