"""
A hypothesis in memory: its fragments as instances of the generated vocabulary, linked
into one structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any

from llm_reasoner.knowledge.ontologies.terms import DescriptionEntity


@dataclass(frozen=True)
class GivenFragments:
    """
    The fragments of stored hypotheses a new hypothesis builds on: it refers to them by
    their names and does not state them again.
    """

    fragments: list[DescriptionEntity] = field(default_factory=list)
    """
    Every given fragment.
    """

    names: dict[int, str] = field(default_factory=dict)
    """
    The name each given fragment is referred to by, keyed by the fragment's identity.
    """

    iris: dict[int, str] = field(default_factory=dict)
    """
    The IRI each given fragment is stored under, keyed by the fragment's identity.
    """

    def named(self) -> dict[str, DescriptionEntity]:
        """
        :return: Every given fragment, by the name it is referred to by.
        """
        return {self.names[id(fragment)]: fragment for fragment in self.fragments}


@dataclass
class DecodedHypothesis:
    """
    The fragments of one answer, linked into instances of the vocabulary.
    """

    root: DescriptionEntity
    """
    The fragment of the type being generated.
    """

    fragments: list[DescriptionEntity]
    """
    Every fragment, in the order the answer lists them.
    """

    identifiers: dict[int, str] = field(default_factory=dict)
    """
    The answer's identifier of each fragment, keyed by the fragment's identity.
    """

    given: GivenFragments = field(default_factory=GivenFragments)
    """
    The stored fragments the hypothesis builds on, which are not among its fragments.
    """

    def identifier_of(self, fragment: Any) -> str:
        """
        :return: The answer's identifier of a fragment, the name of a given fragment, or
            an empty string for a value the answer did not list.
        """
        return self.identifiers.get(id(fragment)) or self.given.names.get(
            id(fragment), ""
        )
