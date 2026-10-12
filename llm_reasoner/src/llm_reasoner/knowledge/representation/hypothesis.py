"""
A hypothesis in memory: its fragments as instances of the generated vocabulary, linked
into one structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any

from llm_reasoner.knowledge.ontologies.terms import DescriptionEntity


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

    def identifier_of(self, fragment: Any) -> str:
        """
        :return: The answer's identifier of a fragment, or an empty string for a value
            the answer did not list.
        """
        return self.identifiers.get(id(fragment), "")
