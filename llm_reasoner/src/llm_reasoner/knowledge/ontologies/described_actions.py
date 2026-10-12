"""
Which knowledge describes which action designator: for a designator class, the action
concept it performs and the knowledge structure describing that action.

The mapping is kept as data and names everything by identifier — the designator by its
module and class name, the concept and the structure by their IRIs — so that knowledge
is found for a designator without importing the package declaring it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from typing_extensions import Any, Mapping, Optional

from llm_reasoner.knowledge.ontologies.terms import ConceptReference, DescriptionEntity
from llm_reasoner.knowledge.ontologies.vocabulary import Vocabulary

DESCRIBED_ACTIONS_FILE = Path(__file__).parent / "described_actions.json"
"""
Where the mapping from action designators to the knowledge describing them is kept.
"""


class DescribedActionKey(StrEnum):
    """
    The keys of one entry of the mapping.
    """

    DESIGNATOR = "designator"
    """
    The module and class name of the action designator.
    """

    CONCEPT = "concept"
    """
    The IRI of the action concept the designator performs.
    """

    STRUCTURE = "structure"
    """
    The IRI of the ontology class of the knowledge structure describing the action.
    """


@dataclass(frozen=True)
class DescribedAction:
    """
    An action designator with the knowledge that describes it.
    """

    designator: str
    """
    The module and class name of the action designator.
    """

    concept: ConceptReference
    """
    The action concept the designator performs, such as CRAM-AICOR's pouring.
    """

    structure: type[DescriptionEntity]
    """
    The generated class of the knowledge structure describing the action.
    """

    @classmethod
    def from_json(
        cls, stored: Mapping[str, Any], vocabulary: Vocabulary
    ) -> DescribedAction:
        """
        :return: The entry stored as a JSON object, its structure resolved to its
            generated class.
        """
        return cls(
            designator=stored[DescribedActionKey.DESIGNATOR],
            concept=ConceptReference(iri=stored[DescribedActionKey.CONCEPT]),
            structure=vocabulary.definition_named(stored[DescribedActionKey.STRUCTURE]),
        )


@dataclass
class DescribedActions:
    """
    The action designators knowledge describes, read from the mapping kept as data.
    """

    vocabulary: Vocabulary = field(default_factory=Vocabulary)
    """
    Resolves the IRIs of knowledge structures to their generated classes.
    """

    entries: list[DescribedAction] = field(init=False)
    """
    Every described action designator, in the order the mapping lists them.
    """

    def __post_init__(self):
        self.entries = [
            DescribedAction.from_json(stored, self.vocabulary)
            for stored in json.loads(DESCRIBED_ACTIONS_FILE.read_text())
        ]

    def describing(self, designator: str) -> Optional[DescribedAction]:
        """
        :param designator: The module and class name of an action designator.
        :return: The knowledge describing it, or ``None`` when no knowledge does.
        """
        return next(
            (
                described
                for described in self.entries
                if described.designator == designator
            ),
            None,
        )
