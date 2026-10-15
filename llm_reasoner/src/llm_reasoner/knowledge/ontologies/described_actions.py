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

from krrood.entity_query_language.factories import the
from krrood.entity_query_language.query.match import Match
from typing_extensions import Any, Mapping, Optional

from llm_reasoner.knowledge.ontologies.action_perspectives import action_core
from llm_reasoner.knowledge.ontologies.predictive_manipulation import core
from llm_reasoner.knowledge.ontologies.terms import ConceptReference
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

    FRAME = "frame"
    """
    The IRI of the class of the action core's frame of the action, when the core has
    one.
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

    structure: type[core.ActionPhaseStructure]
    """
    The generated class of the phase structure describing the action.
    """

    frame: Optional[type[action_core.ActionFrame]] = None
    """
    The generated class of the action core's frame of the action, ``None`` when the core
    has none.
    """

    def query(self, values: Mapping[str, Any]) -> Match:
        """
        :param values: Values of fields of the structure, by field name; ``...`` leaves
            a field open for a backend to fill.
        :return: A query for the one structure decomposing this action's concept, with
            the values given.
        """
        return the(self.structure)(decomposes_task=self.concept, **values)

    def frame_query(self) -> Match:
        """
        :return: A query for the action core's one frame framing this action's concept.
        """
        return the(self.frame)(frames_action=self.concept)

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
            frame=(
                vocabulary.definition_named(stored[DescribedActionKey.FRAME])
                if DescribedActionKey.FRAME in stored
                else None
            ),
        )


@dataclass
class DescribedActions:
    """
    The action designators knowledge describes.
    """

    entries: list[DescribedAction] = field(
        default_factory=lambda: DescribedActions.read(DESCRIBED_ACTIONS_FILE)
    )
    """
    Every described action designator, by default as the mapping kept as data lists
    them.
    """

    @staticmethod
    def read(path: Path) -> list[DescribedAction]:
        """
        :param path: A JSON file listing described action designators.
        :return: Its entries, in order, their structures resolved to generated classes.
        """
        vocabulary = Vocabulary()
        return [
            DescribedAction.from_json(stored, vocabulary)
            for stored in json.loads(path.read_text())
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
