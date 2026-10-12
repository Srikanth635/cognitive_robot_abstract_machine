"""
Consulting the knowledge component about the action a query asks for: the knowledge
describing the action's designator, the situation the action is performed in, as the
situation features that knowledge accepts, and the hypothesis served for that situation.

The knowledge component is reached through its public module only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from krrood.entity_query_language.backends import GenerativeBackend
from krrood.entity_query_language.query.match import Match
from typing_extensions import Optional

from llm_reasoner.knowledge.access import (
    DescribedAction,
    DescribedActions,
    KnowledgeService,
    SituationFeatures,
)
from llm_reasoner.world.access import identifier_of


@dataclass
class KnowledgeConsultation:
    """
    What the knowledge component was asked about an action, and what it served.
    """

    designator: str
    """
    The module and class name of the action designator the query asks for.
    """

    situation: dict[str, StrEnum] = field(default_factory=dict)
    """
    The situation features chosen for the action, by field name; a feature left unknown
    is absent.
    """

    record_id: Optional[str] = None
    """
    The identifier of the stored hypothesis served for the situation, ``None`` when no
    knowledge describes the designator.
    """


@dataclass
class KnowledgeRequester:
    """
    Asks the knowledge component for the hypothesis describing the action a query asks
    for, in the situation the action is performed in.
    """

    service: KnowledgeService
    """
    Serves a stored hypothesis for a request, or generates and stores it.
    """

    described_actions: DescribedActions = field(default_factory=DescribedActions)
    """
    Which knowledge describes which action designator.
    """

    situation_features: SituationFeatures = field(default_factory=SituationFeatures)
    """
    Reads the situation features a knowledge structure accepts from the ontology.
    """

    def consult(
        self, action: Match, situation_backend: GenerativeBackend
    ) -> KnowledgeConsultation:
        """
        :param action: The query for the action.
        :param situation_backend: Fills the situation features of the knowledge
            describing the action, given a query on that knowledge's structure.
        :return: The consultation, with the hypothesis served for the action's
            situation when knowledge describes the action's designator.
        """
        designator = identifier_of(action.type)
        described = self.described_actions.describing(designator)
        if described is None:
            return KnowledgeConsultation(designator=designator)
        situation = self.situation_of(described, situation_backend)
        return KnowledgeConsultation(
            designator=designator,
            situation=situation,
            record_id=self.service.serve(
                described.query(situation), instruction=designator
            ),
        )

    def situation_of(
        self, described: DescribedAction, situation_backend: GenerativeBackend
    ) -> dict[str, StrEnum]:
        """
        :param described: The knowledge describing the action.
        :param situation_backend: Fills the situation features.
        :return: The situation features the backend chose, by field name, the ones it
            left unknown omitted; none when the knowledge accepts no features.
        """
        features = self.situation_features.of(described.structure)
        if not features:
            return {}
        query = described.query({feature.field_name: ... for feature in features})
        (structure,) = query.evaluate(backend=situation_backend)
        chosen = {
            feature.field_name: feature.value_in(structure) for feature in features
        }
        return {name: value for name, value in chosen.items() if value is not None}
