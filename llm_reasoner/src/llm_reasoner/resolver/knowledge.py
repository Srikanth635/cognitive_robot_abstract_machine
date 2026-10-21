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
from typing_extensions import Any, Callable, Mapping, Optional

from llm_reasoner.clients.tools import Tool
from llm_reasoner.knowledge.access import (
    PERSPECTIVE_ROOTS,
    ActionPerspectiveFile,
    DescribedAction,
    DescribedActions,
    DescriptionEntity,
    KnowledgeAccess,
    KnowledgeToolName,
    SituationFeatures,
    StoredRoot,
)
from llm_reasoner.resolver.grounds import GroundKey, GroundSource
from llm_reasoner.resolver.prompt_format import PromptRule, RuleName
from llm_reasoner.resolver.prompts import RequestAnswerKey
from llm_reasoner.world.access import identifier_of

KNOWLEDGE_RULE = PromptRule(
    name=RuleName.KNOWLEDGE,
    text=f"the knowledge tools ({KnowledgeToolName.KNOWLEDGE_OVERVIEW}, "
    f"{KnowledgeToolName.FIND_FRAGMENTS}, {KnowledgeToolName.DESCRIBE_FRAGMENT}) hold "
    "stereotypical knowledge of the requested action in its situation: its phases, "
    f"their subgoals and conditions, and its roles. Read "
    f"{KnowledgeToolName.KNOWLEDGE_OVERVIEW} first, follow what bears on the open "
    "slots, ground your choices on it, and list the fragments you relied on in "
    f'"{RequestAnswerKey.GROUNDS}".',
)
"""
The rule stating how the knowledge about the requested action is read and used, when
the model is offered the knowledge tools.
"""

FRAME_READING_ADVICE = (
    f"{KnowledgeToolName.DESCRIBE_FRAGMENT} on a slot lists the fragments linking to it. "
    "An open slot's \"frame\" names the frame's slots or choices it fills: describe "
    "those first, check what is expected of them against the world, and ground your "
    "choice on them."
)
"""
How the model is told to use the frame once it has read what the frame and the
perspectives on it hold.
"""


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

    perspective_record_ids: list[str] = field(default_factory=list)
    """
    The identifiers of the stored hypotheses about the action core's frame of the
    action and about each perspective on it consulted, the core first; none when no
    perspective is consulted.
    """

    @property
    def record_ids(self) -> list[str]:
        """
        Every stored hypothesis the consultation obtained, the phase structure first.
        """
        return [
            record_id
            for record_id in (self.record_id, *self.perspective_record_ids)
            if record_id is not None
        ]


@dataclass(frozen=True)
class KnowledgeGrounds(GroundSource):
    """
    The fragments of a consulted hypothesis, named as the knowledge tools name them.
    """

    service: KnowledgeAccess
    """
    The knowledge component holding the hypothesis.
    """

    record_ids: tuple[str, ...]
    """
    The record identifiers of the hypotheses read together.
    """

    @property
    def key(self) -> GroundKey:
        return GroundKey.KNOWLEDGE

    @property
    def name_form(self) -> str:
        return "a fragment name, as the knowledge tools give it"

    def identify(self, name: str) -> Optional[str]:
        return next(
            (
                cited
                for cited in (
                    self.service.citation(record_id, name)
                    for record_id in self.record_ids
                )
                if cited is not None
            ),
            None,
        )


@dataclass
class KnowledgeRequester:
    """
    Asks the knowledge component for the hypothesis describing the action a query asks
    for, in the situation the action is performed in.
    """

    service: KnowledgeAccess
    """
    The knowledge component: serves a stored hypothesis for a request, or generates and
    stores it, and offers the tools reading it.
    """

    described_actions: DescribedActions = field(default_factory=DescribedActions)
    """
    Which knowledge describes which action designator.
    """

    situation_features: SituationFeatures = field(default_factory=SituationFeatures)
    """
    Reads the situation features a knowledge structure accepts from the ontology.
    """

    perspectives: tuple[ActionPerspectiveFile, ...] = ()
    """
    The perspectives on the action core consulted beside the phase structure, with the
    core they describe; none by default, which consults neither.
    """

    consults_phase_structure: bool = True
    """
    Whether the phase structure describing the action is consulted, with the situation
    features it is served for; when not, no situation is chosen and only the core and
    its perspectives are.
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
        if not self.consults_phase_structure:
            situation = (
                self.situation_of(
                    described.frame, described.frame_query, situation_backend
                )
                if self.perspectives
                else {}
            )
            return KnowledgeConsultation(
                designator=designator,
                situation=situation,
                perspective_record_ids=self.perspective_records_for(
                    described, designator, situation
                ),
            )
        situation = self.situation_of(
            described.structure, described.query, situation_backend
        )
        return KnowledgeConsultation(
            designator=designator,
            situation=situation,
            record_id=self.service.serve(
                described.query(situation), instruction=designator
            ),
            perspective_record_ids=self.perspective_records_for(
                described, designator, situation
            ),
        )

    def perspective_records_for(
        self,
        described: DescribedAction,
        designator: str,
        situation: Mapping[str, StrEnum],
    ) -> list[str]:
        """
        :param situation: The situation features chosen for the action, by field name.
        :return: The records of the action core's frame of the action in the situation,
            as far as the frame accepts its features, and of each perspective consulted
            on that frame, served and generated first when none is stored; none when no
            perspective is consulted or the core has no frame of the action.
        """
        if not self.perspectives or described.frame is None:
            return []
        accepted = {
            feature.field_name
            for feature in self.situation_features.of(described.frame)
        }
        core_record = self.service.serve(
            described.frame_query(
                {name: value for name, value in situation.items() if name in accepted}
            ),
            instruction=designator,
        )
        return [
            core_record,
            *(
                self.service.serve(
                    PERSPECTIVE_ROOTS[perspective].query(
                        StoredRoot(record_id=core_record)
                    ),
                    instruction=designator,
                )
                for perspective in self.perspectives
            ),
        ]

    def tools_for(self, consultation: KnowledgeConsultation) -> list[Tool]:
        """
        :return: The knowledge tools reading the hypothesis a consultation obtained;
            none when no knowledge describes its designator.
        """
        if not consultation.record_ids:
            return []
        return self.service.tools(*consultation.record_ids)

    def grounds_in(self, consultation: KnowledgeConsultation) -> list[GroundSource]:
        """
        :return: The fragments of the hypothesis a consultation obtained, as what an
            answer may rest on; none when no knowledge describes its designator.
        """
        if not consultation.record_ids:
            return []
        return [
            KnowledgeGrounds(
                service=self.service, record_ids=tuple(consultation.record_ids)
            )
        ]

    def frame_parts_for(
        self, consultation: KnowledgeConsultation
    ) -> dict[str, list[str]]:
        """
        :return: The names of the parts of the consulted core's frame each field of the
            action fills, by field name; none when no core was consulted.
        """
        described = self.described_actions.describing(consultation.designator)
        if not consultation.perspective_record_ids or described is None:
            return {}
        return self.service.frame_parts_filled(
            consultation.perspective_record_ids[0], described.frame_fields
        )

    def rules_for(self, consultation: KnowledgeConsultation) -> list[PromptRule]:
        """
        :return: The rules of reading the knowledge a consultation obtained: how to read
            it, and how to read the action's frame when perspectives were consulted.
        """
        if not consultation.perspective_record_ids:
            return [KNOWLEDGE_RULE]
        return [KNOWLEDGE_RULE, self.frame_rule()]

    def frame_rule(self) -> PromptRule:
        """
        :return: The rule saying what the action's frame and each perspective offered on it
            hold, in their own words, and how to use them.
        """
        modules = (ActionPerspectiveFile.ACTION_CORE, *self.perspectives)
        held = " ".join(module.summary for module in modules)
        return PromptRule(
            name=RuleName.FRAME_KNOWLEDGE,
            text=f'the knowledge also holds the action\'s frame ({KnowledgeToolName.KNOWLEDGE_OVERVIEW} "frame") '
            f"and perspectives on it. {held} {FRAME_READING_ADVICE}",
        )

    def situation_of(
        self,
        root_type: Optional[type[DescriptionEntity]],
        query_for: Callable[[Mapping[str, Any]], Match],
        situation_backend: GenerativeBackend,
    ) -> dict[str, StrEnum]:
        """
        :param root_type: The knowledge the situation is chosen for: the phase structure
            or the action core's frame describing the action.
        :param query_for: Makes the query for that knowledge with the values given.
        :param situation_backend: Fills the situation features.
        :return: The situation features the backend chose, by field name, the ones it
            left unknown omitted; none when the knowledge accepts no features.
        """
        features = [] if root_type is None else self.situation_features.of(root_type)
        if not features:
            return {}
        query = query_for({feature.field_name: ... for feature in features})
        (structure,) = query.evaluate(backend=situation_backend)
        chosen = {
            feature.field_name: feature.value_in(structure) for feature in features
        }
        return {name: value for name, value in chosen.items() if value is not None}
