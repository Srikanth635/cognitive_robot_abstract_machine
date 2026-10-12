"""
Resolving a predefined designator in a scene with llm_reasoner: the scene's world read
through the world injector, the knowledge component consulted about the action, and what
the resolution chose, why, and what it rests on, written for display.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from operator import attrgetter
from pathlib import Path

from llm_reasoner.clients.base import LanguageModelClient
from llm_reasoner.clients.usage import SONNET_5_5_PRICES, TokenPrices
from llm_reasoner.clients.anthropic_client import AnthropicClient
from llm_reasoner.knowledge.access import Citation, KnowledgeService
from llm_reasoner.reasoner import Reasoner
from llm_reasoner.resolver.backend import LanguageModelBackend
from llm_reasoner.resolver.context import SchemaOnlyContextProvider
from llm_reasoner.resolver.grounds import Ground, GroundKey, Justification
from llm_reasoner.resolver.prompts import SourceDetail
from llm_reasoner.world.view import WorldView
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Any, Dict, List, Optional

from cram_viz.hypothesis_knowledge.generation import ModelHypothesisSource
from cram_viz.resolution.designators import (
    DesignatorQuery,
    DesignatorUnavailable,
    PouringDesignator,
    QuerySlots,
    SceneNamespace,
)
from cram_viz.resolution.scene_world import SceneWorlds

RESOLVER_MAX_TOKENS = 2048
"""
The most tokens the resolver's model may answer one turn with.
"""

# %% what a resolution shows


@dataclass(frozen=True)
class ResolvedSlot:
    """
    One slot of a resolved action.
    """

    name: str
    """
    The slot's field name.
    """

    value: str
    """
    The slot's value, written for reading.
    """

    given: bool
    """
    Whether the designator's query stated the value rather than leaving it to the
    resolver.
    """


@dataclass(frozen=True)
class ShownGround:
    """
    One thing a resolved action rests on, named for reading.
    """

    source: GroundKey
    """
    The source it belongs to.
    """

    name: str
    """
    Its name: a knowledge fragment's, or a world individual's handle with what it is.
    """


@dataclass(frozen=True)
class ResolvedAction:
    """
    One action a designator resolved to, with why it was chosen.
    """

    slots: List[ResolvedSlot]
    """
    Every slot of the designator, in the order the query states them.
    """

    why: str
    """
    The model's explanation.
    """

    grounds: List[ShownGround]
    """
    What the choices rest on.
    """


@dataclass(frozen=True)
class ResolvedDesignator:
    """
    What resolving a predefined designator produced.
    """

    identifier: PouringDesignator
    """
    The designator resolved.
    """

    query: str
    """
    The designator's match query, as written.
    """

    designator: str
    """
    The name of the action's class.
    """

    actions: List[ResolvedAction]
    """
    The actions it resolved to.
    """

    situation: Dict[str, str]
    """
    The situation features the knowledge was consulted for, by field name, each by the
    name of its value.
    """

    record_id: Optional[str]
    """
    The identifier of the hypothesis consulted, ``None`` when no knowledge describes the
    action.
    """

    cost: float
    """
    What the resolver's conversations cost, in dollars; a hypothesis generated for it is
    not included.
    """

    def payload(self) -> Dict[str, Any]:
        """
        :return: The resolution as JSON data.
        """
        return asdict(self)


# %% resolving


@dataclass
class DesignatorResolver:
    """
    Resolves predefined designators in the scene the viewer shows.
    """

    client: LanguageModelClient
    """
    The language model the resolver asks, which must be able to call tools.
    """

    knowledge: KnowledgeService
    """
    The knowledge component consulted about the action, whose hypotheses the viewer
    shows.
    """

    worlds: SceneWorlds = field(default_factory=SceneWorlds)
    """
    The world of the scene last resolved in.
    """

    prices: TokenPrices = SONNET_5_5_PRICES
    """
    What the model charges, to report a resolution's cost.
    """

    @classmethod
    def for_viewer(
        cls, source: ModelHypothesisSource, worlds: SceneWorlds
    ) -> DesignatorResolver:
        """
        :param source: The viewer's knowledge source, with its credentials, model and
            store.
        :param worlds: The scene worlds the viewer shares with its designator menu.
        :return: A resolver asking the source's model, consulting the source's
            knowledge.
        """
        source.load_credentials()
        return cls(
            client=AnthropicClient(model=source.model, max_tokens=RESOLVER_MAX_TOKENS),
            knowledge=source.knowledge_service(),
            worlds=worlds,
        )

    def resolve(
        self, designator: DesignatorQuery, scene: str, bundle: Path
    ) -> ResolvedDesignator:
        """
        :param designator: The designator to resolve.
        :param scene: The name of the scene to resolve it in.
        :param bundle: The scene's bundle directory.
        :return: What the resolution chose, why, and what it rests on.
        :raises DesignatorUnavailable: When the scene cannot resolve the designator.
        """
        names = SceneNamespace(world=self.worlds.world_for(scene, bundle)).names()
        offered = designator.offered_in(names)
        if not offered.available:
            raise DesignatorUnavailable(offered=offered)
        query = designator.query_in(names)
        slots = QuerySlots.of(query)
        view = WorldView()
        backend = Reasoner(
            client=self.client, knowledge=self.knowledge, world=view
        ).backend(
            context_provider=SchemaOnlyContextProvider(),
            number_of_candidates=1,
            source_detail=SourceDetail.FULL,
        )
        instances = list(query.evaluate(backend=backend))
        justifications = {
            id(justification.instance): justification
            for justification in backend.justifications
        }
        consultation = backend.consultation
        return ResolvedDesignator(
            identifier=designator.identifier,
            query=designator.source.text,
            designator=query.type.__name__,
            actions=[
                self.resolved_action(instance, slots, justifications, view)
                for instance in instances
            ],
            situation=(
                {}
                if consultation is None
                else {
                    name: value.name for name, value in consultation.situation.items()
                }
            ),
            record_id=None if consultation is None else consultation.record_id,
            cost=self.cost_of(backend),
        )

    def resolved_action(
        self,
        instance: Any,
        slots: QuerySlots,
        justifications: Dict[int, Justification],
        view: WorldView,
    ) -> ResolvedAction:
        """
        :return: One resolved action, written for display.
        """
        justification = justifications.get(id(instance))
        return ResolvedAction(
            slots=[
                ResolvedSlot(
                    name=name,
                    value=self.written(attrgetter(name)(instance)),
                    given=name in slots.given,
                )
                for name in slots.names
            ],
            why="" if justification is None else justification.why,
            grounds=(
                []
                if justification is None
                else [self.shown(ground, view) for ground in justification.grounds]
            ),
        )

    def shown(self, ground: Ground, view: WorldView) -> ShownGround:
        """
        :return: A ground named for reading: a knowledge fragment by its name, a world
            individual by its handle and what it is.
        """
        if ground.source is GroundKey.KNOWLEDGE:
            return ShownGround(
                source=ground.source,
                name=Citation.from_iri(ground.identifier).fragment_name,
            )
        individual = view.individual_of(ground.identifier)
        return ShownGround(
            source=ground.source,
            name=f"{ground.identifier} ({self.written(individual)})",
        )

    @staticmethod
    def written(value: Any) -> str:
        """
        :return: A slot's value written for reading: a body by its name, an annotated
            object by its kind and its body's name, an enumeration member by its name.
        """
        if isinstance(value, Body):
            return value.name.name
        if isinstance(value, HasRootBody):
            return f"{type(value).__name__} ({value.root.name.name})"
        if isinstance(value, Enum):
            return value.name
        return str(value)

    def cost_of(self, backend: LanguageModelBackend) -> float:
        """
        :return: What the backend's conversations, its situation's included, cost.
        """
        situation = (
            []
            if backend.situation_backend is None
            else backend.situation_backend.conversations
        )
        return sum(
            self.prices.cost_of(conversation.usage)
            for conversation in [*situation, *backend.conversations]
        )
