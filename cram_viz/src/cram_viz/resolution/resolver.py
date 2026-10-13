"""
Resolving a predefined designator in a scene with llm_reasoner, each resolution a run of
its own from the same initial environment: the scene's world built anew, read through
the world injector and its geometry stated as a fact of the situation, the knowledge
component consulted about the action over a copy of the viewer's store, and what the
resolution chose, why, and what it rests on, written for display and kept with the run.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from operator import attrgetter
from pathlib import Path

from coraplex.robot_plans.actions.composite.tool_based import PouringAction
from krrood.entity_query_language.query.match import Match
from llm_reasoner.clients.base import LanguageModelClient
from llm_reasoner.clients.usage import SONNET_5_5_PRICES, TokenPrices
from llm_reasoner.clients.anthropic_client import AnthropicClient
from llm_reasoner.knowledge.access import Citation, KnowledgeService
from llm_reasoner.reasoner import Reasoner
from llm_reasoner.resolver.backend import LanguageModelBackend
from llm_reasoner.resolver.grounds import Ground, GroundKey, Justification
from llm_reasoner.resolver.prompts import SourceDetail
from llm_reasoner.world.view import WorldView
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Any, Callable, Dict, List, Optional

from cram_viz import paths
from cram_viz.hypothesis_knowledge.generation import ModelHypothesisSource
from cram_viz.query_source import QuerySource
from cram_viz.resolution.designators import (
    DesignatorQuery,
    DesignatorUnavailable,
    QuerySlots,
    SceneNamespace,
)
from cram_viz.resolution.runs import ResolutionRuns, RunDirectory, RunEnvironment
from cram_viz.resolution.scene_geometry import SceneGeometry
from cram_viz.resolution.slot_data import SlotData
from cram_viz.resolution.trial import SimulatedPouring, TriedPour
from cram_viz.resolution.scene_world import SceneWorlds

RESOLVER_MAX_TOKENS = 2048
"""
The most tokens the resolver's model may answer one turn with.
"""

POUR_TRIALS = 3
"""
How many pours a resolution tries in the simulation; the last is accepted whatever it
brings about.
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

    serialized: Any
    """
    The slot's value as data the run's execution reads back: a world object by its
    body's name, an enumeration member by its name, any other value as it is.
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

    run: str
    """
    The name of the run it was made in, whose directory holds the knowledge it consulted
    and generated.
    """

    trials: List[TriedPour] = field(default_factory=list)
    """
    The pours tried in the simulation before the resolution was accepted, in order.
    """

    @property
    def query_on_one_line(self) -> str:
        """
        :return: The match query with its line breaks and indentation folded into
            single spaces.
        """
        return QuerySource(text=self.query).on_one_line

    def payload(self) -> Dict[str, Any]:
        """
        :return: The resolution as JSON data.
        """
        return asdict(self)


@dataclass(frozen=True)
class ResolutionRun:
    """
    One resolution run: what it resolved and the knowledge it consulted, which no other
    run reads.
    """

    resolution: ResolvedDesignator
    """
    What the run resolved.
    """

    knowledge: KnowledgeService
    """
    The knowledge component over the run's own store.
    """


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

    baseline: KnowledgeService
    """
    The knowledge every run starts from: each run consults it over a copy of its store,
    never the store itself.
    """

    runs: ResolutionRuns
    """
    Where the runs are kept.
    """

    worlds: SceneWorlds = field(default_factory=SceneWorlds)
    """
    The world of the scene last resolved in.
    """

    prices: TokenPrices = SONNET_5_5_PRICES
    """
    What the model charges, to report a resolution's cost.
    """

    simulate: Callable[..., Any] = subprocess.run
    """
    Runs the process trying a pour in the simulation.
    """

    @classmethod
    def for_viewer(
        cls, source: ModelHypothesisSource, worlds: SceneWorlds
    ) -> DesignatorResolver:
        """
        :param source: The viewer's knowledge source, with its credentials, model and
            store.
        :param worlds: The scene worlds the viewer shares with its designator menu.
        :return: A resolver asking the source's model, every run starting from the
            source's knowledge, the runs kept in the viewer's runs directory.
        """
        source.load_credentials()
        return cls(
            client=AnthropicClient(model=source.model, max_tokens=RESOLVER_MAX_TOKENS),
            baseline=source.knowledge_service(),
            runs=ResolutionRuns(directory=paths.resolution_runs_dir()),
            worlds=worlds,
        )

    def resolve(
        self, designator: DesignatorQuery, scene: str, bundle: Path
    ) -> ResolutionRun:
        """
        Resolve a designator in a run of its own: the scene's world built anew, the
        knowledge consulted over a copy of the baseline's store.

        :param designator: The designator to resolve.
        :param scene: The name of the scene to resolve it in.
        :param bundle: The scene's bundle directory.
        :return: The run: what the resolution chose, why, and what it rests on, and the
            knowledge it consulted.
        :raises DesignatorUnavailable: When the scene cannot resolve the designator.
        """
        world = self.worlds.initial_world_for(scene, bundle)
        names = SceneNamespace(world=world).names()
        shortfall = designator.shortfall_in(names)
        if not shortfall.is_empty:
            raise DesignatorUnavailable(shortfall=shortfall)
        run = self.runs.start(
            self.baseline.backend.store, RunEnvironment(scene=scene, bundle=bundle)
        )
        knowledge = self.knowledge_for(run)
        resolution = self.resolution_in(
            run,
            designator,
            designator.query_in(names),
            knowledge,
            SceneGeometry(world=world),
            bundle,
        )
        run.keep(resolution)
        return ResolutionRun(resolution=resolution, knowledge=knowledge)

    def knowledge_for(self, run: RunDirectory) -> KnowledgeService:
        """
        :return: The baseline's knowledge component over the run's own store.
        """
        return KnowledgeService(
            backend=replace(self.baseline.backend, store=run.knowledge_store)
        )

    def resolution_in(
        self,
        run: RunDirectory,
        designator: DesignatorQuery,
        query: Match,
        knowledge: KnowledgeService,
        geometry: SceneGeometry,
        bundle: Path,
    ) -> ResolvedDesignator:
        """
        :param run: The run resolving.
        :param designator: The designator resolved.
        :param query: The designator's query on the scene.
        :param knowledge: The knowledge component the run consults.
        :param geometry: The scene's geometry, stated as a fact of the situation.
        :param bundle: The scene's bundle directory, from which a pour is tried.
        :return: What resolving the query chose, why, what it rests on, and the pours it
            tried in the simulation.
        """
        slots = QuerySlots.of(query)
        trials = self.trials_for(query, slots, bundle)
        view = WorldView()
        backend = Reasoner(client=self.client, knowledge=knowledge, world=view).backend(
            context_provider=geometry,
            number_of_candidates=1,
            source_detail=SourceDetail.FULL,
            outcome_check=trials,
            max_repairs=POUR_TRIALS,
        )
        instances = list(query.evaluate(backend=backend))
        justifications = {
            id(justification.instance): justification
            for justification in backend.justifications
        }
        consultation = backend.consultation
        return ResolvedDesignator(
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
            run=run.name,
            trials=[] if trials is None else trials.trials,
        )

    def trials_for(
        self, query: Match, slots: QuerySlots, bundle: Path
    ) -> Optional[SimulatedPouring]:
        """
        :return: The check trying a pouring in the simulation before it is accepted, or
            none for any other action.
        """
        if query.type is not PouringAction:
            return None
        return SimulatedPouring(
            bundle=bundle,
            slots=slots.names,
            allowed_trials=POUR_TRIALS,
            run=self.simulate,
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
                    serialized=SlotData.from_value(attrgetter(name)(instance)),
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
