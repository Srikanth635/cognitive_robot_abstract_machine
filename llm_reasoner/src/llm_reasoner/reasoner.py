"""
The system assembled in one place: the resolver, the world it reads, and the knowledge
component it consults about every action it resolves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, Optional

from llm_reasoner.clients.base import LanguageModelClient
from llm_reasoner.clients.usage import TokenPrices
from llm_reasoner.knowledge.access import ActionPerspectiveFile, KnowledgeAccess
from llm_reasoner.knowledge.hypotheses.tools import KnowledgeTool
from llm_reasoner.observation.components import ObservedKnowledge, ObservedWorld
from llm_reasoner.observation.events import Component
from llm_reasoner.observation.observed import ObservedClient, ToolOwner
from llm_reasoner.observation.sinks import EventSink
from llm_reasoner.resolver.backend import LanguageModelBackend
from llm_reasoner.resolver.knowledge import KnowledgeRequester
from llm_reasoner.world.access import WorldAccess
from llm_reasoner.world.tools import WorldTool
from llm_reasoner.world.view import WorldView

TOOL_OWNERS = (
    ToolOwner(tool_type=WorldTool, component=Component.WORLD_INJECTOR),
    ToolOwner(tool_type=KnowledgeTool, component=Component.KNOWLEDGE),
)
"""
The component each kind of tool the resolver's model is offered belongs to.
"""


@dataclass
class Reasoner:
    """
    The resolver with the world it reads and the knowledge component it consults.
    """

    client: LanguageModelClient
    """
    The language model the resolver asks, which must be able to call tools.
    """

    knowledge: KnowledgeAccess
    """
    Serves the hypothesis about each action resolved, generated once per situation.
    """

    world: WorldAccess = field(default_factory=WorldView)
    """
    The world the resolver reads, by default krrood's symbol graph, which holds no
    knowledge: that lives in the knowledge component's own graph.
    """

    sink: Optional[EventSink] = None
    """
    Where the calls between the components and to the model are reported, and the
    resolver's own steps; nothing is reported, and nothing wrapped, when absent.
    """

    prices: Optional[TokenPrices] = None
    """
    What the model charges, to report what each completion cost.
    """

    perspectives: tuple[ActionPerspectiveFile, ...] = ()
    """
    The perspectives on the action core consulted with it beside the phase structure;
    none by default.
    """

    def backend(self, **settings: Any) -> LanguageModelBackend:
        """
        :param settings: Further settings of the backend, such as its context provider.
        :return: A backend resolving queries with the world tools, consulting the
            knowledge component about the queried action before asking the model; when
            there is a sink, one whose model, world and knowledge component report to
            it as the resolver's.
        """
        if self.sink is None:
            return LanguageModelBackend.with_world_tools(
                client=self.client,
                world=self.world,
                knowledge=KnowledgeRequester(
                    service=self.knowledge, perspectives=self.perspectives
                ),
                **settings,
            )
        return LanguageModelBackend.with_world_tools(
            client=ObservedClient.of(
                self.client,
                self.sink,
                Component.RESOLVER,
                prices=self.prices,
                tool_owners=TOOL_OWNERS,
            ),
            world=ObservedWorld(world=self.world, sink=self.sink),
            knowledge=KnowledgeRequester(
                service=ObservedKnowledge(knowledge=self.knowledge, sink=self.sink),
                perspectives=self.perspectives,
            ),
            **{"observer": self.sink, **settings},
        )
