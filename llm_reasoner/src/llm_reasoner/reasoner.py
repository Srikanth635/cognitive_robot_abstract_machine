"""
The system assembled in one place: the resolver, the world it reads, and the knowledge
component it consults about every action it resolves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any

from llm_reasoner.clients.base import LanguageModelClient
from llm_reasoner.knowledge.access import KnowledgeService
from llm_reasoner.resolver.backend import LanguageModelBackend
from llm_reasoner.resolver.knowledge import KnowledgeRequester
from llm_reasoner.world.access import WorldAccess
from llm_reasoner.world.view import WorldView


@dataclass
class Reasoner:
    """
    The resolver with the world it reads and the knowledge component it consults.
    """

    client: LanguageModelClient
    """
    The language model the resolver asks, which must be able to call tools.
    """

    knowledge: KnowledgeService
    """
    Serves the hypothesis about each action resolved, generated once per situation.
    """

    world: WorldAccess = field(default_factory=WorldView)
    """
    The world the resolver reads, by default krrood's symbol graph, which holds no
    knowledge: that lives in the knowledge component's own graph.
    """

    def backend(self, **settings: Any) -> LanguageModelBackend:
        """
        :param settings: Further settings of the backend, such as its context provider.
        :return: A backend resolving queries with the world tools, consulting the
            knowledge component about the queried action before asking the model.
        """
        return LanguageModelBackend.with_world_tools(
            client=self.client,
            world=self.world,
            knowledge=KnowledgeRequester(service=self.knowledge),
            **settings,
        )
