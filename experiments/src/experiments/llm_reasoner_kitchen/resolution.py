"""
Resolving the cases with a reasoner as a running CRAM process would, and what a
resolution cost.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.entity_query_language.query.match import Match
from llm_reasoner.clients.tools import ToolConversation
from llm_reasoner.clients.usage import TokenPrices
from llm_reasoner.reasoner import Reasoner
from llm_reasoner.resolver.backend import LanguageModelBackend
from llm_reasoner.resolver.context import (
    PromptContextProvider,
    SchemaOnlyContextProvider,
    TaskContextProvider,
)
from llm_reasoner.resolver.prompts import SourceDetail
from typing_extensions import Any, List


@dataclass
class Resolution:
    """
    The actions a query resolved to, and the backend that resolved them.
    """

    actions: List[Any]
    """
    The resolved actions.
    """

    backend: LanguageModelBackend
    """
    The backend, with its conversations, its consultation of the knowledge component and
    the justifications of its answers.
    """

    @property
    def conversations(self) -> List[ToolConversation]:
        """
        Every conversation the resolution held with the model: choosing the situation,
        when the knowledge was consulted, and the request.
        """
        situation = (
            []
            if self.backend.situation_backend is None
            else self.backend.situation_backend.conversations
        )
        return [*situation, *self.backend.conversations]

    def cost(self, prices: TokenPrices) -> float:
        """
        :return: What the resolution's conversations cost, in dollars; a hypothesis the
            knowledge component generated for it is not included.
        """
        return sum(
            prices.cost_of(conversation.usage) for conversation in self.conversations
        )


@dataclass
class KitchenResolver:
    """
    Resolves queries with a reasoner: the task as context, one answer asked for, and the
    source of the queried type shown to the model.
    """

    reasoner: Reasoner
    """
    The resolver with the world it reads and the knowledge component it consults.
    """

    def resolve(self, query: Match, task: str = "") -> Resolution:
        """
        :param query: The query for an action.
        :param task: What the caller is trying to achieve, none when empty.
        :return: The actions the query resolved to.
        """
        backend = self.reasoner.backend(
            context_provider=self.context_for(task),
            number_of_candidates=1,
            source_detail=SourceDetail.FULL,
        )
        return Resolution(
            actions=list(query.evaluate(backend=backend)), backend=backend
        )

    @staticmethod
    def context_for(task: str) -> PromptContextProvider:
        """
        :return: The task as context, or no context when there is none.
        """
        if not task:
            return SchemaOnlyContextProvider()
        return TaskContextProvider(goal=task)
