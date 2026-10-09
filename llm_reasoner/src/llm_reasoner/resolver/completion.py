"""
How the resolver asks a model for an answer: in one completion, or in a conversation in
which the model may call tools first.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import Sequence

from llm_reasoner.clients.base import LanguageModelClient
from llm_reasoner.clients.tools import Tool, ToolConversation, ToolUsingClient


class Completion(ABC):
    """
    Asks a model for an answer.
    """

    @abstractmethod
    def complete(self, system: str, user: str) -> ToolConversation:
        """
        :param system: The system prompt.
        :param user: The user prompt.
        :return: The model's answer, with the tool calls it made on the way.
        """

    @property
    @abstractmethod
    def identity(self) -> str:
        """
        Which model answers, as its client reports it.
        """


@dataclass(frozen=True)
class PlainCompletion(Completion):
    """
    Asks in one completion, without tools.
    """

    client: LanguageModelClient
    """
    The client asked.
    """

    def complete(self, system: str, user: str) -> ToolConversation:
        return ToolConversation(answer=self.client.complete(system, user))

    @property
    def identity(self) -> str:
        return self.client.identity


@dataclass(frozen=True)
class ToolCompletion(Completion):
    """
    Asks in a conversation in which the model may call tools before it answers.
    """

    client: ToolUsingClient
    """
    The client asked.
    """

    tools: Sequence[Tool]
    """
    The tools the model may call.
    """

    def complete(self, system: str, user: str) -> ToolConversation:
        return self.client.complete_with_tools(system, user, self.tools)

    @property
    def identity(self) -> str:
        return self.client.identity
