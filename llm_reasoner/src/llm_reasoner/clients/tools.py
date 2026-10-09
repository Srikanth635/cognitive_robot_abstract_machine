"""
Tools a language model may call during a completion, independent of the vendor serving
the model: what a tool is, how a call is checked and run, and the record of the calls a
completion made.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from jsonschema import Draft202012Validator
from typing_extensions import Any, Dict, List, Mapping, Sequence, Union

from llm_reasoner.clients.usage import TokenUsage
from llm_reasoner.exceptions import LLMReasonerError


class ToolDefinitionKey(StrEnum):
    """
    The keys of a tool definition, as the model is offered it.
    """

    NAME = "name"
    DESCRIPTION = "description"
    INPUT_SCHEMA = "input_schema"


@dataclass(frozen=True)
class ToolDefinition:
    """
    What the model is told about a tool: its name, what it does, and the input it takes.
    """

    name: str
    """
    The name the model calls the tool by.
    """

    description: str
    """
    What the tool does and when to call it.
    """

    input_schema: Mapping[str, Any]
    """
    The JSON schema of the input the tool takes.
    """

    @classmethod
    def from_file(cls, path: Path) -> ToolDefinition:
        """
        :return: The definition stored as JSON in a file.
        """
        stored = json.loads(path.read_text())
        return cls(
            name=stored[ToolDefinitionKey.NAME],
            description=stored[ToolDefinitionKey.DESCRIPTION],
            input_schema=stored[ToolDefinitionKey.INPUT_SCHEMA],
        )

    def as_request(self) -> Dict[str, Any]:
        """
        :return: The definition as a request offers it to the model.
        """
        return {
            ToolDefinitionKey.NAME: self.name,
            ToolDefinitionKey.DESCRIPTION: self.description,
            ToolDefinitionKey.INPUT_SCHEMA: dict(self.input_schema),
        }


@dataclass(frozen=True)
class ToolOutcome:
    """
    What running a tool returned to the model.
    """

    content: str
    """
    The result, or what went wrong.
    """

    is_error: bool = False
    """
    Whether the call could not be answered.
    """


class Tool(ABC):
    """
    Something a language model may call during a completion.
    """

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """
        What the model is told about the tool.
        """

    @abstractmethod
    def run(self, arguments: Mapping[str, Any]) -> ToolOutcome:
        """
        :param arguments: The input of the call, already checked against the schema.
        :return: What the call returns to the model.
        """


@dataclass(frozen=True)
class ToolCall:
    """
    One call a model made during a completion, with what it returned.
    """

    name: str
    """
    The name of the tool called.
    """

    arguments: Mapping[str, Any]
    """
    The input of the call.
    """

    outcome: ToolOutcome
    """
    What the call returned to the model.
    """


@dataclass(frozen=True)
class ToolBox:
    """
    The tools offered in a completion: checks each call against its tool's schema and
    runs it.
    """

    tools: Sequence[Tool]
    """
    The tools offered.
    """

    def definitions(self) -> List[Dict[str, Any]]:
        """
        :return: The definitions of the tools, as a request offers them.
        """
        return [tool.definition.as_request() for tool in self.tools]

    def call(self, name: str, arguments: Mapping[str, Any]) -> ToolOutcome:
        """
        :return: What the call returns: the tool's result, or an error naming what was
            wrong with the call, so the model can correct it.
        """
        tool = next((tool for tool in self.tools if tool.definition.name == name), None)
        if tool is None:
            offered = [offered.definition.name for offered in self.tools]
            return ToolOutcome(
                content=f"No tool is named {name!r}; the tools are {offered}",
                is_error=True,
            )
        problems = [
            problem.message
            for problem in Draft202012Validator(
                tool.definition.input_schema
            ).iter_errors(arguments)
        ]
        if problems:
            return ToolOutcome(
                content=f"The input does not fit {name}: " + "; ".join(problems),
                is_error=True,
            )
        return tool.run(arguments)


@dataclass(frozen=True)
class ToolConversation:
    """
    A completion in which the model called tools before answering.
    """

    answer: str
    """
    The model's final answer.
    """

    calls: List[ToolCall] = field(default_factory=list)
    """
    The calls it made, in order, with what each returned.
    """

    usage: TokenUsage = field(default_factory=TokenUsage)
    """
    The tokens the whole conversation consumed, every turn added up.
    """


class ToolUsingClient(ABC):
    """
    A client completing prompts in which the model may call tools before it answers.
    """

    @abstractmethod
    def complete_with_tools(
        self, system: str, user: str, tools: Sequence[Tool]
    ) -> ToolConversation:
        """
        :param system: The system prompt.
        :param user: The user prompt.
        :param tools: The tools the model may call.
        :return: The model's answer and the calls it made.
        :raises ToolTurnsExhausted: When the model keeps calling tools past the
            client's budget of turns.
        """


@dataclass(frozen=True)
class ScriptedToolRequest:
    """
    A tool call a scripted model makes.
    """

    name: str
    """
    The name of the tool called.
    """

    arguments: Mapping[str, Any]
    """
    The input of the call.
    """


ScriptedTurn = Union[List[ScriptedToolRequest], str]
"""
One turn of a scripted model: the tool calls it makes, or its final answer.
"""


@dataclass
class ToolTurnsExhausted(LLMReasonerError):
    """
    The model kept calling tools past the client's budget of turns.
    """

    turns: int
    """
    How many turns the client allowed.
    """

    def __post_init__(self):
        super().__init__(f"The model was still calling tools after {self.turns} turns")
