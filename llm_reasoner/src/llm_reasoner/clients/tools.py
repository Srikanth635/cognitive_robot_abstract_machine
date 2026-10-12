"""
Tools a language model may call during a completion, independent of the vendor serving
the model: what a tool is, how a call is checked and run, and the record of the calls a
completion made.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

from jsonschema import Draft202012Validator
from typing_extensions import Any, Dict, List, Mapping, Optional, Sequence, Union

from llm_reasoner.clients.usage import TokenUsage
from llm_reasoner.exceptions import LLMReasonerError


class ToolDefinitionKey(StrEnum):
    """
    The keys of a tool definition, as the model is offered it.
    """

    NAME = "name"
    DESCRIPTION = "description"
    INPUT_SCHEMA = "input_schema"
    OUTPUT_SCHEMA = "output_schema"


class ToolDescriptionKey(StrEnum):
    """
    The parts of a tool's description, each stated on a line of its own.
    """

    PURPOSE = "purpose"
    """
    What the tool does.
    """

    RETURNS = "returns"
    """
    What a call answers.
    """

    USE_WHEN = "use_when"
    """
    When to call the tool.
    """

    AVOID = "avoid"
    """
    When not to call it, and what to do instead.
    """


@dataclass(frozen=True)
class ToolDescription:
    """
    What a model is told a tool is for, in labelled parts.
    """

    purpose: str
    """
    What the tool does.
    """

    returns: str
    """
    What a call answers.
    """

    use_when: Optional[str] = None
    """
    When to call the tool, if that needs saying.
    """

    avoid: Optional[str] = None
    """
    When not to call it, and what to do instead, if that needs saying.
    """

    @classmethod
    def from_json(cls, stored: Mapping[str, str]) -> ToolDescription:
        """
        :return: The description stored as a JSON object of its parts.
        """
        return cls(
            purpose=stored[ToolDescriptionKey.PURPOSE],
            returns=stored[ToolDescriptionKey.RETURNS],
            use_when=stored.get(ToolDescriptionKey.USE_WHEN),
            avoid=stored.get(ToolDescriptionKey.AVOID),
        )

    def render(self) -> str:
        """
        :return: The description as the model reads it, one labelled part per line.
        """
        parts = {
            ToolDescriptionKey.PURPOSE: self.purpose,
            ToolDescriptionKey.RETURNS: self.returns,
            ToolDescriptionKey.USE_WHEN: self.use_when,
            ToolDescriptionKey.AVOID: self.avoid,
        }
        return "\n".join(
            f"{key}: {part}" for key, part in parts.items() if part is not None
        )


@dataclass(frozen=True)
class ToolDefinition:
    """
    What the model is told about a tool: its name, what it does, and the input it takes.
    """

    name: str
    """
    The name the model calls the tool by.
    """

    description: ToolDescription
    """
    What the tool does and when to call it.
    """

    input_schema: Mapping[str, Any]
    """
    The JSON schema of the input the tool takes.
    """

    output_schema: Optional[Mapping[str, Any]] = None
    """
    The JSON schema of the answers the tool gives when it succeeds, if declared; it is a
    contract for the tool's users and is not sent to the model.
    """

    @classmethod
    def from_file(cls, path: Path) -> ToolDefinition:
        """
        :return: The definition stored as JSON in a file.
        """
        stored = json.loads(path.read_text())
        return cls(
            name=stored[ToolDefinitionKey.NAME],
            description=ToolDescription.from_json(
                stored[ToolDefinitionKey.DESCRIPTION]
            ),
            input_schema=stored[ToolDefinitionKey.INPUT_SCHEMA],
            output_schema=stored.get(ToolDefinitionKey.OUTPUT_SCHEMA),
        )

    def sharing(self, definitions: Mapping[str, Any]) -> ToolDefinition:
        """
        :param definitions: Schema definitions the answers of several tools share.
        :return: The definition, its answer shape holding the shared definitions it
            refers to.
        """
        return replace(
            self,
            output_schema={**self.output_schema, SCHEMA_DEFINITIONS: definitions},
        )

    def as_request(self) -> Dict[str, Any]:
        """
        :return: The definition as a request offers it to the model.
        """
        return {
            ToolDefinitionKey.NAME: self.name,
            ToolDefinitionKey.DESCRIPTION: self.description.render(),
            ToolDefinitionKey.INPUT_SCHEMA: dict(self.input_schema),
        }


class ToolErrorKey(StrEnum):
    """
    The keys of an error a tool call returns.
    """

    ERROR = "error"
    """
    What went wrong, named.
    """

    TOOL = "tool"
    """
    The tool called.
    """

    TOOLS = "tools"
    """
    The tools offered.
    """

    PROBLEMS = "problems"
    """
    Each way the input breaks the tool's schema.
    """


SCHEMA_DEFINITIONS = "$defs"
"""
The JSON schema keyword under which a schema holds the definitions it refers to.
"""

TOOL_ERROR_SCHEMA_FILE = Path(__file__).parent / "tool_error_schema.json"
"""
Where the JSON schema every tool error follows is stored.
"""


def tool_error_schema() -> Mapping[str, Any]:
    """
    :return: The JSON schema every error a tool call returns follows.
    """
    return json.loads(TOOL_ERROR_SCHEMA_FILE.read_text())


class ToolErrorKind(StrEnum):
    """
    The errors a tool box reports before any tool runs.
    """

    UNKNOWN_TOOL = "UnknownTool"
    """
    No offered tool has the name called.
    """

    INPUT_BREAKS_SCHEMA = "InputBreaksSchema"
    """
    The input does not fit the tool's schema.
    """


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

    @classmethod
    def error(cls, kind: str, details: Mapping[str, Any]) -> ToolOutcome:
        """
        :param kind: What went wrong, named.
        :param details: What the error found.
        :return: An error outcome stating the error and its details as JSON.
        """
        return cls(
            content=json.dumps({ToolErrorKey.ERROR: kind, **details}), is_error=True
        )


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
            return ToolOutcome.error(
                ToolErrorKind.UNKNOWN_TOOL,
                {
                    ToolErrorKey.TOOL: name,
                    ToolErrorKey.TOOLS: [
                        offered.definition.name for offered in self.tools
                    ],
                },
            )
        problems = [
            problem.message
            for problem in Draft202012Validator(
                tool.definition.input_schema
            ).iter_errors(arguments)
        ]
        if problems:
            return ToolOutcome.error(
                ToolErrorKind.INPUT_BREAKS_SCHEMA,
                {ToolErrorKey.TOOL: name, ToolErrorKey.PROBLEMS: problems},
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
