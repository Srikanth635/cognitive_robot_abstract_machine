"""
The events the system reports about its own work: who asked what of whom, and what came
back. They are written as plain JSON, so a viewer can follow a run as it happens.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from uuid import uuid4

from typing_extensions import Any, Dict, List, Mapping, Optional

from llm_reasoner.clients.usage import TokenUsage

# %% what an event is about


class Component(StrEnum):
    """
    The parts of the system a call goes between.
    """

    CALLER = "caller"
    """
    What asks the system to resolve something, such as a plan or a viewer.
    """

    RESOLVER = "resolver"
    """
    Turns an underspecified query into a resolved one.
    """

    WORLD_INJECTOR = "world_injector"
    """
    Answers questions about the live world.
    """

    KNOWLEDGE = "knowledge"
    """
    Generates and serves the knowledge about actions.
    """

    LANGUAGE_MODEL = "language_model"
    """
    The model that is asked.
    """

    SIMULATION = "simulation"
    """
    The simulation a resolved action is tried in.
    """


class CompletionStyle(StrEnum):
    """
    How a language model is asked to complete.
    """

    PLAIN = "plain"
    """
    One answer in free text.
    """

    STRUCTURED = "structured"
    """
    One answer that must satisfy a JSON schema.
    """

    WITH_TOOLS = "with_tools"
    """
    An answer after the model may have called tools.
    """


def new_call() -> str:
    """
    :return: The identifier of a new call, which ties the event that opens it to the one
        that closes it.
    """
    return uuid4().hex


# %% events


class EventKey(StrEnum):
    """
    The keys an event is written with besides its own fields.
    """

    EVENT = "event"
    """
    The kind of the event, the name of its class.
    """


@dataclass(kw_only=True)
class Event:
    """
    Something the system reports about its work.
    """

    at: float = field(default_factory=time.time)
    """
    When it happened, in seconds since the epoch.
    """

    def to_json(self) -> Dict[str, Any]:
        """
        :return: The event as plain JSON: its kind, then its fields.
        """
        return {EventKey.EVENT: type(self).__name__, **asdict(self)}


@dataclass
class ModelAsked(Event):
    """
    A language model was asked to complete.
    """

    call: str
    """
    Identifies the call, which the answer or the failure ending it repeats.
    """

    source: Component
    """
    The component that asked.
    """

    style: CompletionStyle
    """
    How the model was asked.
    """

    identity: str
    """
    The client and model that were asked.
    """

    system: str
    """
    The system prompt.
    """

    user: str
    """
    The user prompt.
    """

    schema: Optional[Mapping[str, Any]] = None
    """
    The JSON schema the answer had to satisfy, when it had to.
    """

    tools: List[str] = field(default_factory=list)
    """
    The names of the tools the model could call.
    """


@dataclass
class ModelAnswered(Event):
    """
    A language model completed.
    """

    call: str
    """
    The call that was answered.
    """

    answer: str
    """
    What the model answered.
    """

    seconds: float
    """
    How long the call took.
    """

    usage: Optional[TokenUsage] = None
    """
    The tokens the call used, when the client reports them.
    """

    cost: Optional[float] = None
    """
    What the call cost in dollars, when the tokens and their prices are known.
    """


@dataclass
class CallFailed(Event):
    """
    A call ended by raising an error.
    """

    call: str
    """
    The call that failed.
    """

    error_type: str
    """
    The name of the error raised.
    """

    reason: str
    """
    What the error said.
    """

    seconds: float
    """
    How long the call ran before it failed.
    """


@dataclass
class ModelFailed(CallFailed):
    """
    A call to a language model failed.
    """


@dataclass
class ToolCalled(Event):
    """
    A language model called a tool.
    """

    call: str
    """
    Identifies the tool call, which what it returned repeats.
    """

    parent: str
    """
    The call to the model during which the tool was called.
    """

    tool: str
    """
    The name of the tool.
    """

    arguments: Mapping[str, Any]
    """
    The input of the call.
    """

    owner: Optional[Component] = None
    """
    The component the tool belongs to, when it is known.
    """


@dataclass
class ToolReturned(Event):
    """
    A tool returned to the model.
    """

    call: str
    """
    The tool call that returned.
    """

    content: str
    """
    What it returned, or what went wrong.
    """

    is_error: bool
    """
    Whether the call could not be answered.
    """

    seconds: float
    """
    How long the call took.
    """


@dataclass
class ToolFailed(CallFailed):
    """
    A tool raised instead of returning to the model.
    """
