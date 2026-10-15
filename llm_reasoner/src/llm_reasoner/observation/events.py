"""
The events the system reports about its own work: who asked what of whom, and what came
back. They are written as plain JSON, so a viewer can follow a run as it happens.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from uuid import uuid4

from typing_extensions import Any, Dict, List, Mapping, Optional, Sequence

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


class ResolutionPurpose(StrEnum):
    """
    What a resolution is for.
    """

    ACTION = "action"
    """
    Resolving the action that was asked for.
    """

    SITUATION = "situation"
    """
    Choosing the situation features of the knowledge that describes the action.
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


@dataclass
class ToolRefused(Event):
    """
    The tool box refused a call a model made of a tool, before running it: no such tool
    was offered, or the input did not fit its schema. When it was refused is not known,
    so the event stands just before the next call the tool box ran.
    """

    call: str
    """
    Identifies the refused call.
    """

    parent: str
    """
    The call to the model during which the tool was called.
    """

    tool: str
    """
    The name of the tool the model called.
    """

    arguments: Mapping[str, Any]
    """
    The input of the call.
    """

    content: str
    """
    Why the call was refused, as the model was told.
    """


@dataclass
class ComponentCalled(Event):
    """
    One component called another.
    """

    call: str
    """
    Identifies the call, which what it returned repeats.
    """

    source: Component
    """
    The component that called.
    """

    target: Component
    """
    The component that was called.
    """

    operation: str
    """
    What was asked of it.
    """

    arguments: Mapping[str, Any]
    """
    What it was asked about, an individual or a request named by its kind.
    """


@dataclass
class ComponentReturned(Event):
    """
    A component answered a call from another.
    """

    call: str
    """
    The call that was answered.
    """

    result: Any
    """
    What it answered, an individual or a collection named by its kind or its size.
    """

    seconds: float
    """
    How long the call took.
    """


@dataclass
class ComponentFailed(CallFailed):
    """
    A call from one component to another failed.
    """


# %% the resolver


@dataclass
class SlotSummary:
    """
    A slot of the query the resolver must fill.
    """

    path: str
    """
    Where the slot is in the action, as the model is asked about it.
    """

    kind: str
    """
    What kind of answer the slot takes: a value, an individual, a kind, or none because
    the query gives it.
    """

    type_name: str
    """
    The type the slot declares.
    """

    candidates: Optional[int] = None
    """
    How many individuals the slot may be filled with, for a slot chosen from some.
    """


@dataclass
class GroundSummary:
    """
    One thing an answer rests on.
    """

    source: str
    """
    Where it is: the world or the knowledge.
    """

    identifier: str
    """
    Its stable identifier there.
    """


@dataclass
class JustificationSummary:
    """
    Why an instance was chosen, as the model stated it.
    """

    why: str
    """
    The model's explanation.
    """

    grounds: List[GroundSummary] = field(default_factory=list)
    """
    What the choice rests on.
    """


@dataclass
class ResolutionStarted(Event):
    """
    The resolver began to resolve a query.
    """

    resolution: str
    """
    Identifies the resolution, which the events that follow repeat.
    """

    purpose: ResolutionPurpose
    """
    What the resolution is for.
    """

    action: str
    """
    The module and name of the action the query asks for.
    """

    slots: List[SlotSummary]
    """
    The slots of the query.
    """

    max_repairs: int
    """
    How often an unusable answer is sent back before the resolution gives up.
    """


@dataclass
class KnowledgeConsulted(Event):
    """
    The resolver consulted the knowledge component about the action.
    """

    resolution: str
    """
    The resolution that consulted it.
    """

    designator: str
    """
    The action the knowledge was asked about.
    """

    situation: Mapping[str, str]
    """
    The situation features chosen for the action.
    """

    record_id: Optional[str] = None
    """
    The hypothesis served, none when no knowledge describes the action.
    """

    perspective_record_ids: Sequence[str] = ()
    """
    The hypotheses about the action core's frame and the perspectives on it served, the
    core first; none when no perspective was consulted.
    """


@dataclass
class AttemptStarted(Event):
    """
    The resolver asked the model for an answer.
    """

    resolution: str
    """
    The resolution the attempt belongs to.
    """

    attempt: int
    """
    Which attempt this is, counting from zero; every one after the first repairs the one
    before.
    """


@dataclass
class AttemptRejected(Event):
    """
    The resolver found an answer unusable and sent it back to the model.
    """

    resolution: str
    """
    The resolution the attempt belongs to.
    """

    attempt: int
    """
    The attempt whose answer was rejected.
    """

    kind: str
    """
    Why it was rejected, the name of the failure.
    """

    details: Mapping[str, Any]
    """
    What the failure found, as the model is shown it.
    """

    answer: str
    """
    The answer that was rejected.
    """


@dataclass
class ResolutionFinished(Event):
    """
    The resolver resolved the query.
    """

    resolution: str
    """
    The resolution that finished.
    """

    attempts: int
    """
    How many answers it took.
    """

    instances: List[str]
    """
    The kinds of the instances resolved.
    """

    justifications: List[JustificationSummary] = field(default_factory=list)
    """
    Why each instance was chosen.
    """


@dataclass
class ResolutionFailed(Event):
    """
    The resolver could not resolve the query.
    """

    resolution: str
    """
    The resolution that failed.
    """

    error_type: str
    """
    The name of the failure.
    """

    reason: str
    """
    What it said.
    """


# %% the knowledge component


@dataclass
class HypothesisRequested(Event):
    """
    The knowledge component was asked for a hypothesis.
    """

    root_type: str
    """
    The module and name of the vocabulary class the hypothesis is an instance of.
    """

    fixed_values: Mapping[str, str]
    """
    The values the request fixes, as the model is told them and the store keys them.
    """

    instruction: str
    """
    The action the hypothesis is for, recorded as provenance.
    """


@dataclass
class GenerationAttemptStarted(Event):
    """
    The knowledge component asked the model for a hypothesis.
    """

    attempt: int
    """
    Which attempt this is, counting from zero; every one after the first repairs the one
    before.
    """


@dataclass
class GenerationAttemptRejected(Event):
    """
    The knowledge component found a hypothesis the model gave inadmissible and sent it
    back.
    """

    attempt: int
    """
    The attempt whose answer was rejected.
    """

    problems: List[str]
    """
    What was wrong with the answer.
    """


@dataclass
class GenerationFailed(Event):
    """
    No hypothesis was admitted within the repair budget.
    """

    attempts: int
    """
    How many answers were tried.
    """

    problems: List[str]
    """
    What was wrong with the answer closest to admissible.
    """


@dataclass
class HypothesisServed(Event):
    """
    The knowledge component served a hypothesis.
    """

    record_id: str
    """
    The identifier of the stored hypothesis.
    """

    from_store: bool
    """
    Whether the store held it, rather than it being generated for the request.
    """

    attempts: int
    """
    How many answers generating it took; none when the store held it.
    """

    warnings: List[str] = field(default_factory=list)
    """
    What it was admitted with despite.
    """
