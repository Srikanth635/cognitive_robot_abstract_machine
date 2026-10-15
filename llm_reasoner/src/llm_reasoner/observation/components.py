"""
Reporting the calls between components from outside: the world and the knowledge
component are wrapped, and report what they are asked and what they answer, while the
component asking goes on as before.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from time import perf_counter

from krrood.entity_query_language.query.match import Match
from typing_extensions import Any, Callable, List, Mapping, Optional, TypeVar

from llm_reasoner.clients.tools import Tool
from llm_reasoner.knowledge.hypotheses.knowledge_access import KnowledgeAccess
from llm_reasoner.observation.events import (
    Component,
    ComponentCalled,
    ComponentFailed,
    ComponentReturned,
    new_call,
)
from llm_reasoner.observation.sinks import EventSink
from llm_reasoner.world.access import Holder, WorldAccess, identifier_of

T = TypeVar("T")

# %% what is reported


class WorldOperation(StrEnum):
    """
    What the world is asked.
    """

    HANDLE_OF = "handle_of"
    INDIVIDUAL_OF = "individual_of"
    IS_LIVE = "is_live"
    HOLDERS = "holders"
    TOOLS = "tools"


class KnowledgeOperation(StrEnum):
    """
    What the knowledge component is asked.
    """

    SERVE = "serve"
    TOOLS = "tools"
    CITATION = "citation"
    INSTANCES_OF = "instances_of"
    IS_VOCABULARY_CLASS = "is_vocabulary_class"


class ArgumentKey(StrEnum):
    """
    The keys of what a call is reported to ask about.
    """

    INDIVIDUAL = "individual"
    HANDLE = "handle"
    QUERY = "query"
    INSTRUCTION = "instruction"
    RECORD = "record"
    RECORDS = "records"
    NAME = "name"
    KIND = "kind"


def kind_named(asked: Any) -> str:
    """
    :return: A class by its identifier, anything else by the name of its class.
    """
    return identifier_of(asked) if isinstance(asked, type) else type(asked).__name__


def tool_names(tools: List[Tool]) -> List[str]:
    """
    :return: The names of the tools.
    """
    return [tool.definition.name for tool in tools]


# %% reporting


@dataclass(frozen=True)
class CallReporter:
    """
    Makes calls from one component to another and reports each, how it ends included.
    """

    sink: EventSink
    """
    Where the calls are reported.
    """

    source: Component
    """
    The component calling.
    """

    target: Component
    """
    The component called.
    """

    def made(
        self,
        operation: str,
        arguments: Mapping[str, Any],
        call: Callable[[], T],
        result_of: Callable[[T], Any],
    ) -> T:
        """
        :param operation: What is asked.
        :param arguments: What it is asked about, as it is to be reported.
        :param call: Asks it.
        :param result_of: What the answer is reported as.
        :return: The answer, or the failure going on as it would.
        """
        identifier = new_call()
        self.sink.emit(
            ComponentCalled(
                call=identifier,
                source=self.source,
                target=self.target,
                operation=operation,
                arguments=dict(arguments),
            )
        )
        started = perf_counter()
        try:
            answer = call()
        except Exception as error:
            self.sink.emit(
                ComponentFailed(
                    call=identifier,
                    error_type=type(error).__name__,
                    reason=str(error),
                    seconds=perf_counter() - started,
                )
            )
            raise
        self.sink.emit(
            ComponentReturned(
                call=identifier,
                result=result_of(answer),
                seconds=perf_counter() - started,
            )
        )
        return answer


# %% the world


@dataclass
class ObservedWorld(WorldAccess):
    """
    A world that reports what it is asked and what it answers, and otherwise is the
    world it wraps.
    """

    world: WorldAccess
    """
    The world that answers.
    """

    sink: EventSink
    """
    Where the calls are reported.
    """

    source: Component = Component.RESOLVER
    """
    The component asking.
    """

    reporter: CallReporter = field(init=False, repr=False, compare=False)
    """
    Makes and reports the calls.
    """

    def __post_init__(self) -> None:
        self.reporter = CallReporter(
            sink=self.sink, source=self.source, target=Component.WORLD_INJECTOR
        )

    @property
    def version(self) -> str:
        return self.world.version

    def handle_of(self, individual: Any) -> str:
        return self.reporter.made(
            WorldOperation.HANDLE_OF,
            {ArgumentKey.INDIVIDUAL: type(individual).__name__},
            lambda: self.world.handle_of(individual),
            lambda handle: handle,
        )

    def individual_of(self, handle: str) -> Any:
        return self.reporter.made(
            WorldOperation.INDIVIDUAL_OF,
            {ArgumentKey.HANDLE: handle},
            lambda: self.world.individual_of(handle),
            lambda individual: type(individual).__name__,
        )

    def is_live(self, handle: str) -> bool:
        return self.reporter.made(
            WorldOperation.IS_LIVE,
            {ArgumentKey.HANDLE: handle},
            lambda: self.world.is_live(handle),
            lambda live: live,
        )

    def holders(self, handle: str) -> List[Holder]:
        return self.reporter.made(
            WorldOperation.HOLDERS,
            {ArgumentKey.HANDLE: handle},
            lambda: self.world.holders(handle),
            lambda holders: [asdict(holder) for holder in holders],
        )

    def tools(self) -> List[Tool]:
        return self.reporter.made(
            WorldOperation.TOOLS, {}, self.world.tools, tool_names
        )


# %% the knowledge component


@dataclass
class ObservedKnowledge(KnowledgeAccess):
    """
    A knowledge component that reports what it is asked and what it answers, and
    otherwise is the knowledge component it wraps.
    """

    knowledge: KnowledgeAccess
    """
    The knowledge component that answers.
    """

    sink: EventSink
    """
    Where the calls are reported.
    """

    source: Component = Component.RESOLVER
    """
    The component asking.
    """

    reporter: CallReporter = field(init=False, repr=False, compare=False)
    """
    Makes and reports the calls.
    """

    def __post_init__(self) -> None:
        self.reporter = CallReporter(
            sink=self.sink, source=self.source, target=Component.KNOWLEDGE
        )

    @property
    def version(self) -> str:
        return self.knowledge.version

    def serve(self, query: Match, instruction: str = "") -> str:
        return self.reporter.made(
            KnowledgeOperation.SERVE,
            {
                ArgumentKey.QUERY: kind_named(query.type),
                ArgumentKey.INSTRUCTION: instruction,
            },
            lambda: self.knowledge.serve(query, instruction),
            lambda record_id: record_id,
        )

    def tools(self, *record_ids: str) -> List[Tool]:
        return self.reporter.made(
            KnowledgeOperation.TOOLS,
            (
                {ArgumentKey.RECORD: record_ids[0]}
                if len(record_ids) == 1
                else {ArgumentKey.RECORDS: list(record_ids)}
            ),
            lambda: self.knowledge.tools(*record_ids),
            tool_names,
        )

    def citation(self, record_id: str, name: str) -> Optional[str]:
        return self.reporter.made(
            KnowledgeOperation.CITATION,
            {ArgumentKey.RECORD: record_id, ArgumentKey.NAME: name},
            lambda: self.knowledge.citation(record_id, name),
            lambda cited: cited,
        )

    def instances_of(self, kind: type) -> list:
        return self.reporter.made(
            KnowledgeOperation.INSTANCES_OF,
            {ArgumentKey.KIND: kind_named(kind)},
            lambda: self.knowledge.instances_of(kind),
            len,
        )

    def is_vocabulary_class(self, kind: type) -> bool:
        return self.reporter.made(
            KnowledgeOperation.IS_VOCABULARY_CLASS,
            {ArgumentKey.KIND: kind_named(kind)},
            lambda: self.knowledge.is_vocabulary_class(kind),
            lambda belongs: belongs,
        )
