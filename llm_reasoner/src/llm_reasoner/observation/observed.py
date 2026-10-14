"""
Reporting the calls between components from outside: a language-model client and the
tools it runs are wrapped, and report what they are asked and what they answer, while
the components using them go on as before.
"""

from __future__ import annotations

from contextlib import contextmanager
import time
from dataclasses import dataclass, field
from time import perf_counter

from typing_extensions import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from llm_reasoner.clients.base import LanguageModelClient
from llm_reasoner.clients.tools import (
    Tool,
    ToolConversation,
    ToolDefinition,
    ToolOutcome,
    ToolUsingClient,
)
from llm_reasoner.clients.usage import TokenPrices, TokenUsage, UsageReporting
from llm_reasoner.observation.events import (
    Component,
    CompletionStyle,
    ModelAnswered,
    ModelAsked,
    ModelFailed,
    ToolCalled,
    ToolFailed,
    ToolRefused,
    ToolReturned,
    new_call,
)
from llm_reasoner.observation.sinks import EventSink

# %% tools


@dataclass(frozen=True)
class ToolOwner:
    """
    The component that a kind of tool belongs to.
    """

    tool_type: type
    """
    The tools of this type, and of the types specialising it.
    """

    component: Component
    """
    The component they belong to.
    """


JUST_BEFORE = 1e-6
"""
How long before the call the tool box ran next a refused call is said to have been
made, in seconds.
"""


@dataclass(frozen=True)
class ToolRun:
    """
    A call of a tool that the tool box ran.
    """

    tool: str
    """
    The name of the tool.
    """

    arguments: Dict[str, Any]
    """
    The input of the call.
    """

    at: float
    """
    When the call was made, in seconds since the epoch.
    """


@dataclass
class ObservedTool(Tool):
    """
    A tool that reports each call the model makes of it and what comes back, and
    otherwise is the tool it wraps.
    """

    tool: Tool
    """
    The tool that answers.
    """

    sink: EventSink
    """
    Where the calls are reported.
    """

    parent: str
    """
    The call to the model during which the tool is offered.
    """

    owner: Optional[Component] = None
    """
    The component the tool belongs to, when it is known.
    """

    runs: List[ToolRun] = field(default_factory=list)
    """
    Where each call the tool box runs is noted, shared by the tools offered in one
    completion, so the calls the tool box refused can be told from those it ran.
    """

    @property
    def definition(self) -> ToolDefinition:
        return self.tool.definition

    def run(self, arguments: Mapping[str, Any]) -> ToolOutcome:
        call = new_call()
        called = ToolCalled(
            call=call,
            parent=self.parent,
            tool=self.definition.name,
            arguments=dict(arguments),
            owner=self.owner,
        )
        self.sink.emit(called)
        self.runs.append(
            ToolRun(tool=called.tool, arguments=called.arguments, at=called.at)
        )
        started = perf_counter()
        try:
            outcome = self.tool.run(arguments)
        except Exception as error:
            self.sink.emit(
                ToolFailed(
                    call=call,
                    error_type=type(error).__name__,
                    reason=str(error),
                    seconds=perf_counter() - started,
                )
            )
            raise
        self.sink.emit(
            ToolReturned(
                call=call,
                content=outcome.content,
                is_error=outcome.is_error,
                seconds=perf_counter() - started,
            )
        )
        return outcome


# %% clients


@dataclass(frozen=True)
class AskedCall:
    """
    A call to a language model that has been reported as asked.
    """

    id: str
    """
    The identifier of the call.
    """

    started: float
    """
    When the call started, on the clock durations are measured with.
    """


@dataclass
class ObservedClient(LanguageModelClient):
    """
    A client that reports each completion it is asked for and how it ends, and otherwise
    is the client it wraps, with its identity and abilities.
    """

    client: LanguageModelClient
    """
    The client that completes.
    """

    sink: EventSink
    """
    Where the completions are reported.
    """

    source: Component
    """
    The component asking through this client.
    """

    prices: Optional[TokenPrices] = None
    """
    What the model charges, to report what a completion cost.
    """

    @classmethod
    def of(
        cls,
        client: LanguageModelClient,
        sink: EventSink,
        source: Component,
        prices: Optional[TokenPrices] = None,
        tool_owners: Sequence[ToolOwner] = (),
    ) -> ObservedClient:
        """
        :param client: The client to observe.
        :param sink: Where its completions are reported.
        :param source: The component asking through it.
        :param prices: What the model charges.
        :param tool_owners: Which component each kind of tool belongs to; of no use to a
            client that cannot let a model call tools.
        :return: The observed client, one that lets a model call tools if the client
            does and one that does not if it does not.
        """
        if isinstance(client, ToolUsingClient):
            return ObservedToolClient(
                client=client,
                sink=sink,
                source=source,
                prices=prices,
                tool_owners=tuple(tool_owners),
            )
        return cls(client=client, sink=sink, source=source, prices=prices)

    @property
    def identity(self) -> str:
        return self.client.identity

    @property
    def supports_structured_output(self) -> bool:
        return self.client.supports_structured_output

    def complete(self, system: str, user: str) -> str:
        with self._asking(CompletionStyle.PLAIN, system, user) as call:
            answer = self.client.complete(system, user)
            self._answered(call, answer, self._latest_usage())
            return answer

    def complete_structured(
        self, system: str, user: str, json_schema: Mapping[str, Any]
    ) -> str:
        with self._asking(
            CompletionStyle.STRUCTURED, system, user, schema=json_schema
        ) as call:
            answer = self.client.complete_structured(system, user, json_schema)
            self._answered(call, answer, self._latest_usage())
            return answer

    def _latest_usage(self) -> Optional[TokenUsage]:
        """
        :return: The tokens the client says its latest completion used, ``None`` when it
            does not say.
        """
        if isinstance(self.client, UsageReporting):
            return self.client.latest_usage
        return None

    @contextmanager
    def _asking(
        self,
        style: CompletionStyle,
        system: str,
        user: str,
        schema: Optional[Mapping[str, Any]] = None,
        tools: Sequence[str] = (),
    ) -> Iterator[AskedCall]:
        """
        Report a completion as asked, and as failed if it ends by raising, which then
        goes on.
        """
        call = AskedCall(id=new_call(), started=perf_counter())
        self.sink.emit(
            ModelAsked(
                call=call.id,
                source=self.source,
                style=style,
                identity=self.client.identity,
                system=system,
                user=user,
                schema=schema,
                tools=list(tools),
            )
        )
        try:
            yield call
        except Exception as error:
            self.sink.emit(
                ModelFailed(
                    call=call.id,
                    error_type=type(error).__name__,
                    reason=str(error),
                    seconds=perf_counter() - call.started,
                )
            )
            raise

    def _answered(
        self, call: AskedCall, answer: str, usage: Optional[TokenUsage] = None
    ) -> None:
        """
        Report a completion as answered, with what it cost where that is known.
        """
        self.sink.emit(
            ModelAnswered(
                call=call.id,
                answer=answer,
                seconds=perf_counter() - call.started,
                usage=usage,
                cost=(
                    self.prices.cost_of(usage)
                    if self.prices is not None and usage is not None
                    else None
                ),
            )
        )


@dataclass
class ObservedToolClient(ObservedClient, ToolUsingClient):
    """
    An observed client whose model may call tools, which report their calls too.
    """

    tool_owners: Tuple[ToolOwner, ...] = ()
    """
    Which component each kind of tool belongs to.
    """

    def owner_of(self, tool: Tool) -> Optional[Component]:
        """
        :return: The component the tool belongs to, ``None`` when no owner is known.
        """
        return next(
            (
                owner.component
                for owner in self.tool_owners
                if isinstance(tool, owner.tool_type)
            ),
            None,
        )

    def complete_with_tools(
        self, system: str, user: str, tools: Sequence[Tool]
    ) -> ToolConversation:
        with self._asking(
            CompletionStyle.WITH_TOOLS,
            system,
            user,
            tools=[tool.definition.name for tool in tools],
        ) as call:
            runs: List[ToolRun] = []
            conversation = self.client.complete_with_tools(
                system,
                user,
                [
                    ObservedTool(
                        tool=tool,
                        sink=self.sink,
                        parent=call.id,
                        owner=self.owner_of(tool),
                        runs=runs,
                    )
                    for tool in tools
                ],
            )
            self._refused(call, conversation, runs)
            self._answered(call, conversation.answer, conversation.usage)
            return conversation

    def _refused(
        self, call: AskedCall, conversation: ToolConversation, runs: List[ToolRun]
    ) -> None:
        """
        Report the calls of a conversation that the tool box refused: those no tool ran.
        """
        ran = iter(runs)
        next_run = next(ran, None)
        for made in conversation.calls:
            if (
                next_run is not None
                and next_run.tool == made.name
                and next_run.arguments == dict(made.arguments)
            ):
                next_run = next(ran, None)
                continue
            self.sink.emit(
                ToolRefused(
                    at=(
                        next_run.at - JUST_BEFORE
                        if next_run is not None
                        else time.time()
                    ),
                    call=new_call(),
                    parent=call.id,
                    tool=made.name,
                    arguments=dict(made.arguments),
                    content=made.outcome.content,
                )
            )
