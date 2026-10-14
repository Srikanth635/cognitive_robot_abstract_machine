"""
Tests of observing the components' calls from outside: a language-model client and the
tools it runs report what they are asked and what they answer, without the components
behind them changing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from typing_extensions import Any, Mapping, Sequence

from llm_reasoner.clients.base import (
    LanguageModelClient,
    ScriptedClient,
    ScriptExhausted,
)
from llm_reasoner.clients.tools import (
    ScriptedToolRequest,
    Tool,
    ToolConversation,
    ToolDefinition,
    ToolDescription,
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
    ToolReturned,
)
from llm_reasoner.observation.observed import (
    ObservedClient,
    ObservedToolClient,
    ToolOwner,
)
from llm_reasoner.observation.sinks import MemorySink

# %% helpers

ECHO = "echo"
"""
The name of the tool that answers with its input.
"""

FAILURE = "the tool broke"
"""
What the failing tool says.
"""


class ToolAnsweringWithItsInput(Tool):
    """
    A tool that answers with the input it was called with.
    """

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=ECHO,
            description=ToolDescription(
                purpose="Answers with its input.", returns="The input."
            ),
            input_schema={"type": "object"},
        )

    def run(self, arguments: Mapping[str, Any]) -> ToolOutcome:
        return ToolOutcome(content=json.dumps(arguments))


class ToolThatFails(ToolAnsweringWithItsInput):
    """
    A tool that raises instead of answering.
    """

    def run(self, arguments: Mapping[str, Any]) -> ToolOutcome:
        raise RuntimeError(FAILURE)


@dataclass
class ClientWithoutTools(LanguageModelClient):
    """
    A client that completes plainly and cannot let a model call tools.
    """

    def complete(self, system: str, user: str) -> str:
        return "plain"


@dataclass
class ClientCountingTokens(LanguageModelClient, ToolUsingClient):
    """
    A client whose conversations report the tokens they used.
    """

    usage: TokenUsage

    def complete(self, system: str, user: str) -> str:
        return "plain"

    def complete_with_tools(
        self, system: str, user: str, tools: Sequence[Tool]
    ) -> ToolConversation:
        return ToolConversation(answer="counted", usage=self.usage)


@dataclass
class ClientReportingItsUsage(LanguageModelClient, UsageReporting):
    """
    A client that says what its latest plain or structured completion used.
    """

    used: TokenUsage

    @property
    def latest_usage(self) -> TokenUsage:
        return self.used

    def complete(self, system: str, user: str) -> str:
        return "plain"


def observing(client: LanguageModelClient, **settings: Any):
    """
    :return: The client observed by a memory sink, which is returned beside it.
    """
    sink = MemorySink()
    return ObservedClient.of(client, sink, Component.RESOLVER, **settings), sink


def kinds(sink: MemorySink) -> list:
    """
    :return: The kinds of the events the sink holds, in order.
    """
    return [type(event) for event in sink.events]


# %% a completion


class TestCompletions:
    def test_a_plain_completion_returns_what_the_client_answered(self):
        client, _ = observing(ScriptedClient(responses=["RIGHT"]))
        assert client.complete("system", "user") == "RIGHT"

    def test_a_plain_completion_is_reported_with_its_prompts_and_its_answer(self):
        client, sink = observing(ScriptedClient(responses=["RIGHT"]))
        client.complete("system", "user")
        asked, answered = sink.events
        assert (type(asked), type(answered)) == (ModelAsked, ModelAnswered)
        assert (asked.system, asked.user) == ("system", "user")
        assert asked.style is CompletionStyle.PLAIN
        assert asked.source is Component.RESOLVER
        assert answered.answer == "RIGHT"
        assert answered.call == asked.call

    def test_a_completion_is_reported_as_asked_by_the_component_that_asked(self):
        sink = MemorySink()
        client = ObservedClient.of(
            ScriptedClient(responses=["ok"]), sink, Component.KNOWLEDGE
        )
        client.complete("system", "user")
        assert sink.events[0].source is Component.KNOWLEDGE

    def test_a_structured_completion_is_reported_with_its_schema(self):
        schema = {"type": "object", "properties": {"arm": {"type": "string"}}}
        client, sink = observing(ScriptedClient(responses=["{}"]))
        assert client.complete_structured("system", "user", schema) == "{}"
        asked = sink.events[0]
        assert asked.style is CompletionStyle.STRUCTURED
        assert asked.schema == schema

    def test_the_time_a_completion_took_is_reported(self):
        client, sink = observing(ScriptedClient(responses=["ok"]))
        client.complete("system", "user")
        asked, answered = sink.events
        assert answered.seconds >= 0.0
        assert answered.at >= asked.at

    def test_a_failed_completion_is_reported_and_fails_as_before(self):
        client, sink = observing(ScriptedClient())
        with pytest.raises(ScriptExhausted):
            client.complete("system", "user")
        asked, failed = sink.events
        assert type(failed) is ModelFailed
        assert failed.call == asked.call
        assert failed.error_type == ScriptExhausted.__name__


# %% what the wrapped client is


class TestTheWrappedClient:
    def test_the_identity_is_the_wrapped_clients(self):
        wrapped = ScriptedClient()
        client, _ = observing(wrapped)
        assert client.identity == wrapped.identity

    def test_structured_output_is_supported_as_the_wrapped_client_does(self):
        wrapped = ScriptedClient()
        client, _ = observing(wrapped)
        assert client.supports_structured_output == wrapped.supports_structured_output

    def test_a_client_that_lets_a_model_call_tools_still_does(self):
        client, _ = observing(ScriptedClient())
        assert isinstance(client, ToolUsingClient)
        assert isinstance(client, ObservedToolClient)

    def test_a_client_that_cannot_let_a_model_call_tools_still_cannot(self):
        client, _ = observing(ClientWithoutTools())
        assert not isinstance(client, ToolUsingClient)


# %% a completion with tools


class TestToolCompletions:
    def test_the_tools_offered_are_reported(self):
        client, sink = observing(ScriptedClient(tool_turns=["done"]))
        client.complete_with_tools("system", "user", [ToolAnsweringWithItsInput()])
        asked = sink.events[0]
        assert asked.style is CompletionStyle.WITH_TOOLS
        assert asked.tools == [ECHO]

    def test_the_conversation_is_what_the_wrapped_client_returned(self):
        client, _ = observing(
            ScriptedClient(tool_turns=[[ScriptedToolRequest(ECHO, {"x": 1})], "done"])
        )
        conversation = client.complete_with_tools(
            "system", "user", [ToolAnsweringWithItsInput()]
        )
        assert conversation.answer == "done"
        assert [call.name for call in conversation.calls] == [ECHO]

    def test_a_tool_call_is_reported_between_the_question_and_the_answer(self):
        client, sink = observing(
            ScriptedClient(tool_turns=[[ScriptedToolRequest(ECHO, {"x": 1})], "done"])
        )
        client.complete_with_tools("system", "user", [ToolAnsweringWithItsInput()])
        assert kinds(sink) == [ModelAsked, ToolCalled, ToolReturned, ModelAnswered]

    def test_a_tool_call_is_reported_with_its_input_and_what_it_returned(self):
        client, sink = observing(
            ScriptedClient(tool_turns=[[ScriptedToolRequest(ECHO, {"x": 1})], "done"])
        )
        client.complete_with_tools("system", "user", [ToolAnsweringWithItsInput()])
        asked, called, returned, _ = sink.events
        assert called.tool == ECHO
        assert called.arguments == {"x": 1}
        assert called.parent == asked.call
        assert returned.call == called.call
        assert returned.content == json.dumps({"x": 1})
        assert returned.is_error is False

    def test_a_tool_is_reported_with_the_component_that_owns_it(self):
        client, sink = observing(
            ScriptedClient(tool_turns=[[ScriptedToolRequest(ECHO, {})], "done"]),
            tool_owners=(
                ToolOwner(
                    tool_type=ToolAnsweringWithItsInput,
                    component=Component.WORLD_INJECTOR,
                ),
            ),
        )
        client.complete_with_tools("system", "user", [ToolAnsweringWithItsInput()])
        assert sink.events[1].owner is Component.WORLD_INJECTOR

    def test_a_tool_nobody_owns_is_reported_without_an_owner(self):
        client, sink = observing(
            ScriptedClient(tool_turns=[[ScriptedToolRequest(ECHO, {})], "done"]),
        )
        client.complete_with_tools("system", "user", [ToolAnsweringWithItsInput()])
        assert sink.events[1].owner is None

    def test_a_tool_that_fails_is_reported_and_its_completion_fails_as_before(self):
        client, sink = observing(
            ScriptedClient(tool_turns=[[ScriptedToolRequest(ECHO, {})], "done"])
        )
        with pytest.raises(RuntimeError, match=FAILURE):
            client.complete_with_tools("system", "user", [ToolThatFails()])
        assert kinds(sink) == [ModelAsked, ToolCalled, ToolFailed, ModelFailed]
        failed = sink.events[2]
        assert failed.error_type == RuntimeError.__name__
        assert failed.reason == FAILURE
        assert failed.call == sink.events[1].call


# %% tokens and cost


class TestCost:
    USAGE = TokenUsage(input_tokens=1000, output_tokens=200)
    PRICES = TokenPrices(
        input_tokens=3.0,
        output_tokens=15.0,
        cache_creation_input_tokens=3.75,
        cache_read_input_tokens=0.3,
    )

    def test_a_conversation_reports_the_tokens_it_used(self):
        client, sink = observing(ClientCountingTokens(usage=self.USAGE))
        client.complete_with_tools("system", "user", [])
        assert sink.events[-1].usage == self.USAGE

    def test_a_conversation_reports_what_it_cost_when_prices_are_known(self):
        client, sink = observing(
            ClientCountingTokens(usage=self.USAGE), prices=self.PRICES
        )
        client.complete_with_tools("system", "user", [])
        assert sink.events[-1].cost == self.PRICES.cost_of(self.USAGE)

    def test_a_conversation_reports_no_cost_without_prices(self):
        client, sink = observing(ClientCountingTokens(usage=self.USAGE))
        client.complete_with_tools("system", "user", [])
        assert sink.events[-1].cost is None

    def test_a_plain_completion_reports_the_tokens_its_client_says_it_used(self):
        client, sink = observing(ClientReportingItsUsage(used=self.USAGE))
        client.complete("system", "user")
        assert sink.events[-1].usage == self.USAGE

    def test_a_structured_completion_reports_the_tokens_its_client_says_it_used(self):
        client, sink = observing(ClientReportingItsUsage(used=self.USAGE))
        client.complete_structured("system", "user", {"type": "object"})
        assert sink.events[-1].usage == self.USAGE

    def test_a_completion_reports_what_it_cost_when_prices_are_known(self):
        client, sink = observing(
            ClientReportingItsUsage(used=self.USAGE), prices=self.PRICES
        )
        client.complete("system", "user")
        assert sink.events[-1].cost == self.PRICES.cost_of(self.USAGE)

    def test_a_completion_of_a_client_that_does_not_say_reports_no_tokens(self):
        client, sink = observing(ClientWithoutTools())
        client.complete("system", "user")
        assert sink.events[-1].usage is None
