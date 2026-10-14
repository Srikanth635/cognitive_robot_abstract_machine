from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from pathlib import Path

import anthropic
import pytest

from llm_reasoner.clients.anthropic_client import AnthropicClient
from llm_reasoner.clients.base import LanguageModelClient
from llm_reasoner.clients.tools import (
    Tool,
    ToolDefinition,
    ToolOutcome,
    ToolTurnsExhausted,
)
from llm_reasoner.clients.usage import TokenUsage

TOOL_CASES = Path(__file__).parent / "tool_cases"
"""
The tool definitions the tool-loop tests offer.
"""

# %% Anthropic service mimics


@dataclass
class TextBlockMimic:
    """Provide a text content block as returned by the Messages endpoint."""

    text: str
    """The block's text payload."""

    type: str = "text"
    """The content block's discriminator, fixed to the client's expectation."""


@dataclass
class ToolUseBlockMimic:
    """Provide a tool-use content block as returned by a forced tool call."""

    name: str
    """The name of the tool the model was forced to call."""

    input: dict[str, Any]
    """The arguments the model supplied to the forced tool."""

    id: str = "toolu_mimic"
    """The identifier a tool result answering this call must cite."""

    type: str = "tool_use"
    """The content block's discriminator, fixed to the client's expectation."""


@dataclass
class UsageMimic:
    """Provide the token-usage figures of a response."""

    output_tokens: int = 0
    """The number of tokens generated for this response."""

    input_tokens: int = 0
    """The number of uncached input tokens read for this response."""

    cache_creation_input_tokens: int | None = None
    """The number of input tokens written to the cache, absent when none were."""

    cache_read_input_tokens: int | None = None
    """The number of input tokens read from the cache, absent when none were."""


@dataclass
class AnthropicResponseMimic:
    """Provide the response attributes consumed by :class:`AnthropicClient`."""

    content: list[TextBlockMimic | ToolUseBlockMimic]
    """The response's content blocks."""

    stop_reason: str = "end_turn"
    """Why generation stopped."""

    usage: UsageMimic = field(default_factory=UsageMimic)
    """Token-usage figures for this response."""


@dataclass
class StreamMimic:
    """Provide the context-manager stream interface used by structured completions."""

    response: AnthropicResponseMimic
    """The response handed back once the stream is drained."""

    def __enter__(self) -> "StreamMimic":
        return self

    def __exit__(self, *exception_info: object) -> None:
        return None

    def get_final_message(self) -> AnthropicResponseMimic:
        """Return the accumulated message, matching the real SDK's stream API."""
        return self.response


@dataclass
class MessagesEndpointRecorder:
    """Record requests made through the mimicked Messages endpoint."""

    response: AnthropicResponseMimic
    """The response returned for every request."""

    requests: list[dict[str, Any]] = field(default_factory=list)
    """The keyword arguments received by the endpoint, one entry per call."""

    def create(self, **request: Any) -> AnthropicResponseMimic:
        """Record and answer one unstructured completion request."""
        self.requests.append(request)
        return self.response

    def stream(self, **request: Any) -> StreamMimic:
        """Record and answer one structured completion request."""
        self.requests.append(request)
        return StreamMimic(self.response)


@dataclass
class AnthropicServiceMimic:
    """Expose the Messages endpoint expected by the client."""

    response: AnthropicResponseMimic
    """The response returned by the endpoint."""

    messages: MessagesEndpointRecorder = field(init=False)
    """The endpoint recorder attached to the service."""

    def __post_init__(self) -> None:
        self.messages = MessagesEndpointRecorder(self.response)


@dataclass
class AnthropicServiceFactoryRecorder:
    """Return a fixed service while recording API-key configuration."""

    service: AnthropicServiceMimic
    """The service returned to the production client."""

    api_keys: list[str | None] = field(default_factory=list)
    """The API keys passed while constructing services."""

    def __call__(self, *, api_key: str | None) -> AnthropicServiceMimic:
        """Record the key and return the configured service."""
        self.api_keys.append(api_key)
        return self.service


@dataclass
class AnthropicClientHarness:
    """Bundle one client with its mocked service observations."""

    client: AnthropicClient
    """The production client under test."""

    service: AnthropicServiceMimic
    """The mocked Anthropic service used by the client."""

    factory: AnthropicServiceFactoryRecorder
    """The factory recording SDK construction."""


def build_client_harness(
    monkeypatch: pytest.MonkeyPatch,
    response: AnthropicResponseMimic,
    **configuration: Any,
) -> AnthropicClientHarness:
    """
    Build a client backed by a deterministic Anthropic service mimic.
    """
    service = AnthropicServiceMimic(response)
    factory = AnthropicServiceFactoryRecorder(service)
    monkeypatch.setattr(anthropic, "Anthropic", factory)
    client = AnthropicClient(**configuration)
    return AnthropicClientHarness(client, service, factory)


# %% Completion contract


class TestAnthropicCompletionContract:
    """Anthropic completions preserve the package's vendor-neutral contract."""

    def test_client_inherits_language_model_interface(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = build_client_harness(
            monkeypatch,
            AnthropicResponseMimic([TextBlockMimic("answer")]),
        )

        assert isinstance(harness.client, LanguageModelClient)
        assert harness.client.identity == "anthropic:claude-haiku-4-5"

    def test_complete_sends_no_removed_sampling_parameters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        The installed SDK generation removed ``temperature`` from
        ``messages.create`` outright; passing it raises ``TypeError`` against
        the real client. This mimic does not reject unknown keywords, so the
        request contract is asserted explicitly instead of relying on the
        mimic to fail the way the real SDK does.
        """
        harness = build_client_harness(
            monkeypatch,
            AnthropicResponseMimic([TextBlockMimic("answer")]),
            api_key="secret",
            model="claude-sonnet-5",
            max_tokens=4_096,
        )

        result = harness.client.complete("system prompt", "user prompt")

        assert result == "answer"
        assert harness.factory.api_keys == ["secret"]
        assert harness.service.messages.requests == [
            {
                "model": "claude-sonnet-5",
                "max_tokens": 4_096,
                "system": "system prompt",
                "messages": [{"role": "user", "content": "user prompt"}],
            }
        ]

    def test_complete_concatenates_only_text_blocks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = build_client_harness(
            monkeypatch,
            AnthropicResponseMimic(
                [TextBlockMimic("first "), TextBlockMimic("second")]
            ),
        )

        assert harness.client.complete("system", "user") == "first second"


# %% Usage of a completion


USED = UsageMimic(
    input_tokens=11,
    output_tokens=7,
    cache_creation_input_tokens=3,
    cache_read_input_tokens=5,
)
"""
The tokens a response says it consumed.
"""

USED_AS_REPORTED = TokenUsage(
    input_tokens=USED.input_tokens,
    output_tokens=USED.output_tokens,
    cache_creation_input_tokens=USED.cache_creation_input_tokens,
    cache_read_input_tokens=USED.cache_read_input_tokens,
)
"""
The same tokens as the client reports them.
"""


class TestAnthropicUsageReporting:
    """A completion says what it used, so what it cost can be told."""

    def test_no_tokens_are_reported_before_a_completion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = build_client_harness(
            monkeypatch, AnthropicResponseMimic([TextBlockMimic("answer")])
        )

        assert harness.client.latest_usage == TokenUsage()

    def test_a_plain_completion_reports_the_tokens_it_used(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = build_client_harness(
            monkeypatch, AnthropicResponseMimic([TextBlockMimic("answer")], usage=USED)
        )

        harness.client.complete("system", "user")

        assert harness.client.latest_usage == USED_AS_REPORTED

    def test_a_structured_completion_reports_the_tokens_it_used(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = build_client_harness(
            monkeypatch,
            AnthropicResponseMimic(
                [ToolUseBlockMimic(name="emit_hypothesis", input={})], usage=USED
            ),
        )

        harness.client.complete_structured("system", "user", {"type": "object"})

        assert harness.client.latest_usage == USED_AS_REPORTED

    def test_a_completion_in_another_thread_does_not_change_what_this_thread_reads(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = build_client_harness(
            monkeypatch, AnthropicResponseMimic([TextBlockMimic("answer")], usage=USED)
        )
        harness.client.complete("system", "user")
        harness.service.messages.response = AnthropicResponseMimic(
            [TextBlockMimic("other")], usage=UsageMimic(input_tokens=1, output_tokens=1)
        )
        worker = threading.Thread(
            target=harness.client.complete, args=("system", "user")
        )
        worker.start()
        worker.join()

        assert harness.client.latest_usage == USED_AS_REPORTED


# %% Structured completion contract


class TestAnthropicStructuredCompletionContract:
    """Structured completions ask for one tool call and send no removed parameters."""

    def test_structured_completion_sends_no_removed_sampling_parameters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = build_client_harness(
            monkeypatch,
            AnthropicResponseMimic(
                [ToolUseBlockMimic("emit_hypothesis", {"answer": "yes"})]
            ),
            model="claude-sonnet-5",
        )
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": [],
        }

        result = harness.client.complete_structured("system", "user", schema)

        assert result == '{"answer": "yes"}'
        assert harness.client.supports_structured_output is True
        request = harness.service.messages.requests[0]
        assert "temperature" not in request
        assert request["model"] == "claude-sonnet-5"
        assert request["system"][0]["text"].startswith("system")
        assert request["tools"][0]["input_schema"] == schema

    @pytest.mark.parametrize(
        "model, ceiling",
        [
            ("claude-sonnet-4-5", 64_000),
            ("claude-opus-4-5", 64_000),
            ("claude-sonnet-5-5", 128_000),
            ("claude-haiku-4-5", 64_000),
        ],
    )
    def test_structured_completion_requests_the_models_output_ceiling(
        self, monkeypatch: pytest.MonkeyPatch, model: str, ceiling: int
    ) -> None:
        harness = build_client_harness(
            monkeypatch,
            AnthropicResponseMimic(
                [ToolUseBlockMimic("emit_hypothesis", {"answer": "yes"})]
            ),
            model=model,
        )

        harness.client.complete_structured("system", "user", {})

        assert harness.service.messages.requests[0]["max_tokens"] == ceiling

    def test_structured_completion_asks_for_its_one_tool_call_without_forcing_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = build_client_harness(
            monkeypatch,
            AnthropicResponseMimic(
                [ToolUseBlockMimic("emit_hypothesis", {"answer": "yes"})]
            ),
            model="claude-sonnet-5-5",
        )

        harness.client.complete_structured("system", "user", {})

        request = harness.service.messages.requests[0]
        assert request["tool_choice"] == {
            "type": "auto",
            "disable_parallel_tool_use": True,
        }
        assert "emit_hypothesis" in request["system"][0]["text"]

    def test_structured_system_prompt_is_cached(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = build_client_harness(
            monkeypatch,
            AnthropicResponseMimic(
                [ToolUseBlockMimic("emit_hypothesis", {"answer": "yes"})]
            ),
            model="claude-sonnet-5-5",
        )

        harness.client.complete_structured("system", "user", {})

        (block,) = harness.service.messages.requests[0]["system"]
        assert block["type"] == "text"
        assert block["text"].startswith("system")
        assert block["cache_control"] == {"type": "ephemeral"}

    def test_structured_answer_written_as_text_is_returned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = build_client_harness(
            monkeypatch,
            AnthropicResponseMimic([TextBlockMimic('{"fragments": []}')]),
            model="claude-sonnet-5-5",
        )

        assert (
            harness.client.complete_structured("system", "user", {})
            == '{"fragments": []}'
        )

    def test_truncated_structured_response_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = build_client_harness(
            monkeypatch,
            AnthropicResponseMimic(
                [ToolUseBlockMimic("emit_hypothesis", {})],
                stop_reason="max_tokens",
                usage=UsageMimic(output_tokens=64_000),
            ),
        )

        with pytest.raises(RuntimeError, match="truncated"):
            harness.client.complete_structured("system", "user", {})


# %% Tool loop contract


@dataclass
class SequencedMessagesEndpoint:
    """Answer successive requests with successive responses, recording each request."""

    responses: list[AnthropicResponseMimic]
    """The responses, returned in order."""

    requests: list[dict[str, Any]] = field(default_factory=list)
    """The keyword arguments received by the endpoint, one entry per call."""

    def create(self, **request: Any) -> AnthropicResponseMimic:
        """Record one request and answer it with the next response."""
        self.requests.append(request)
        return self.responses[len(self.requests) - 1]


@dataclass
class SequencedServiceMimic:
    """Expose a Messages endpoint answering with a sequence of responses."""

    messages: SequencedMessagesEndpoint
    """The endpoint the client talks to."""


@dataclass
class EchoTool(Tool):
    """A tool repeating its text, or failing when told to."""

    fails: bool = False
    """Whether every call is answered as an error."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition.from_file(TOOL_CASES / "echo.json")

    def run(self, arguments: dict[str, Any]) -> ToolOutcome:
        return ToolOutcome(content=arguments["text"], is_error=self.fails)


def build_tool_loop_client(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[AnthropicResponseMimic],
    **configuration: Any,
) -> tuple[AnthropicClient, SequencedMessagesEndpoint]:
    """
    Build a client whose service answers successive requests with the given responses.
    """
    endpoint = SequencedMessagesEndpoint(responses)
    monkeypatch.setattr(
        anthropic, "Anthropic", lambda *, api_key: SequencedServiceMimic(endpoint)
    )
    return AnthropicClient(**configuration), endpoint


def tool_results_of(request: dict[str, Any]) -> list[dict[str, Any]]:
    """The tool results the last message of a request carries."""
    return request["messages"][-1]["content"]


class TestAnthropicToolLoop:
    """Tool calls are run and answered until the model writes its answer."""

    def test_a_tool_call_is_run_and_its_result_sent_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        call = ToolUseBlockMimic("echo", {"text": "the jug"}, id="toolu_1")
        client, endpoint = build_tool_loop_client(
            monkeypatch,
            [
                AnthropicResponseMimic([call], stop_reason="tool_use"),
                AnthropicResponseMimic([TextBlockMimic("done")]),
            ],
        )

        conversation = client.complete_with_tools("system", "user", [EchoTool()])

        assert conversation.answer == "done"
        assert [recorded.arguments for recorded in conversation.calls] == [call.input]
        (result,) = tool_results_of(endpoint.requests[1])
        assert result["tool_use_id"] == call.id
        assert result["content"] == call.input["text"]
        assert endpoint.requests[0]["tools"] == [EchoTool().definition.as_request()]

    def test_a_failing_tool_is_reported_to_the_model_as_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, endpoint = build_tool_loop_client(
            monkeypatch,
            [
                AnthropicResponseMimic(
                    [ToolUseBlockMimic("echo", {"text": "x"})], stop_reason="tool_use"
                ),
                AnthropicResponseMimic([TextBlockMimic("done")]),
            ],
        )

        client.complete_with_tools("system", "user", [EchoTool(fails=True)])

        (result,) = tool_results_of(endpoint.requests[1])
        assert result["is_error"] is True

    def test_the_loop_stops_at_its_turn_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        budget = 2
        calling = AnthropicResponseMimic(
            [ToolUseBlockMimic("echo", {"text": "again"})], stop_reason="tool_use"
        )
        client, endpoint = build_tool_loop_client(
            monkeypatch, [calling] * (budget + 1), max_tool_turns=budget
        )

        with pytest.raises(ToolTurnsExhausted):
            client.complete_with_tools("system", "user", [EchoTool()])
        assert len(endpoint.requests) == budget

    def test_the_tokens_of_every_turn_are_added_up(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = UsageMimic(
            output_tokens=30, input_tokens=500, cache_creation_input_tokens=200
        )
        second = UsageMimic(
            output_tokens=40, input_tokens=600, cache_read_input_tokens=200
        )
        client, _ = build_tool_loop_client(
            monkeypatch,
            [
                AnthropicResponseMimic(
                    [ToolUseBlockMimic("echo", {"text": "x"})],
                    stop_reason="tool_use",
                    usage=first,
                ),
                AnthropicResponseMimic([TextBlockMimic("done")], usage=second),
            ],
        )

        conversation = client.complete_with_tools("system", "user", [EchoTool()])

        assert conversation.usage == TokenUsage(
            input_tokens=first.input_tokens + second.input_tokens,
            output_tokens=first.output_tokens + second.output_tokens,
            cache_creation_input_tokens=first.cache_creation_input_tokens,
            cache_read_input_tokens=second.cache_read_input_tokens,
        )

    def test_the_conversation_is_cached_as_it_grows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, endpoint = build_tool_loop_client(
            monkeypatch,
            [
                AnthropicResponseMimic(
                    [ToolUseBlockMimic("echo", {"text": "x"})], stop_reason="tool_use"
                ),
                AnthropicResponseMimic([TextBlockMimic("done")]),
            ],
        )

        client.complete_with_tools("system", "user", [EchoTool()])

        for request in endpoint.requests:
            (system_block,) = request["system"]
            assert system_block["text"] == "system"
            assert system_block["cache_control"] == {"type": "ephemeral"}
            assert request["cache_control"] == {"type": "ephemeral"}

    def test_a_turn_stopping_for_tools_without_calling_one_is_the_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        text = TextBlockMimic("done")
        client, endpoint = build_tool_loop_client(
            monkeypatch, [AnthropicResponseMimic([text], stop_reason="tool_use")]
        )

        conversation = client.complete_with_tools("system", "user", [EchoTool()])

        assert conversation.answer == text.text
        assert conversation.calls == []
        assert len(endpoint.requests) == 1
