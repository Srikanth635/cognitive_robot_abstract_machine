from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import openai
import pytest

from llm_reasoner import clients as client_package
from llm_reasoner.clients.base import LanguageModelClient
from llm_reasoner.clients.openai_client import (
    OpenAIClient,
    OpenAIResponseUnusable,
)
from llm_reasoner.clients.tools import (
    ResponseTruncated,
    ToolBox,
    ToolTurnsExhausted,
    ToolUsingClient,
)
from llm_reasoner.clients.usage import TokenUsage, UsageReporting

from .test_anthropic_client import EchoTool

# %% OpenAI service mimics


@dataclass
class IncompleteDetailsMimic:
    """Describe why a mimicked response stopped early."""

    reason: str
    """The service-provided reason for incompleteness."""


@dataclass
class ResponseErrorMimic:
    """Provide the service error detail consumed by :class:`OpenAIClient`."""

    message: str
    """The service-provided error message."""


@dataclass
class InputTokensDetailsMimic:
    """Break down the input tokens of a mimicked response."""

    cached_tokens: int = 0
    """The input tokens read from the cache."""

    cache_write_tokens: int = 0
    """The input tokens written to the cache."""


@dataclass
class ResponseUsageMimic:
    """Provide the usage of a mimicked response."""

    input_tokens: int
    """Every input token, those of the cache included."""

    output_tokens: int
    """The tokens generated, reasoning included."""

    input_tokens_details: InputTokensDetailsMimic = field(
        default_factory=InputTokensDetailsMimic
    )
    """How the input tokens break down."""


@dataclass
class FunctionCallMimic:
    """An output item asking for a function to be called."""

    name: str
    """The function called."""

    arguments: str
    """The input of the call, as JSON text."""

    call_id: str = "call_1"
    """The identifier the call's output answers."""

    type: str = "function_call"
    """The kind of output item."""


@dataclass
class ReasoningItemMimic:
    """An output item holding the model's reasoning, encrypted."""

    encrypted_content: str = "encrypted reasoning"
    """The reasoning in the form it can be sent back in."""

    type: str = "reasoning"
    """The kind of output item."""


@dataclass
class OpenAIResponseMimic:
    """Provide the response attributes consumed by :class:`OpenAIClient`."""

    output_text: str
    """The aggregated text returned by the response."""

    status: str = "completed"
    """The response lifecycle status."""

    incomplete_details: IncompleteDetailsMimic | None = None
    """The details supplied for an incomplete response."""

    error: ResponseErrorMimic | None = None
    """The service error attached to a failed response."""

    output: list[Any] = field(default_factory=list)
    """The output items of the response."""

    usage: ResponseUsageMimic | None = None
    """The tokens the response used, when the service reports them."""


@dataclass
class ResponsesEndpointRecorder:
    """Record requests made through the mimicked Responses endpoint."""

    response: OpenAIResponseMimic
    """The response returned for every request."""

    requests: list[dict[str, Any]] = field(default_factory=list)
    """The keyword arguments received by the endpoint."""

    def create(self, **request: Any) -> OpenAIResponseMimic:
        """Record and answer one Responses API request."""
        self.requests.append(request)
        return self.response


@dataclass
class OpenAIServiceMimic:
    """Expose the Responses endpoint expected by the client."""

    response: OpenAIResponseMimic
    """The response returned by the endpoint."""

    responses: ResponsesEndpointRecorder = field(init=False)
    """The endpoint recorder attached to the service."""

    def __post_init__(self) -> None:
        self.responses = ResponsesEndpointRecorder(self.response)


@dataclass
class OpenAIServiceFactoryRecorder:
    """Return a fixed service while recording API-key configuration."""

    service: OpenAIServiceMimic
    """The service returned to the production client."""

    api_keys: list[str | None] = field(default_factory=list)
    """The API keys passed while constructing services."""

    def __call__(self, *, api_key: str | None) -> OpenAIServiceMimic:
        """Record the key and return the configured service."""
        self.api_keys.append(api_key)
        return self.service


@dataclass
class OpenAIClientHarness:
    """Bundle one client with its mocked service observations."""

    client: OpenAIClient
    """The production client under test."""

    service: OpenAIServiceMimic
    """The mocked OpenAI service used by the client."""

    factory: OpenAIServiceFactoryRecorder
    """The factory recording SDK construction."""


def build_client_harness(
    monkeypatch: pytest.MonkeyPatch,
    response: OpenAIResponseMimic,
    **configuration: Any,
) -> OpenAIClientHarness:
    """Build a client backed by a deterministic OpenAI service mimic."""
    service = OpenAIServiceMimic(response)
    factory = OpenAIServiceFactoryRecorder(service)
    monkeypatch.setattr(openai, "OpenAI", factory)
    client = OpenAIClient(**configuration)
    return OpenAIClientHarness(client, service, factory)


# %% Completion contract


class TestOpenAICompletionContract:
    """OpenAI completions preserve the package's vendor-neutral contract."""

    def test_client_inherits_language_model_interface(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = build_client_harness(
            monkeypatch,
            OpenAIResponseMimic("answer"),
        )

        assert isinstance(harness.client, LanguageModelClient)
        assert harness.client.identity == "openai:gpt-5.6-sol"
        assert client_package.OpenAIClient is OpenAIClient
        assert client_package.OpenAIResponseUnusable is OpenAIResponseUnusable

    def test_complete_uses_responses_api_and_returns_output_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = build_client_harness(
            monkeypatch,
            OpenAIResponseMimic("answer"),
            api_key="secret",
            model="gpt-5.4",
            max_output_tokens=12_345,
        )

        result = harness.client.complete("system prompt", "user prompt")

        assert result == "answer"
        assert "secret" not in repr(harness.client)
        assert harness.factory.api_keys == ["secret"]
        assert harness.service.responses.requests == [
            {
                "model": "gpt-5.4",
                "instructions": "system prompt",
                "input": "user prompt",
                "max_output_tokens": 12_345,
                "store": False,
            }
        ]

    def test_incomplete_response_raises_service_specific_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = build_client_harness(
            monkeypatch,
            OpenAIResponseMimic(
                "partial",
                status="incomplete",
                incomplete_details=IncompleteDetailsMimic("max_output_tokens"),
            ),
        )

        with pytest.raises(OpenAIResponseUnusable) as raised:
            harness.client.complete("system", "user")

        assert raised.value.status == "incomplete"
        assert raised.value.reason == "max_output_tokens"

    def test_failed_response_reports_service_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = build_client_harness(
            monkeypatch,
            OpenAIResponseMimic(
                "",
                status="failed",
                error=ResponseErrorMimic("generation failed"),
            ),
        )

        with pytest.raises(OpenAIResponseUnusable) as raised:
            harness.client.complete("system", "user")

        assert raised.value.status == "failed"
        assert raised.value.reason == "generation failed"

    def test_completed_response_without_text_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = build_client_harness(
            monkeypatch,
            OpenAIResponseMimic(""),
        )

        with pytest.raises(OpenAIResponseUnusable) as raised:
            harness.client.complete("system", "user")

        assert raised.value.status == "completed"
        assert raised.value.reason == "the completed response contained no output text"


# %% Structured completion contract


class TestOpenAIStructuredCompletionContract:
    """Structured completions send the supplied JSON Schema to OpenAI."""

    def test_schema_uses_compatible_non_strict_mode_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = build_client_harness(
            monkeypatch,
            OpenAIResponseMimic('{"answer":"yes"}'),
            max_output_tokens=20_000,
        )
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": [],
            "additionalProperties": False,
        }

        result = harness.client.complete_structured("system", "user", schema)

        assert result == '{"answer":"yes"}'
        assert harness.client.supports_structured_output is True
        assert harness.service.responses.requests == [
            {
                "model": "gpt-5.6-sol",
                "instructions": "system",
                "input": "user",
                "max_output_tokens": 20_000,
                "store": False,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "llm_reasoner_response",
                        "schema": schema,
                        "strict": False,
                    }
                },
            }
        ]

    def test_strict_mode_can_be_enabled_for_compatible_schemas(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = build_client_harness(
            monkeypatch,
            OpenAIResponseMimic('{"answer":"yes"}'),
            strict_schema=True,
        )
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }

        harness.client.complete_structured("system", "user", schema)

        request = harness.service.responses.requests[0]
        assert request["text"]["format"]["strict"] is True

    def test_const_schema_nodes_receive_their_json_primitive_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = build_client_harness(
            monkeypatch,
            OpenAIResponseMimic('{"kind":"Thing"}'),
        )
        schema = {
            "type": "object",
            "properties": {"kind": {"const": "Thing"}},
            "required": [],
            "additionalProperties": False,
        }

        harness.client.complete_structured("system", "user", schema)

        request = harness.service.responses.requests[0]
        assert request["text"]["format"]["schema"]["properties"]["kind"] == {
            "const": "Thing",
            "type": "string",
        }

    def test_enum_schema_nodes_receive_their_json_primitive_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = build_client_harness(
            monkeypatch,
            OpenAIResponseMimic('{"kind":"Thing"}'),
        )
        schema = {
            "type": "object",
            "properties": {"kind": {"enum": ["Thing", "Other"]}},
            "required": [],
            "additionalProperties": False,
        }

        harness.client.complete_structured("system", "user", schema)

        request = harness.service.responses.requests[0]
        assert request["text"]["format"]["schema"]["properties"]["kind"] == {
            "enum": ["Thing", "Other"],
            "type": "string",
        }


# %% Completion usage

USAGE = ResponseUsageMimic(
    input_tokens=100,
    output_tokens=50,
    input_tokens_details=InputTokensDetailsMimic(
        cached_tokens=30, cache_write_tokens=10
    ),
)
"""
The usage a mimicked response reports: 60 input tokens read without the cache.
"""


class TestOpenAIUsage:
    """A completion says how many tokens of each kind it used."""

    def test_the_client_reports_the_usage_of_its_latest_completion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = build_client_harness(
            monkeypatch, OpenAIResponseMimic("answer", usage=USAGE)
        )

        harness.client.complete("system", "user")

        assert isinstance(harness.client, UsageReporting)
        assert harness.client.latest_usage == TokenUsage(
            input_tokens=60,
            output_tokens=50,
            cache_creation_input_tokens=10,
            cache_read_input_tokens=30,
        )

    def test_a_structured_completion_reports_its_usage_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = build_client_harness(
            monkeypatch, OpenAIResponseMimic('{"answer":"yes"}', usage=USAGE)
        )

        harness.client.complete_structured("system", "user", {"type": "object"})

        assert harness.client.latest_usage.output_tokens == USAGE.output_tokens

    def test_a_response_reporting_no_usage_used_no_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness = build_client_harness(monkeypatch, OpenAIResponseMimic("answer"))

        harness.client.complete("system", "user")

        assert harness.client.latest_usage == TokenUsage()


# %% Completion with tools


@dataclass
class SequencedResponsesEndpoint:
    """Answer successive requests with successive responses, recording each."""

    responses: list[OpenAIResponseMimic]
    """The responses, in the order they are given."""

    requests: list[dict[str, Any]] = field(default_factory=list)
    """The keyword arguments received by the endpoint."""

    def create(self, **request: Any) -> OpenAIResponseMimic:
        """Record one request and answer it with the next response."""
        self.requests.append(request)
        return self.responses[len(self.requests) - 1]


@dataclass
class SequencedServiceMimic:
    """Expose an endpoint answering with a sequence of responses."""

    responses: SequencedResponsesEndpoint
    """The endpoint the client talks to."""


def build_tool_loop_client(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[OpenAIResponseMimic],
    **configuration: Any,
) -> tuple[OpenAIClient, SequencedResponsesEndpoint]:
    """
    Build a client whose service answers successive requests with the given responses.
    """
    endpoint = SequencedResponsesEndpoint(responses)
    monkeypatch.setattr(
        openai, "OpenAI", lambda *, api_key: SequencedServiceMimic(endpoint)
    )
    return OpenAIClient(**configuration), endpoint


ECHO_CALL = FunctionCallMimic("echo", '{"text": "the jug"}')
"""
A call of the echo tool.
"""


class TestOpenAIToolLoop:
    """Tool calls are run and answered until the model writes its answer."""

    def test_the_client_lets_the_model_call_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, _ = build_tool_loop_client(monkeypatch, [])

        assert isinstance(client, ToolUsingClient)

    def test_the_tools_are_offered_as_functions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, endpoint = build_tool_loop_client(
            monkeypatch, [OpenAIResponseMimic("done")]
        )

        client.complete_with_tools("system", "user", [EchoTool()])

        definition = EchoTool().definition
        assert endpoint.requests[0]["tools"] == [
            {
                "type": "function",
                "name": definition.name,
                "description": definition.description.render(),
                "parameters": definition.input_schema,
                "strict": False,
            }
        ]

    def test_the_first_turn_asks_with_the_prompts_and_without_storing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, endpoint = build_tool_loop_client(
            monkeypatch,
            [OpenAIResponseMimic("done")],
            model="gpt-5.4",
            max_output_tokens=12_345,
        )

        client.complete_with_tools("system prompt", "user prompt", [EchoTool()])

        request = endpoint.requests[0]
        assert {key: value for key, value in request.items() if key != "tools"} == {
            "model": "gpt-5.4",
            "instructions": "system prompt",
            "input": [{"role": "user", "content": "user prompt"}],
            "max_output_tokens": 12_345,
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }

    def test_a_tool_call_is_run_and_its_result_sent_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, endpoint = build_tool_loop_client(
            monkeypatch,
            [
                OpenAIResponseMimic("", output=[ECHO_CALL]),
                OpenAIResponseMimic("done"),
            ],
        )

        conversation = client.complete_with_tools("system", "user", [EchoTool()])

        assert conversation.answer == "done"
        assert [call.arguments for call in conversation.calls] == [{"text": "the jug"}]
        assert endpoint.requests[1]["input"] == [
            *endpoint.requests[0]["input"],
            ECHO_CALL,
            {
                "type": "function_call_output",
                "call_id": ECHO_CALL.call_id,
                "output": "the jug",
            },
        ]

    def test_the_reasoning_of_a_turn_is_sent_back_with_its_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reasoning = ReasoningItemMimic()
        client, endpoint = build_tool_loop_client(
            monkeypatch,
            [
                OpenAIResponseMimic("", output=[reasoning, ECHO_CALL]),
                OpenAIResponseMimic("done"),
            ],
        )

        client.complete_with_tools("system", "user", [EchoTool()])

        assert endpoint.requests[1]["input"][1:3] == [reasoning, ECHO_CALL]

    def test_a_call_the_toolbox_refuses_is_answered_with_its_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        unknown = FunctionCallMimic("no_such_tool", "{}")
        client, endpoint = build_tool_loop_client(
            monkeypatch,
            [
                OpenAIResponseMimic("", output=[unknown]),
                OpenAIResponseMimic("done"),
            ],
        )

        client.complete_with_tools("system", "user", [EchoTool()])

        assert endpoint.requests[1]["input"][-1]["output"] == (
            ToolBox(tools=[EchoTool()]).call(unknown.name, {}).content
        )

    def test_the_loop_stops_at_its_turn_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        budget = 2
        calling = OpenAIResponseMimic("", output=[ECHO_CALL])
        client, endpoint = build_tool_loop_client(
            monkeypatch, [calling] * (budget + 1), max_tool_turns=budget
        )

        with pytest.raises(ToolTurnsExhausted):
            client.complete_with_tools("system", "user", [EchoTool()])
        assert len(endpoint.requests) == budget

    def test_an_answer_cut_off_at_the_clients_limit_is_reported_with_that_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        limit = 1_000
        client, _ = build_tool_loop_client(
            monkeypatch,
            [
                OpenAIResponseMimic(
                    "[{",
                    status="incomplete",
                    incomplete_details=IncompleteDetailsMimic("max_output_tokens"),
                    usage=ResponseUsageMimic(input_tokens=10, output_tokens=limit),
                )
            ],
            max_output_tokens=limit,
        )

        with pytest.raises(ResponseTruncated) as raised:
            client.complete_with_tools("system", "user", [EchoTool()])
        assert raised.value.token_limit == limit
        assert raised.value.partial_answer == "[{"

    def test_a_turn_that_fails_otherwise_is_reported_as_unusable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, _ = build_tool_loop_client(
            monkeypatch,
            [
                OpenAIResponseMimic(
                    "", status="failed", error=ResponseErrorMimic("generation failed")
                )
            ],
        )

        with pytest.raises(OpenAIResponseUnusable):
            client.complete_with_tools("system", "user", [EchoTool()])

    def test_the_conversation_adds_up_the_tokens_of_every_turn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, _ = build_tool_loop_client(
            monkeypatch,
            [
                OpenAIResponseMimic("", output=[ECHO_CALL], usage=USAGE),
                OpenAIResponseMimic("done", usage=USAGE),
            ],
        )

        conversation = client.complete_with_tools("system", "user", [EchoTool()])

        single = TokenUsage(
            input_tokens=60,
            output_tokens=50,
            cache_creation_input_tokens=10,
            cache_read_input_tokens=30,
        )
        assert conversation.usage == single + single
