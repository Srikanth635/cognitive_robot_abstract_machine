from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

from llm_reasoner.clients.base import LanguageModelClient

if TYPE_CHECKING:
    from llm_reasoner.clients.anthropic_client import AnthropicClient

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

    type: str = "tool_use"
    """The content block's discriminator, fixed to the client's expectation."""


@dataclass
class UsageMimic:
    """Provide the token-usage figures consumed by truncation checks."""

    output_tokens: int = 0
    """The number of tokens generated for this response."""


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

    Imports the vendor SDK and :class:`AnthropicClient` here, at call time,
    rather than at module scope: importing ``anthropic`` before pytest
    collects ``llm_reasoner.knowledge.microworlds.flanagan`` corrupts that
    module's dataclass field annotations (a vendor-SDK import side effect,
    order-dependent on collection order, not something this package causes or
    can fix). Deferring the import until a test actually runs avoids the
    ordering entirely, since collection finishes importing every test module
    before any test body executes.
    """
    import anthropic

    from llm_reasoner.clients.anthropic_client import AnthropicClient

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


# %% Structured completion contract


class TestAnthropicStructuredCompletionContract:
    """Structured completions force a tool call and send no removed parameters."""

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
        assert request["system"] == "system"
        assert request["tool_choice"] == {"type": "tool", "name": "emit_hypothesis"}
        assert request["tools"][0]["input_schema"] == schema

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
