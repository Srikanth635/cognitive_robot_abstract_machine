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
