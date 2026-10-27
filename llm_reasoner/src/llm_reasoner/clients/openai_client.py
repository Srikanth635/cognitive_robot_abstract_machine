from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import openai
from openai.types.responses.response import Response

from llm_reasoner.clients.base import LanguageModelClient
from llm_reasoner.exceptions import LLMReasonerError

# %% OpenAI client errors


@dataclass
class OpenAIResponseUnusable(LLMReasonerError):
    """
    Report a response that cannot satisfy the completion contract.
    """

    status: str
    """The lifecycle status returned by OpenAI."""

    reason: str
    """
    The service-provided reason or a local description of the failure.
    """

    def __str__(self) -> str:
        """
        Return a concise diagnostic for the unusable response.
        """
        return f"OpenAI response is {self.status}: {self.reason}."


# %% OpenAI client


@dataclass
class OpenAIClient(LanguageModelClient):
    """
    Complete prompts through OpenAI's Responses API.
    """

    model: str = "gpt-5.6-sol"
    """
    The OpenAI model identifier.
    """

    max_output_tokens: int = 64_000
    """
    The maximum combined reasoning and visible output token count.
    """

    api_key: str | None = field(default=None, repr=False)
    """
    The API key, falling back to the ``OPENAI_API_KEY`` environment variable.
    """

    strict_schema: bool = False
    """
    Whether structured requests require strict JSON Schema adherence.

    The package's generated schemas contain optional properties, which OpenAI's strict
    mode does not accept. Enable this only for schemas whose object properties are all
    required.
    """

    _client: openai.OpenAI = field(init=False, repr=False)
    """
    The underlying OpenAI SDK client.
    """

    def __post_init__(self) -> None:
        """
        Create the SDK client from the supplied or environment API key.
        """
        self._client = openai.OpenAI(api_key=self.api_key)

    def complete(self, system: str, user: str) -> str:
        """
        Return one unstructured model completion.
        """
        response = self._create_response(system, user)
        return self._extract_text(response)

    def complete_structured(
        self, system: str, user: str, json_schema: Mapping[str, Any]
    ) -> str:
        """
        Return JSON requested through the Responses API schema format.
        """
        text_configuration = {
            "format": {
                "type": "json_schema",
                "name": "llm_reasoner_response",
                "schema": self._normalize_schema(json_schema),
                "strict": self.strict_schema,
            }
        }
        response = self._create_response(system, user, text_configuration)
        return self._extract_text(response)

    @property
    def supports_structured_output(self) -> bool:
        """
        Return whether requests use the provider's JSON Schema format.
        """
        return True

    @property
    def identity(self) -> str:
        """
        Return the provider and model identifier used for provenance.
        """
        return f"openai:{self.model}"

    def _create_response(
        self,
        system: str,
        user: str,
        text_configuration: Mapping[str, Any] | None = None,
    ) -> Response:
        """
        Issue one stateless Responses API request.
        """
        request: dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": user,
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }
        if text_configuration is not None:
            request["text"] = dict(text_configuration)
        return self._client.responses.create(**request)

    @staticmethod
    def _extract_text(response: Response) -> str:
        """
        Return aggregated output text or reject an unusable response.
        """
        status = response.status or "unknown"
        if status != "completed":
            raise OpenAIResponseUnusable(
                status=status,
                reason=OpenAIClient._failure_reason(response),
            )
        if not response.output_text:
            raise OpenAIResponseUnusable(
                status=status,
                reason="the completed response contained no output text",
            )
        return response.output_text

    @staticmethod
    def _failure_reason(response: Response) -> str:
        """
        Describe an unsuccessful response using the available service detail.
        """
        if response.incomplete_details is not None:
            return response.incomplete_details.reason or "unspecified reason"
        if response.error is not None:
            return response.error.message
        return "unspecified reason"

    @staticmethod
    def _normalize_schema(value: Any) -> Any:
        """
        Adapt JSON Schema discriminator nodes to OpenAI's schema dialect.

        OpenAI requires a primitive ``type`` alongside a primitive ``const``; the
        generated knowledge schema uses the shorter, valid JSON Schema form. The input
        is copied so the shared schema remains vendor-neutral.
        """
        if isinstance(value, Mapping):
            normalized = {
                key: OpenAIClient._normalize_schema(child)
                for key, child in value.items()
            }
            if "const" in normalized and "type" not in normalized:
                primitive_type = OpenAIClient._primitive_type(normalized["const"])
                if primitive_type is not None:
                    normalized["type"] = primitive_type
            if "enum" in normalized and "type" not in normalized:
                primitive_types = {
                    OpenAIClient._primitive_type(item) for item in normalized["enum"]
                }
                if len(primitive_types) == 1:
                    normalized["type"] = primitive_types.pop()
            return normalized
        if isinstance(value, list):
            return [OpenAIClient._normalize_schema(item) for item in value]
        return value

    @staticmethod
    def _primitive_type(value: Any) -> str | None:
        """
        Return the JSON primitive type for a scalar schema value.
        """
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, str):
            return "string"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "number"
        if value is None:
            return "null"
        return None
