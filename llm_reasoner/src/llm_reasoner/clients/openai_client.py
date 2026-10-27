from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import openai
from openai.types.responses.response import Response

from llm_reasoner.clients.base import LanguageModelClient
from llm_reasoner.clients.tools import (
    ResponseTruncated,
    Tool,
    ToolBox,
    ToolCall,
    ToolConversation,
    ToolTurnsExhausted,
    ToolUsingClient,
)
from llm_reasoner.clients.usage import LatestUsage, TokenUsage, UsageReporting
from llm_reasoner.exceptions import LLMReasonerError

# %% Responses API vocabulary


class RequestKey(StrEnum):
    """
    The parameters of a Responses API request the client sets.
    """

    MODEL = "model"
    INSTRUCTIONS = "instructions"
    INPUT = "input"
    MAX_OUTPUT_TOKENS = "max_output_tokens"
    STORE = "store"
    TEXT = "text"
    TOOLS = "tools"
    INCLUDE = "include"


class ItemKey(StrEnum):
    """
    The keys of the items, tools and formats a request is made of.
    """

    TYPE = "type"
    ROLE = "role"
    CONTENT = "content"
    CALL_ID = "call_id"
    OUTPUT = "output"
    NAME = "name"
    DESCRIPTION = "description"
    PARAMETERS = "parameters"
    STRICT = "strict"
    FORMAT = "format"
    SCHEMA = "schema"


class ItemType(StrEnum):
    """
    The kinds of item, tool and format the client sends or reads.
    """

    FUNCTION = "function"
    """
    A tool offered as a function the model may call.
    """

    FUNCTION_CALL = "function_call"
    """
    An output item asking for a function to be called.
    """

    FUNCTION_CALL_OUTPUT = "function_call_output"
    """
    What a call returned, sent back to the model.
    """

    JSON_SCHEMA = "json_schema"
    """
    An answer format checked against a JSON schema.
    """


class InputRole(StrEnum):
    """
    Who an input message is from.
    """

    USER = "user"


class ResponseStatus(StrEnum):
    """
    The lifecycle states of a response the client tells apart.
    """

    COMPLETED = "completed"
    INCOMPLETE = "incomplete"


class IncompleteReason(StrEnum):
    """
    Why a response ended before it was complete.
    """

    MAX_OUTPUT_TOKENS = "max_output_tokens"
    """
    It reached the request's limit of output tokens.
    """


class Includable(StrEnum):
    """
    The extra output a request asks the service to include.
    """

    ENCRYPTED_REASONING = "reasoning.encrypted_content"
    """
    The model's reasoning in a form that can be sent back in a later turn, which an
    unstored conversation needs to keep it.
    """


class FormatName(StrEnum):
    """
    The names an answer format is sent under.
    """

    STRUCTURED_ANSWER = "llm_reasoner_response"
    """
    The format of a structured completion's answer.
    """


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
class OpenAIClient(LanguageModelClient, ToolUsingClient, UsageReporting):
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
    Whether structured requests and tool inputs require strict JSON Schema adherence.

    The package's generated schemas contain optional properties, which OpenAI's strict
    mode does not accept. Enable this only for schemas whose object properties are all
    required.
    """

    max_tool_turns: int = 10
    """
    How many turns of tool calls a completion with tools may take before it is given
    up, so a model that keeps asking cannot run on without end.
    """

    _client: openai.OpenAI = field(init=False, repr=False)
    """
    The underlying OpenAI SDK client.
    """

    _latest: LatestUsage = field(
        default_factory=LatestUsage, init=False, repr=False, compare=False
    )
    """
    The tokens of the latest plain or structured completion, for each thread.
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
        self._latest.usage = self._usage_of(response)
        return self._extract_text(response)

    def complete_structured(
        self, system: str, user: str, json_schema: Mapping[str, Any]
    ) -> str:
        """
        Return JSON requested through the Responses API schema format.
        """
        text_configuration = {
            ItemKey.FORMAT: {
                ItemKey.TYPE: ItemType.JSON_SCHEMA,
                ItemKey.NAME: FormatName.STRUCTURED_ANSWER,
                ItemKey.SCHEMA: self._normalize_schema(json_schema),
                ItemKey.STRICT: self.strict_schema,
            }
        }
        response = self._create_response(system, user, text_configuration)
        self._latest.usage = self._usage_of(response)
        return self._extract_text(response)

    def complete_with_tools(
        self, system: str, user: str, tools: Sequence[Tool]
    ) -> ToolConversation:
        """
        Let the model call tools, running each call and sending its result back, until
        it writes its answer.

        Every turn resends the conversation so far, the output items of each earlier
        turn included, since nothing is stored with the service; its reasoning is asked
        for in encrypted form so that it is kept across the turns.

        :raises ResponseTruncated: When a turn reaches the client's output limit.
        :raises OpenAIResponseUnusable: When a turn fails or ends incomplete otherwise.
        """
        toolbox = ToolBox(tools=tools)
        functions = [self._function_of(tool) for tool in tools]
        conversation: list[Any] = [
            {ItemKey.ROLE: InputRole.USER, ItemKey.CONTENT: user}
        ]
        calls: list[ToolCall] = []
        usage = TokenUsage()
        for _ in range(self.max_tool_turns):
            response = self._client.responses.create(
                **{
                    RequestKey.MODEL: self.model,
                    RequestKey.INSTRUCTIONS: system,
                    RequestKey.INPUT: list(conversation),
                    RequestKey.MAX_OUTPUT_TOKENS: self.max_output_tokens,
                    RequestKey.STORE: False,
                    RequestKey.INCLUDE: [Includable.ENCRYPTED_REASONING],
                    RequestKey.TOOLS: functions,
                }
            )
            usage = usage + self._usage_of(response)
            self._check_truncation(response)
            requested = [
                item for item in response.output if item.type == ItemType.FUNCTION_CALL
            ]
            if not requested:
                return ToolConversation(
                    answer=self._extract_text(response), calls=calls, usage=usage
                )
            self._check_completed(response)
            conversation.extend(response.output)
            for item in requested:
                arguments = json.loads(item.arguments)
                outcome = toolbox.call(item.name, arguments)
                calls.append(
                    ToolCall(name=item.name, arguments=arguments, outcome=outcome)
                )
                conversation.append(
                    {
                        ItemKey.TYPE: ItemType.FUNCTION_CALL_OUTPUT,
                        ItemKey.CALL_ID: item.call_id,
                        ItemKey.OUTPUT: outcome.content,
                    }
                )
        raise ToolTurnsExhausted(turns=self.max_tool_turns)

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

    @property
    def latest_usage(self) -> TokenUsage:
        return self._latest.usage

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
            RequestKey.MODEL: self.model,
            RequestKey.INSTRUCTIONS: system,
            RequestKey.INPUT: user,
            RequestKey.MAX_OUTPUT_TOKENS: self.max_output_tokens,
            RequestKey.STORE: False,
        }
        if text_configuration is not None:
            request[RequestKey.TEXT] = dict(text_configuration)
        return self._client.responses.create(**request)

    def _function_of(self, tool: Tool) -> dict[str, Any]:
        """
        :return: The tool as the function a request offers the model.
        """
        definition = tool.definition
        return {
            ItemKey.TYPE: ItemType.FUNCTION,
            ItemKey.NAME: definition.name,
            ItemKey.DESCRIPTION: definition.description.render(),
            ItemKey.PARAMETERS: self._normalize_schema(definition.input_schema),
            ItemKey.STRICT: self.strict_schema,
        }

    def _check_truncation(self, response: Response) -> None:
        """
        Fail specifically when the response was cut off at the client's output limit.

        :raises ResponseTruncated: When the model was cut off.
        """
        if (
            response.status == ResponseStatus.INCOMPLETE
            and response.incomplete_details is not None
            and response.incomplete_details.reason == IncompleteReason.MAX_OUTPUT_TOKENS
        ):
            raise ResponseTruncated(
                output_tokens=self._usage_of(response).output_tokens,
                token_limit=self.max_output_tokens,
                partial_answer=response.output_text or "",
            )

    @staticmethod
    def _check_completed(response: Response) -> None:
        """
        Reject a response that did not complete.

        :raises OpenAIResponseUnusable: When it failed or ended incomplete.
        """
        status = response.status or "unknown"
        if status != ResponseStatus.COMPLETED:
            raise OpenAIResponseUnusable(
                status=status,
                reason=OpenAIClient._failure_reason(response),
            )

    @staticmethod
    def _extract_text(response: Response) -> str:
        """
        Return aggregated output text or reject an unusable response.
        """
        OpenAIClient._check_completed(response)
        if not response.output_text:
            raise OpenAIResponseUnusable(
                status=response.status,
                reason="the completed response contained no output text",
            )
        return response.output_text

    @staticmethod
    def _usage_of(response: Response) -> TokenUsage:
        """
        :return: The tokens a response consumed, none when the service reports none.
            The service counts the input tokens of the cache among all input tokens;
            here they are counted apart.
        """
        reported = response.usage
        if reported is None:
            return TokenUsage()
        details = reported.input_tokens_details
        return TokenUsage(
            input_tokens=reported.input_tokens
            - details.cached_tokens
            - details.cache_write_tokens,
            output_tokens=reported.output_tokens,
            cache_creation_input_tokens=details.cache_write_tokens,
            cache_read_input_tokens=details.cached_tokens,
        )

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
