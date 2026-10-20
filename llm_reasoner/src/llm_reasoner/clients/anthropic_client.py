from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum

import anthropic
from typing_extensions import Any, Dict, List, Mapping, Optional, Sequence

from llm_reasoner.clients.base import LanguageModelClient
from llm_reasoner.clients.usage import LatestUsage, TokenUsage, UsageReporting
from llm_reasoner.clients.tools import (
    ResponseTruncated,
    Tool,
    ToolBox,
    ToolCall,
    ToolConversation,
    ToolTurnsExhausted,
    ToolUsingClient,
)

# %% messages


class StopReason(StrEnum):
    """
    Why the model stopped writing a response.
    """

    TOOL_USE = "tool_use"
    """
    It is waiting for the results of the tools it called.
    """

    MAX_TOKENS = "max_tokens"
    """
    It was cut off at the token limit of the request.
    """


class ContentBlockType(StrEnum):
    """
    The kinds of content block a conversation is made of.
    """

    TEXT = "text"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"


class MessageKey(StrEnum):
    """
    The keys of the messages and tool results a conversation is sent as.
    """

    ROLE = "role"
    CONTENT = "content"
    TYPE = "type"
    TEXT = "text"
    TOOL_USE_ID = "tool_use_id"
    IS_ERROR = "is_error"
    CACHE_CONTROL = "cache_control"


class CacheType(StrEnum):
    """
    How long a cached prompt prefix is kept.
    """

    EPHEMERAL = "ephemeral"
    """
    Kept for minutes, refreshed by every read: long enough for the turns of one
    conversation.
    """


class MessageRole(StrEnum):
    """
    Who a message of a conversation is from.
    """

    USER = "user"
    ASSISTANT = "assistant"


# %% anthropic client

_STRUCTURED_TOOL_NAME = "emit_hypothesis"
"""Name of the synthetic tool the model answers through.

Not a real tool: it exists only so the answer arrives as the tool's input, checked
against the tool's input schema. Nothing calls it and nothing reads its name except
this client.
"""

_STRUCTURED_TOOL_INSTRUCTION = (
    "Answer by calling the {tool} tool exactly once, with the whole answer as its "
    "input."
)
"""Appended to the system prompt, since the call is asked for rather than forced."""


@dataclass
class AnthropicClient(LanguageModelClient, ToolUsingClient, UsageReporting):
    """Completes prompts with an Anthropic model."""

    model: str = "claude-haiku-4-5"
    """The model identifier. claude-sonnet-4-5"""

    max_tokens: int = 2048
    """Longest response requested for an ordinary, unstructured completion.

    NOT used for structured hypothesis generation -- see
    :meth:`complete_structured`. A fixed ceiling here was previously the cause
    of silent truncation: a pouring hypothesis with a dozen-plus fragments needs
    several thousand tokens of JSON, and 2048 cuts it off mid-structure on every
    attempt, which repair cannot fix because the retry is asked to emit the same
    size answer against the same ceiling.
    """

    api_key: Optional[str] = None
    """The API key, falling back to the ``ANTHROPIC_API_KEY`` environment variable."""

    max_tool_turns: int = 10
    """
    How many turns of tool calls a completion with tools may take before it is given
    up, so a model that keeps asking cannot run on without end.
    """

    _client: anthropic.Anthropic = field(default=None, init=False, repr=False)
    """The underlying vendor client."""

    _latest: LatestUsage = field(
        default_factory=LatestUsage, init=False, repr=False, compare=False
    )
    """The tokens of the latest plain or structured completion, for each thread."""

    def __post_init__(self) -> None:
        self._client = anthropic.Anthropic(
            api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    # -- unstructured -------------------------------------------------------

    def complete(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{MessageKey.ROLE: MessageRole.USER, MessageKey.CONTENT: user}],
        )
        self._latest.usage = self._usage_of(response)
        return self._extract_text(response)

    # -- structured -----------------------------------------------------

    def complete_structured(
        self, system: str, user: str, json_schema: Mapping[str, Any]
    ) -> str:
        """
        Ask for the response as the input of one tool call checked against
        ``json_schema``.

        The call is asked for in the system prompt rather than forced through
        ``tool_choice``, because the newest models reject forced tool use; with only
        one tool on offer and parallel calls disabled, the model makes that one
        call. An answer written as text instead is returned as text, for the
        decoder to read or reject.

        The system prompt, which a caller keeps the same across the attempts of a
        request, is cached together with the tool definition ahead of it.

        The token budget is sized from the model's real ceiling rather than a
        fixed constant, because a hypothesis's length scales with how many
        fragment types the schema offers and a constant ceiling silently
        truncates as that grows -- which is exactly what happened with the
        unstructured default above.

        Issued as a STREAMED request, not because the caller wants a stream --
        it is collected in full before returning -- but because the SDK itself
        refuses a non-streamed call whose declared ``max_tokens`` could
        plausibly run past its timeout, and every current model's real output
        ceiling is over that line. Streaming and waiting on
        ``get_final_message()`` sidesteps the SDK's own guard without changing
        the shape of what this method returns: the accumulated message is
        identical in structure to what ``messages.create`` would give.
        """
        request_limit = self._structured_token_budget(json_schema)
        with self._client.messages.stream(
            model=self.model,
            max_tokens=request_limit,
            system=[
                {
                    "type": "text",
                    "text": f"{system}\n\n"
                    + _STRUCTURED_TOOL_INSTRUCTION.format(tool=_STRUCTURED_TOOL_NAME),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{MessageKey.ROLE: MessageRole.USER, MessageKey.CONTENT: user}],
            tools=[
                {
                    "name": _STRUCTURED_TOOL_NAME,
                    "description": "Emit the hypothesis.",
                    "input_schema": dict(json_schema),
                }
            ],
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        ) as stream:
            response = stream.get_final_message()
        self._latest.usage = self._usage_of(response)
        self._check_truncation(response, request_limit)
        return self._extract_tool_input(response)

    @property
    def supports_structured_output(self) -> bool:
        return True

    # -- token budgeting ------------------------------------------------

    # Confirmed against Anthropic's model documentation and third-party model
    # cards as of August 2026. Matched by substring so a dated model id
    # ("claude-haiku-4-5-20251001") still resolves. THIS DRIFTS: check
    # https://docs.anthropic.com/en/docs/about-claude/models/overview before
    # trusting it for a model released after this was written, and prefer
    # widening the fallback over narrowing a real ceiling.
    _MAX_OUTPUT_TOKENS_BY_MODEL = (
        # order matters: checked top to bottom, first match wins, so the
        # narrower/older pattern must precede the newer one it is a substring
        # risk against. Both hyphen orderings of "3-5" and "haiku" are listed
        # because Anthropic's model ids are not consistent about which comes
        # first ("claude-3-5-haiku-..." vs a hypothetical "haiku-3-5-...").
        ("haiku-3-5", 8_192),
        ("3-5-haiku", 8_192),
        ("haiku-4", 64_000),
        ("sonnet-4-5", 64_000),
        ("opus-4-5", 64_000),
        ("sonnet", 128_000),
        ("opus", 128_000),
        ("fable", 128_000),
        ("mythos", 128_000),
    )
    _FALLBACK_MAX_OUTPUT_TOKENS = 64_000
    """Used for a model id matching none of the above. Set to the smallest
    ceiling across current-generation models rather than the largest, so an
    unrecognised future model id fails toward a rejected request (visible
    immediately) rather than toward truncation (visible only after the JSON
    fails to parse, three layers away from the actual cause)."""

    def _structured_token_budget(self, json_schema: Mapping[str, Any]) -> int:
        """
        Use the model's real output ceiling, not an estimate of what a
        hypothesis needs.

        There is no cost to asking for more than gets used: Anthropic bills and
        rate-limits by tokens actually generated, not by the ``max_tokens``
        value requested, so a high cap that goes unused is free. Estimating a
        tighter budget from the schema was solving a cost problem that does not
        exist, at the price of a real one: the estimate landed at 8,192 tokens
        for a hypothesis that needed more, and the response was truncated
        exactly there.

        ``json_schema`` is accepted but unused. Kept in the signature because
        a future version of this estimate may want it again, and because it
        keeps the call site the same regardless of which strategy is behind it.
        """
        model = self.model.lower()
        for needle, ceiling in self._MAX_OUTPUT_TOKENS_BY_MODEL:
            if needle in model:
                return ceiling
        return self._FALLBACK_MAX_OUTPUT_TOKENS

    # -- response handling ------------------------------------------------

    @staticmethod
    def _check_truncation(response: Any, token_limit: int) -> None:
        """
        Fail specifically when the response was cut off at the request's token limit.

        :param response: The model's response.
        :param token_limit: The most output tokens the request allowed.
        :raises ResponseTruncated: When the model was cut off.
        """
        if response.stop_reason == StopReason.MAX_TOKENS:
            raise ResponseTruncated(
                output_tokens=response.usage.output_tokens,
                token_limit=token_limit,
                partial_answer=AnthropicClient._extract_text(response),
            )

    @property
    def latest_usage(self) -> TokenUsage:
        return self._latest.usage

    @staticmethod
    def _usage_of(response: Any) -> TokenUsage:
        """
        :return: The tokens a response consumed; cache figures the service leaves out
            count as none.
        """
        reported = response.usage
        return TokenUsage(
            input_tokens=reported.input_tokens,
            output_tokens=reported.output_tokens,
            cache_creation_input_tokens=reported.cache_creation_input_tokens or 0,
            cache_read_input_tokens=reported.cache_read_input_tokens or 0,
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
        return "".join(block.text for block in response.content if block.type == "text")

    @staticmethod
    def _extract_tool_input(response: Any) -> str:
        """
        Pull the tool call's input back out as JSON text.

        Returned as text rather than the parsed dict, matching every other
        client: provenance records what the model said, and the generation
        loop's own decoder turns text into fragments -- duplicating that here
        would mean two parsers that must agree.
        """
        import json

        for block in response.content:
            if block.type == "tool_use" and block.name == _STRUCTURED_TOOL_NAME:
                return json.dumps(block.input)
        return AnthropicClient._extract_text(response)

    # -- with tools -----------------------------------------------------

    def complete_with_tools(
        self, system: str, user: str, tools: Sequence[Tool]
    ) -> ToolConversation:
        """
        Let the model call tools, running each call and sending its result back, until
        it writes its answer.

        The loop is written out rather than left to the SDK's tool runner, because the
        tools are vendor-neutral objects that scripted clients run as well, and every
        call is recorded with what it returned.

        A turn that calls no tool ends the conversation, its text the answer, even when
        it stopped as if to call one; there is nothing to send back for it.

        Every turn resends the conversation so far, so the system prompt, which caches
        the tools ahead of it, carries a cache breakpoint of its own, and the request
        asks for the growing conversation to be cached up to its last block; each turn
        then reads what the turn before wrote.
        """
        cache = {MessageKey.TYPE: CacheType.EPHEMERAL}
        toolbox = ToolBox(tools=tools)
        messages: List[Dict[str, Any]] = [
            {MessageKey.ROLE: MessageRole.USER, MessageKey.CONTENT: user}
        ]
        calls: List[ToolCall] = []
        usage = TokenUsage()
        for _ in range(self.max_tool_turns):
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[
                    {
                        MessageKey.TYPE: ContentBlockType.TEXT,
                        MessageKey.TEXT: system,
                        MessageKey.CACHE_CONTROL: cache,
                    }
                ],
                messages=list(messages),
                tools=toolbox.definitions(),
                cache_control=cache,
            )
            usage = usage + self._usage_of(response)
            requested = [
                block
                for block in response.content
                if block.type == ContentBlockType.TOOL_USE
            ]
            if response.stop_reason != StopReason.TOOL_USE or not requested:
                self._check_truncation(response, self.max_tokens)
                return ToolConversation(
                    answer=self._extract_text(response), calls=calls, usage=usage
                )
            results = []
            for block in requested:
                outcome = toolbox.call(block.name, block.input)
                calls.append(
                    ToolCall(name=block.name, arguments=block.input, outcome=outcome)
                )
                results.append(
                    {
                        MessageKey.TYPE: ContentBlockType.TOOL_RESULT,
                        MessageKey.TOOL_USE_ID: block.id,
                        MessageKey.CONTENT: outcome.content,
                        MessageKey.IS_ERROR: outcome.is_error,
                    }
                )
            messages.append(
                {
                    MessageKey.ROLE: MessageRole.ASSISTANT,
                    MessageKey.CONTENT: response.content,
                }
            )
            messages.append(
                {MessageKey.ROLE: MessageRole.USER, MessageKey.CONTENT: results}
            )
        raise ToolTurnsExhausted(turns=self.max_tool_turns)

    @property
    def identity(self) -> str:
        return f"anthropic:{self.model}"
