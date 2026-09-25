from __future__ import annotations

import os
from dataclasses import dataclass, field

import anthropic
from typing_extensions import Any, Mapping, Optional

from llm_reasoner.clients.base import LanguageModelClient

# %% anthropic client

_STRUCTURED_TOOL_NAME = "emit_hypothesis"
"""Name of the synthetic tool used to force schema-conformant JSON.

Not a real tool: it exists only so ``tool_choice`` can force the model to
respond through it, which is Anthropic's actual mechanism for constrained JSON
output. Nothing calls it and nothing reads its name except this client.
"""


@dataclass
class AnthropicClient(LanguageModelClient):
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

    _client: anthropic.Anthropic = field(default=None, init=False, repr=False)
    """The underlying vendor client."""

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
            messages=[{"role": "user", "content": user}],
        )
        return self._extract_text(response)

    # -- structured -----------------------------------------------------

    def complete_structured(
        self, system: str, user: str, json_schema: Mapping[str, Any]
    ) -> str:
        """
        Force the response through ``json_schema`` using tool use.

        A forced tool call is Anthropic's actual constrained-decoding path: the
        model's output is checked against the tool's input schema before it
        ever reaches this client, so a malformed field or an invented type name
        is rejected earlier than a JSON parse of free text ever could catch it.

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
        with self._client.messages.stream(
            model=self.model,
            max_tokens=self._structured_token_budget(json_schema),
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[
                {
                    "name": _STRUCTURED_TOOL_NAME,
                    "description": "Emit the hypothesis.",
                    "input_schema": dict(json_schema),
                }
            ],
            tool_choice={"type": "tool", "name": _STRUCTURED_TOOL_NAME},
        ) as stream:
            response = stream.get_final_message()
        self._check_truncation(response)
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
    def _check_truncation(response: Any) -> None:
        """
        Fail loudly and specifically when the response was cut off.

        Without this, a truncated tool call surfaces several layers away as a
        generic JSON decode error with no indication of the actual cause, which
        is what made the original failure look like a formatting problem rather
        than a token-budget one.
        """
        if getattr(response, "stop_reason", None) == "max_tokens":
            raise RuntimeError(
                "Anthropic response was truncated at the token limit "
                f"({response.usage.output_tokens} output tokens used). "
                "The structured token budget was too low for this schema; "
                "this should not happen with the derived budget in "
                "_structured_token_budget -- if it does, the schema is larger "
                "than estimated and the budgeting heuristic needs revisiting."
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
        raise RuntimeError(
            "Anthropic response contained no call to the forced tool "
            f"{_STRUCTURED_TOOL_NAME!r}; got stop_reason="
            f"{getattr(response, 'stop_reason', None)!r}"
        )

    @property
    def identity(self) -> str:
        return f"anthropic:{self.model}"
