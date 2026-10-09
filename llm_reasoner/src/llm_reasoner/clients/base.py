from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import Any, List, Mapping, Optional, Sequence, Tuple

from llm_reasoner.clients.tools import (
    ScriptedTurn,
    Tool,
    ToolBox,
    ToolCall,
    ToolConversation,
    ToolUsingClient,
)
from llm_reasoner.exceptions import LLMReasonerError

# %% client interface


@dataclass
class LanguageModelClient(ABC):
    """
    A single-turn completion interface, kept minimal so vendors stay swappable.
    """

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """
        :param system: The system prompt stating the role and output contract.
        :param user: The user prompt carrying schema, constraints and situation.
        :return: The raw response text.
        """

    def complete_structured(
        self, system: str, user: str, json_schema: Mapping[str, Any]
    ) -> str:
        """
        Complete under a schema the response must conform to.

        Optional, and falling back to :meth:`complete` so that a client without
        constrained decoding keeps working rather than failing. Vendors able to
        constrain the decoder override this and the schema does real work; the rest
        degrade to instructing the model in the prompt and hoping.

        The return type is deliberately the raw text either way. Provenance records what
        the model actually said, and a decoded object would lose the thing a rejected
        response has to be shown to correct.

        Worth overriding wherever possible. Recovering JSON from free text is workable
        for a flat list of values; it does not survive a hypothesis, which is a graph of
        a hundred interlinked fragments where one malformed bracket invalidates
        everything and the model has no way to know which fragment was at fault.

        :param json_schema: JSON Schema the response must satisfy.
        :return: The raw response text.
        """
        return self.complete(system, user)

    @property
    def supports_structured_output(self) -> bool:
        """
        :return: Whether :meth:`complete_structured` constrains the decoder
            rather than falling back. Callers may use this to decide how much
            format instruction to spend prompt budget on.
        """
        return False

    @property
    def identity(self) -> str:
        """
        A short identifier recorded alongside every exchange.

        Includes the model when the client carries one, because the identity is part of
        a hypothesis's cache key: two different models behind a single client class
        would otherwise report identically, and swapping the model would never
        invalidate anything generated under the old one.

        :return: The client class, and the model where one is known.
        """
        model = getattr(self, "model", None)
        return f"{type(self).__name__}/{model}" if model else type(self).__name__


# %% provenance


@dataclass
class Exchange:
    """
    One completion, retained so a result can be traced to the prompt that produced it.

    Lives beside the interface it records rather than in either half of the reasoner:
    its fields mirror :meth:`LanguageModelClient.complete` exactly, and both resolution
    and hypothesis generation need the same record of the same call. Keeping it here is
    what lets ``knowledge`` retain provenance without importing ``resolver``.

    Callers accumulate these in order, repairs included. The whole sequence is worth
    keeping rather than only the completion that succeeded: a result that needed two
    corrections carries evidence of why, which is diagnostic information about the
    prompt and the schema, not only about the model.
    """

    system: str
    """
    The system prompt that was sent.
    """

    user: str
    """
    The user prompt that was sent.
    """

    raw_response: str
    """
    The response that came back.
    """

    client_identity: str
    """
    Which model produced the response, as reported by the client.
    """

    attempt: int
    """
    Which attempt this was, counting repairs from zero.
    """


# %% offline client


@dataclass
class ScriptedClient(LanguageModelClient, ToolUsingClient):
    """
    Replays a fixed list of responses without any network access.

    Makes reflection, prompt assembly, decoding, write-back and validation testable
    deterministically.
    """

    responses: List[str] = field(default_factory=list)
    """
    Responses returned in order, one per completion.
    """

    tool_turns: List[ScriptedTurn] = field(default_factory=list)
    """
    The turns of completions with tools, used up in order: the tool calls of each turn,
    which are run against the tools offered, and the answer ending each completion.
    """

    calls: List[Tuple[str, str]] = field(default_factory=list)
    """
    The system and user prompts that were issued, in order.
    """

    schemas: List[Optional[Mapping[str, Any]]] = field(default_factory=list)
    """
    The schema passed with each call, ``None`` where none was.

    Retained so a test can assert that a caller derived and passed the schema it should
    have, which is otherwise invisible: the fallback path returns the same text either
    way.
    """

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        self.schemas.append(None)
        if not self.responses:
            raise ScriptExhausted(len(self.calls))
        return self.responses.pop(0)

    def complete_structured(
        self, system: str, user: str, json_schema: Mapping[str, Any]
    ) -> str:
        self.calls.append((system, user))
        self.schemas.append(json_schema)
        if not self.responses:
            raise ScriptExhausted(len(self.calls))
        return self.responses.pop(0)

    def complete_with_tools(
        self, system: str, user: str, tools: Sequence[Tool]
    ) -> ToolConversation:
        self.calls.append((system, user))
        toolbox = ToolBox(tools=tools)
        calls: List[ToolCall] = []
        while self.tool_turns:
            turn = self.tool_turns.pop(0)
            if isinstance(turn, str):
                return ToolConversation(answer=turn, calls=calls)
            calls.extend(
                ToolCall(
                    name=request.name,
                    arguments=request.arguments,
                    outcome=toolbox.call(request.name, request.arguments),
                )
                for request in turn
            )
        raise ScriptExhausted(len(self.calls))

    @property
    def identity(self) -> str:
        return "scripted"


@dataclass
class ScriptExhausted(LLMReasonerError):
    """
    Raised when a scripted client is asked for more responses than it holds.
    """

    calls: int
    """
    How many completions had been requested when the script ran out.
    """

    def __str__(self) -> str:
        return f"Scripted client ran out of responses after {self.calls} call(s)."
