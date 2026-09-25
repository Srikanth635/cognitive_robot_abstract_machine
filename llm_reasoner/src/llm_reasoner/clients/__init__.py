"""
Language-model client interfaces and implementations.
"""

from typing import TYPE_CHECKING

from llm_reasoner.clients.base import (
    Exchange,
    LanguageModelClient,
    ScriptedClient,
    ScriptExhausted,
)
from llm_reasoner.exceptions import LLMReasonerError

if TYPE_CHECKING:
    from llm_reasoner.clients.anthropic_client import AnthropicClient
    from llm_reasoner.clients.openai_client import (
        OpenAIClient,
        OpenAIResponseUnusable,
    )

__all__ = [
    "AnthropicClient",
    "Exchange",
    "LanguageModelClient",
    "OpenAIClient",
    "OpenAIResponseUnusable",
    "ScriptedClient",
    "ScriptExhausted",
]


def __getattr__(name: str) -> type[LanguageModelClient] | type[LLMReasonerError]:
    """
    Import vendor clients lazily.

    Keeps tests and offline tooling able to use ``ScriptedClient`` without the vendor
    SDKs installed, while package-level client imports still work.
    """
    if name == "AnthropicClient":
        from llm_reasoner.clients.anthropic_client import AnthropicClient

        return AnthropicClient
    if name in {"OpenAIClient", "OpenAIResponseUnusable"}:
        from llm_reasoner.clients.openai_client import (
            OpenAIClient,
            OpenAIResponseUnusable,
        )

        return {
            "OpenAIClient": OpenAIClient,
            "OpenAIResponseUnusable": OpenAIResponseUnusable,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
