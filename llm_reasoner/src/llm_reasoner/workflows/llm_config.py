"""
LLM factory — lightweight, provider-agnostic.

Design difference from llmr/workflows/llm_configuration.py:
  - No global singletons (no module-level default_llm, gpt_llm_small, etc.)
  - No hardcoded default model names
  - No abstract base class wrapping LangChain — use BaseChatModel directly
  - Users create their LLM explicitly and inject it into LLMBackend

This keeps the dependency surface minimal: only langchain-core is required.
Provider-specific packages (langchain-openai, langchain-ollama) are optional
extras installed by the user based on their setup.
"""
from __future__ import annotations

from enum import Enum
from typing import Any


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    OLLAMA = "ollama"


def make_llm(
    provider: LLMProvider,
    model: str,
    temperature: float = 0.0,
    **kwargs: Any,
):
    """
    Factory function for creating a LangChain-compatible chat model.

    The returned object is a standard LangChain BaseChatModel — it can be
    passed directly to LLMBackend, nl_plan(), nl_sequential(), or TaskDecomposer.

    :param provider:    LLM provider (OPENAI or OLLAMA).
    :param model:       Model name/identifier (e.g. "gpt-4o", "qwen3:14b").
    :param temperature: Sampling temperature. Use 0.0 for deterministic output.
    :param kwargs:      Additional provider-specific arguments passed to the client.
    :returns: A LangChain BaseChatModel instance.

    Example::

        from llm_reasoner.workflows.llm_config import make_llm, LLMProvider
        from llm_reasoner import nl_plan

        llm = make_llm(LLMProvider.OPENAI, model="gpt-4o")
        plan = nl_plan("pick up the milk", context=context, llm=llm)

        # Or with Ollama for local inference:
        llm = make_llm(LLMProvider.OLLAMA, model="qwen3:14b")
    """
    if provider == LLMProvider.OPENAI:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError(
                "langchain-openai is not installed. "
                "Run: pip install 'llm-reasoner[openai]'"
            ) from e
        return ChatOpenAI(model=model, temperature=temperature, **kwargs)

    if provider == LLMProvider.OLLAMA:
        try:
            from langchain_ollama import ChatOllama
        except ImportError as e:
            raise ImportError(
                "langchain-ollama is not installed. "
                "Run: pip install 'llm-reasoner[ollama]'"
            ) from e
        return ChatOllama(model=model, temperature=temperature, **kwargs)

    raise ValueError(
        f"Unknown LLM provider: {provider!r}. "
        f"Valid options: {[p.value for p in LLMProvider]}"
    )
