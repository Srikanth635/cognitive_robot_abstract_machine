"""LLM provider abstractions for the generative backend.

Mirrors the structure of ``llmr.workflows.llm_configuration`` so that when llmr is
available in the same environment the two modules are drop-in compatible:

    # To switch to llmr's providers just replace this import everywhere:
    from llmr.workflows.llm_configuration import LLMFactory, gpt_llm_small, default_llm
"""

from __future__ import annotations

import os
import pathlib
from abc import ABC, abstractmethod

from dotenv import find_dotenv, load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

_ENV_FILE = pathlib.Path(__file__).parent / ".env"
load_dotenv(_ENV_FILE if _ENV_FILE.exists() else find_dotenv(), override=True)

_GPT_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")


class LLMs(ABC):
    """Abstract base class for different LLM providers."""

    def __init__(self, model_name: str, temperature: float = 0.7, **kwargs: object) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.kwargs = kwargs
        self.client = self._initialize_client()

    @abstractmethod
    def _initialize_client(self) -> object:
        """Initialise the provider-specific LLM client."""

    @abstractmethod
    def invoke(self, prompt: str, **kwargs: object) -> str:
        """Generate a plain-text response from the LLM."""

    @abstractmethod
    def stream(self, prompt: str, **kwargs: object):
        """Stream a response from the LLM chunk by chunk."""

    @abstractmethod
    def with_structured_output(self, schema: object, **kwargs: object) -> object:
        """Return a client configured to return structured output matching *schema*."""


class OllamaLLM(LLMs):
    """Ollama-hosted LLM (e.g. qwen3:14b)."""

    def _initialize_client(self) -> ChatOllama:
        return ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            **self.kwargs,
        )

    def invoke(self, prompt: str, **kwargs: object) -> str:
        return self.client.invoke(prompt, **kwargs).content

    def stream(self, prompt: str, **kwargs: object):
        for chunk in self.client.stream(prompt, **kwargs):
            yield chunk.content

    def with_structured_output(self, schema: object, **kwargs: object) -> object:
        return self.client.with_structured_output(schema, **kwargs)


class OpenAILLM(LLMs):
    """OpenAI GPT LLM."""

    def _initialize_client(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.model_name,
            api_key=_GPT_API_KEY,
            temperature=self.temperature,
            **self.kwargs,
        )

    def invoke(self, prompt: str, **kwargs: object) -> str:
        return self.client.invoke(prompt, **kwargs).content

    def stream(self, prompt: str, **kwargs: object):
        for chunk in self.client.stream(prompt, **kwargs):
            yield chunk.content

    def with_structured_output(self, schema: object, **kwargs: object) -> object:
        return self.client.with_structured_output(schema, **kwargs)


class LLMFactory:
    """Creates LLM instances by provider name."""

    _PROVIDERS: dict[str, type[LLMs]] = {
        "openai": OpenAILLM,
        "ollama": OllamaLLM,
    }

    @classmethod
    def create_llm(
        cls,
        provider: str = "ollama",
        model_name: str = "qwen3:14b",
        **kwargs: object,
    ) -> LLMs:
        key = provider.lower()
        if key not in cls._PROVIDERS:
            raise ValueError(
                f"Unknown provider '{provider}'. Choose from {list(cls._PROVIDERS)}"
            )
        return cls._PROVIDERS[key](model_name=model_name, **kwargs)


# ── Module-level singletons ────────────────────────────────────────────────────

gpt_llm_small: LLMs = LLMFactory.create_llm(
    provider="openai", model_name="gpt-4o-mini", temperature=0.5
)
gpt_llm_large: LLMs = LLMFactory.create_llm(
    provider="openai", model_name="gpt-4o", temperature=0.5
)
ollama_llm_large: LLMs = LLMFactory.create_llm(
    provider="ollama", model_name="qwen3:14b", temperature=0.5
)

#: Default client used across all agent nodes unless overridden.
default_llm = gpt_llm_small.client
