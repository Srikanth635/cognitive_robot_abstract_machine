"""LLM provider configuration — unchanged from original llmr."""

from __future__ import annotations

import os
import pathlib
from abc import ABC, abstractmethod
from typing_extensions import Dict, Iterator, Type

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
    def _initialize_client(self) -> object: ...

    @abstractmethod
    def invoke(self, prompt: str, **kwargs: object) -> str: ...

    @abstractmethod
    def stream(self, prompt: str, **kwargs: object) -> Iterator[str]: ...

    @abstractmethod
    def with_structured_output(self, schema: object, **kwargs: object) -> object: ...


class OllamaLLM(LLMs):
    def _initialize_client(self) -> ChatOllama:
        return ChatOllama(model=self.model_name, temperature=self.temperature, **self.kwargs)

    def invoke(self, prompt: str, **kwargs: object) -> str:
        return self.client.invoke(prompt, **kwargs).content

    def stream(self, prompt: str, **kwargs: object) -> Iterator[str]:
        for chunk in self.client.stream(prompt, **kwargs):
            yield chunk.content

    def with_structured_output(self, schema: object, **kwargs: object) -> object:
        return self.client.with_structured_output(schema, **kwargs)


class OpenAILLM(LLMs):
    def _initialize_client(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.model_name, api_key=_GPT_API_KEY,
            temperature=self.temperature, **self.kwargs,
        )

    def invoke(self, prompt: str, **kwargs: object) -> str:
        return self.client.invoke(prompt, **kwargs).content

    def stream(self, prompt: str, **kwargs: object) -> Iterator[str]:
        for chunk in self.client.stream(prompt, **kwargs):
            yield chunk.content

    def with_structured_output(self, schema: object, **kwargs: object) -> object:
        return self.client.with_structured_output(schema, **kwargs)


class LLMFactory:
    _PROVIDERS: Dict[str, Type[LLMs]] = {"openai": OpenAILLM, "ollama": OllamaLLM}

    @classmethod
    def create_llm(cls, provider: str = "ollama", model_name: str = "qwen3:14b", **kwargs: object) -> LLMs:
        key = provider.lower()
        if key not in cls._PROVIDERS:
            raise ValueError(f"Unknown provider '{provider}'. Choose from {list(cls._PROVIDERS)}")
        return cls._PROVIDERS[key](model_name=model_name, **kwargs)


# Lazy singletons
gpt_llm_small: LLMs
gpt_llm_large: LLMs
ollama_llm_large: LLMs
default_llm: LLMs

_LAZY_CONFIGS: dict = {
    "gpt_llm_small": ("openai", "gpt-4o-mini", 0.5),
    "gpt_llm_large": ("openai", "gpt-4o", 0.5),
    "ollama_llm_large": ("ollama", "qwen3:14b", 0.5),
}


def __getattr__(name: str) -> LLMs:
    if name in _LAZY_CONFIGS:
        provider, model_name, temperature = _LAZY_CONFIGS[name]
        instance = LLMFactory.create_llm(provider=provider, model_name=model_name, temperature=temperature)
        globals()[name] = instance
        return instance
    if name == "default_llm":
        instance = __getattr__("gpt_llm_small")
        globals()["default_llm"] = instance
        return instance
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
