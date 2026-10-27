"""
The language models the add-on asks, each chosen in the environment: the provider
serving it, the model, and what it charges.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields, replace
from enum import StrEnum
from pathlib import Path
from typing import Optional

from krrood.ormatic.utils import classproperty
from llm_reasoner import clients
from llm_reasoner.clients.base import LanguageModelClient
from llm_reasoner.clients.usage import TokenPrices

from cramera.resolution.exceptions import (
    MissingCredentialsError,
    UnreadableModelChoice,
    UnreadablePrices,
)

# %% what the environment names


class ModelVariable(StrEnum):
    """
    The environment variables choosing the models the add-on asks and stating their
    prices.
    """

    RESOLVER_MODEL = "CRAMERA_RESOLVER_MODEL"
    """
    The model resolving designators, as ``provider:model``.
    """

    RESOLVER_PRICES = "CRAMERA_RESOLVER_PRICES"
    """
    What the resolver's model charges, as a JSON object of dollars per million tokens of
    each kind.
    """

    KNOWLEDGE_MODEL = "CRAMERA_KNOWLEDGE_MODEL"
    """
    The model generating hypothesis knowledge, as ``provider:model``.
    """

    KNOWLEDGE_PRICES = "CRAMERA_KNOWLEDGE_PRICES"
    """
    What the knowledge model charges, as a JSON object of dollars per million tokens of
    each kind.
    """


class ApiKeyVariable(StrEnum):
    """
    The environment variables holding the key each provider's client authenticates with,
    named for the provider.
    """

    ANTHROPIC = "ANTHROPIC_API_KEY"
    OPENAI = "OPENAI_API_KEY"


class ModelProvider(StrEnum):
    """
    The services a model is asked through, each named as its client names it in a
    model's identity.
    """

    ANTHROPIC = "anthropic"
    OPENAI = "openai"

    @property
    def api_key_variable(self) -> ApiKeyVariable:
        """
        :return: The environment variable holding the key the provider's client
            authenticates with.
        """
        return ApiKeyVariable[self.name]


# %% a model chosen


@dataclass(frozen=True)
class ModelChoice:
    """
    A language model the add-on asks: who serves it, which it is, and what it charges.
    """

    provider: ModelProvider
    """
    The service the model is asked through.
    """

    model: str
    """
    The model, by the identifier its provider gives it.
    """

    prices: Optional[TokenPrices] = None
    """
    What the model charges; ``None`` when that is not known, which leaves the cost of
    what it is asked unknown.
    """

    @classproperty
    def identity_separator(cls) -> str:
        """
        :return: What separates the provider from the model in a model's identity.
        """
        return ":"

    @property
    def identity(self) -> str:
        """
        :return: The provider and the model, as a client names them in provenance.
        """
        return f"{self.provider}{self.identity_separator}{self.model}"

    @classmethod
    def from_environment(
        cls,
        model_variable: ModelVariable,
        prices_variable: ModelVariable,
        default: ModelChoice,
    ) -> ModelChoice:
        """
        :param model_variable: The variable choosing the model.
        :param prices_variable: The variable stating what the model charges.
        :param default: The model asked when the variable chooses none; its prices are
            also those of the same model named in the variable.
        :return: The model the environment chooses, at the prices it states, else at
            those known for it.
        :raises UnreadableModelChoice: When the choice names no provider and model.
        :raises UnreadablePrices: When the prices are not one price for each kind of
            token.
        """
        identity = os.environ.get(model_variable) or default.identity
        chosen = default if identity == default.identity else cls.of_identity(identity)
        stated_prices = os.environ.get(prices_variable)
        if not stated_prices:
            return chosen
        return replace(chosen, prices=cls.prices_stated_in(stated_prices))

    @classmethod
    def of_identity(cls, identity: str) -> ModelChoice:
        """
        :param identity: The provider and the model, as ``provider:model``.
        :return: That model, at no known prices.
        :raises UnreadableModelChoice: When the identity names no provider and model.
        """
        provider, _, model = identity.partition(cls.identity_separator)
        providers = [known.value for known in ModelProvider]
        if provider not in providers or not model:
            raise UnreadableModelChoice(stated=identity, providers=", ".join(providers))
        return cls(provider=ModelProvider(provider), model=model)

    @staticmethod
    def prices_stated_in(stated: str) -> TokenPrices:
        """
        :param stated: A JSON object of dollars per million tokens of each kind.
        :return: The prices it states.
        :raises UnreadablePrices: When it is not one price for each kind of token.
        """
        prices = json.loads(stated)
        kinds = [kind.name for kind in fields(TokenPrices)]
        if not isinstance(prices, dict) or sorted(prices) != sorted(kinds):
            raise UnreadablePrices(stated=stated, kinds=", ".join(kinds))
        return TokenPrices(**prices)

    def client(self, answer_token_limit: Optional[int] = None) -> LanguageModelClient:
        """
        :param answer_token_limit: The most tokens the model may answer an ordinary or
            tool-using turn with; the client's own limit when ``None``. An OpenAI
            model's limit counts its reasoning as well, so it keeps its own.
        :return: A client asking the model.
        """
        if self.provider is ModelProvider.OPENAI:
            return clients.OpenAIClient(model=self.model)
        if answer_token_limit is None:
            return clients.AnthropicClient(model=self.model)
        return clients.AnthropicClient(model=self.model, max_tokens=answer_token_limit)

    def require_api_key(self, searched: Optional[Path] = None) -> None:
        """
        :param searched: The ``.env`` file that was read for the key, if any.
        :raises MissingCredentialsError: When the provider's key is not set.
        """
        variable = self.provider.api_key_variable
        if not os.environ.get(variable):
            raise MissingCredentialsError(variable=variable, searched=searched)
