"""
Tests of choosing the language models the add-on asks from the environment: the provider
and the model, what they charge, and the key they need.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields

import pytest
from llm_reasoner.clients.anthropic_client import AnthropicClient
from llm_reasoner.clients.openai_client import OpenAIClient
from llm_reasoner.clients.usage import TokenPrices

from cramera.resolution.exceptions import (
    MissingCredentialsError,
    UnreadableModelChoice,
    UnreadablePrices,
)
from cramera.resolution.hypothesis_knowledge.generation import ModelHypothesisSource
from cramera.resolution.model_choice import ModelChoice, ModelProvider, ModelVariable

STATED_PRICES = TokenPrices(
    input_tokens=1.25,
    output_tokens=10.0,
    cache_creation_input_tokens=0.0,
    cache_read_input_tokens=0.125,
)
"""
Prices a test states in the environment.
"""

# %% fixtures


@pytest.fixture
def default() -> ModelChoice:
    """
    The model the viewer asks unless the environment chooses another.
    """
    return ModelHypothesisSource.default_model


@pytest.fixture
def keys(monkeypatch):
    """
    A key for every provider, so that each provider's client can be built.
    """
    for provider in ModelProvider:
        monkeypatch.setenv(provider.api_key_variable, "a-key-for-the-test")


def resolver_choice(default: ModelChoice) -> ModelChoice:
    """
    :return: The resolver's model as the environment chooses it.
    """
    return ModelChoice.from_environment(
        ModelVariable.RESOLVER_MODEL, ModelVariable.RESOLVER_PRICES, default
    )


# %% the model chosen


class TestChosenModel:
    def test_without_a_choice_the_default_is_asked(self, default):
        assert resolver_choice(default) == default

    def test_the_environment_names_the_provider_and_the_model(
        self, monkeypatch, default
    ):
        monkeypatch.setenv(ModelVariable.RESOLVER_MODEL, "openai:gpt-5.4")
        assert resolver_choice(default) == ModelChoice(
            provider=ModelProvider.OPENAI, model="gpt-5.4"
        )

    def test_naming_the_default_model_keeps_its_prices(self, monkeypatch, default):
        monkeypatch.setenv(ModelVariable.RESOLVER_MODEL, default.identity)
        assert resolver_choice(default).prices == default.prices

    def test_prices_stated_in_the_environment_are_the_models(
        self, monkeypatch, default
    ):
        monkeypatch.setenv(ModelVariable.RESOLVER_MODEL, "openai:gpt-5.4")
        monkeypatch.setenv(
            ModelVariable.RESOLVER_PRICES, json.dumps(asdict(STATED_PRICES))
        )
        assert resolver_choice(default).prices == STATED_PRICES

    @pytest.mark.parametrize("stated", ["gpt-5.4", "mistral:large", "openai:"])
    def test_a_choice_naming_no_provider_and_model_is_refused(
        self, monkeypatch, default, stated
    ):
        monkeypatch.setenv(ModelVariable.RESOLVER_MODEL, stated)
        with pytest.raises(UnreadableModelChoice):
            resolver_choice(default)

    def test_prices_leaving_out_a_kind_of_token_are_refused(self, monkeypatch, default):
        stated = asdict(STATED_PRICES)
        del stated[fields(TokenPrices)[0].name]
        monkeypatch.setenv(ModelVariable.RESOLVER_PRICES, json.dumps(stated))
        with pytest.raises(UnreadablePrices):
            resolver_choice(default)


# %% the client asking the model


class TestClient:
    @pytest.mark.parametrize("provider", list(ModelProvider))
    def test_the_client_asks_the_chosen_model_of_the_chosen_provider(
        self, keys, provider
    ):
        choice = ModelChoice(provider=provider, model="a-model")
        assert choice.client().identity == choice.identity

    def test_an_anthropic_model_answers_within_the_limit_it_is_given(self, keys):
        client = ModelChoice(provider=ModelProvider.ANTHROPIC, model="a-model").client(
            answer_token_limit=1234
        )
        assert isinstance(client, AnthropicClient)
        assert client.max_tokens == 1234

    def test_an_openai_model_keeps_its_own_limit_which_counts_its_reasoning(self, keys):
        client = ModelChoice(provider=ModelProvider.OPENAI, model="a-model").client(
            answer_token_limit=1234
        )
        assert isinstance(client, OpenAIClient)
        assert client.max_output_tokens == next(
            field.default
            for field in fields(OpenAIClient)
            if field.name == "max_output_tokens"
        )


# %% the key the provider needs


class TestApiKey:
    def test_a_provider_whose_key_is_set_is_asked(self, keys):
        ModelChoice(provider=ModelProvider.OPENAI, model="a-model").require_api_key()

    def test_a_provider_whose_key_is_not_set_is_refused(self, monkeypatch):
        monkeypatch.delenv(ModelProvider.OPENAI.api_key_variable, raising=False)
        with pytest.raises(MissingCredentialsError):
            ModelChoice(
                provider=ModelProvider.OPENAI, model="a-model"
            ).require_api_key()
