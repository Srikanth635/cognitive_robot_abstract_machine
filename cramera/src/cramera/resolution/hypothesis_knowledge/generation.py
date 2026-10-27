"""
Turning a natural-language instruction into displayed, queryable knowledge.

An instruction is answered by ``llm_reasoner``'s
:class:`~llm_reasoner.knowledge.hypotheses.service.KnowledgeService`: a stored hypothesis for the
same instruction is reused, and otherwise a model generates one, which is checked
against the acceptance rules, repaired when it breaks them, and stored. The resulting
view becomes the viewer's active knowledge -- projected and exposed as EQL domains
exactly like the checked-in sample.

A generation that is never admitted raises
:class:`~llm_reasoner.knowledge.generation.backend.HypothesisRejected`, naming the problems of the
last answer, and the previously active knowledge stays displayed.

.. note::
   Every instruction is answered with a phase structure decomposing CRAM-AICOR's
   *Pouring* task. The instruction text steers the model, but choosing a different task
   from arbitrary English is a separate problem this does not attempt.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from krrood.entity_query_language.factories import the
from krrood.entity_query_language.query.match import Match
from krrood.ormatic.utils import classproperty
from llm_reasoner.knowledge.access import (
    AcceptanceSuite,
    ConceptReference,
    ConsistencyCheck,
    CRAM_AICOR_NAMESPACE,
    HermitCheck,
    HypothesisBackend,
    HypothesisStore,
    HypothesisView,
    KnowledgeService,
    LinkedOntologyTerms,
    pouring,
    predictive_manipulation_suite,
)
from llm_reasoner.clients.base import LanguageModelClient
from llm_reasoner.clients.usage import SONNET_5_5_PRICES
from typing_extensions import Optional

from cramera.resolution import paths
from cramera.resolution.exceptions import MissingCredentialsError
from cramera.resolution.hypothesis_knowledge import HypothesisKnowledge, activate
from cramera.resolution.model_choice import (
    ApiKeyVariable,
    ModelChoice,
    ModelProvider,
    ModelVariable,
)

POURING_TASK = ConceptReference(iri=CRAM_AICOR_NAMESPACE + "Pouring")
"""
The CRAM-AICOR task every generated structure decomposes.
"""

MAX_REPAIRS = 4
"""
How many times a rejected answer may be fed its own problems and retried.
"""

GENERATION_MODEL = "claude-sonnet-5-5"
"""
The model generating hypotheses when no client is given: answers describe a few hundred
linked fragments, which smaller models type inconsistently.
"""

ANTHROPIC_API_KEY_NAME = ApiKeyVariable.ANTHROPIC
"""
The environment variable holding the key the default model's client authenticates with.
"""


def pouring_query() -> Match:
    """
    :return: A query for one pouring phase structure decomposing the pouring task, all
        else open.
    """
    return the(pouring.PouringPhaseStructure)(decomposes_task=POURING_TASK)


# %% credentials
def _find_environment_file() -> Optional[Path]:
    """
    The nearest ``.env``, searched from the working directory and then from this package
    upwards, so the viewer finds the workspace's file wherever it is started.
    """
    working_directory = Path.cwd().resolve()
    package_directory = Path(__file__).resolve()
    for directory in (
        working_directory,
        *working_directory.parents,
        *package_directory.parents,
    ):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def _load_environment_file(environment_file: Path) -> None:
    """
    Read ``KEY=value`` lines into the environment without overriding what is set.

    ``python-dotenv`` is used when available so behaviour matches the notebooks;
    otherwise the same simple format is parsed directly, since :mod:`cramera.resolution`
    keeps its own dependencies to the standard library.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        for line in environment_file.read_text().splitlines():
            entry = line.strip()
            if not entry or entry.startswith("#") or "=" not in entry:
                continue
            name, _, value = entry.partition("=")
            os.environ.setdefault(name.strip(), value.strip().strip("\"'"))
        return
    load_dotenv(environment_file, override=False)


# %% sources
class HypothesisSource(ABC):
    """
    Something that answers an instruction with a stored hypothesis.
    """

    @abstractmethod
    def view_for(self, instruction: str) -> HypothesisView:
        """
        :return: The stored hypothesis describing what the instruction's action is
            generally like.
        """


@dataclass
class ModelHypothesisSource(HypothesisSource):
    """
    Answers instructions with a language model, consulting the store first.
    """

    store_directory: Optional[Path] = None
    """
    Where admitted hypotheses are stored;
    :func:`~cramera.resolution.paths.hypothesis_store_dir` when unset.
    """

    client: Optional[LanguageModelClient] = None
    """
    The model client; one asking :meth:`chosen_model` is built on demand when unset.
    """

    environment_file: Optional[Path] = None
    """
    The ``.env`` to read credentials from; searched for from the working directory
    upwards when unset.
    """

    model: Optional[ModelChoice] = None
    """
    The model the default client generates with; the one the environment chooses
    (:attr:`~cramera.resolution.model_choice.ModelVariable.KNOWLEDGE_MODEL`) when unset.
    """

    suite: AcceptanceSuite = field(default_factory=predictive_manipulation_suite)
    """
    The acceptance rules a generated hypothesis must meet.
    """

    consistency: Optional[ConsistencyCheck] = field(default_factory=HermitCheck.pinned)
    """
    The open-world check against the ontology and the pinned CRAM-AICOR and DUL; none
    skips it.
    """

    linked_terms: LinkedOntologyTerms = field(
        default_factory=LinkedOntologyTerms.pinned
    )
    """
    The classes the pinned CRAM-AICOR and DUL declare, which the concepts of an answer
    named in their namespaces must be among.
    """

    _service: Optional[KnowledgeService] = field(default=None, init=False)
    """
    The knowledge service, built on first use.
    """

    @classproperty
    def default_model(cls) -> ModelChoice:
        """
        :return: The model asked unless the environment chooses another, at its prices.
        """
        return ModelChoice(
            provider=ModelProvider.ANTHROPIC,
            model=GENERATION_MODEL,
            prices=SONNET_5_5_PRICES,
        )

    def chosen_model(self) -> ModelChoice:
        """
        :return: The model generating hypotheses: the one given, else the one the
            environment chooses.
        """
        return self.model or ModelChoice.from_environment(
            ModelVariable.KNOWLEDGE_MODEL,
            ModelVariable.KNOWLEDGE_PRICES,
            default=self.default_model,
        )

    def load_credentials(self) -> None:
        """
        Make the API key and the model choices visible to the client.

        The viewer is normally started from a shell that never sourced ``.env``, so the
        file the notebooks rely on is read here too. Values already exported win, and an
        injected client is assumed to carry its own credentials.

        :raises MissingCredentialsError: if the key of the chosen model's provider is
            not available.
        """
        if self.client is not None:
            return
        environment_file = self.environment_file or _find_environment_file()
        if environment_file is not None and environment_file.is_file():
            _load_environment_file(environment_file)
        self.chosen_model().require_api_key(environment_file)

    def view_for(self, instruction: str) -> HypothesisView:
        return self.knowledge_service().view_for(pouring_query(), instruction)

    def client_for_generation(self) -> LanguageModelClient:
        """
        :return: The given client, or a client asking the chosen model.
        """
        return self.client or self.chosen_model().client()

    def knowledge_service(self) -> KnowledgeService:
        """
        Build the service on first use, so importing costs no credentials.
        """
        self.load_credentials()
        if self._service is None:
            self._service = KnowledgeService(
                backend=HypothesisBackend(
                    client=self.client_for_generation(),
                    suite=self.suite,
                    max_repairs=MAX_REPAIRS,
                    consistency=self.consistency,
                    linked_terms=self.linked_terms,
                    store=HypothesisStore(
                        directory=self.store_directory or paths.hypothesis_store_dir()
                    ),
                )
            )
        return self._service


# %% service
@dataclass
class HypothesisGenerationService:
    """
    Makes a generated hypothesis the knowledge the viewer shows and queries.
    """

    source: HypothesisSource
    """
    Where hypotheses come from.
    """

    def generate(self, instruction: str) -> HypothesisKnowledge:
        """
        Answer an instruction and make the answer the active knowledge.

        :raises HypothesisRejected: If no generated answer is admitted; the active
            knowledge is then left unchanged.
        """
        return activate(
            HypothesisKnowledge.from_view(
                self.source.view_for(instruction), generated=True
            )
        )
