"""
Turning a natural-language instruction into displayed, queryable knowledge.

An instruction is answered by ``llm_reasoner``'s
:class:`~llm_reasoner.hypotheses.service.KnowledgeService`: a stored hypothesis for the
same instruction is reused, and otherwise a model generates one, which is checked
against the acceptance rules, repaired when it breaks them, and stored. The resulting
view becomes the viewer's active knowledge -- projected and exposed as EQL domains
exactly like the checked-in sample.

A generation that is never admitted raises
:class:`~llm_reasoner.generation.backend.HypothesisRejected`, naming the problems of the
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
from llm_reasoner.knowledge.acceptance.rules import AcceptanceSuite
from llm_reasoner.knowledge.acceptance.suite import predictive_manipulation_suite
from llm_reasoner.admission.consistency import ConsistencyCheck
from llm_reasoner.admission.hermit import HermitCheck
from llm_reasoner.admission.store import HypothesisStore
from llm_reasoner.clients import AnthropicClient
from llm_reasoner.clients.base import LanguageModelClient
from llm_reasoner.generation.backend import HypothesisBackend
from llm_reasoner.hypotheses.service import KnowledgeService
from llm_reasoner.hypotheses.view import HypothesisView
from llm_reasoner.knowledge.ontologies import CRAM_AICOR_NAMESPACE
from llm_reasoner.knowledge.ontologies.linked import LinkedOntologyTerms
from llm_reasoner.knowledge.ontologies.predictive_manipulation import pouring
from llm_reasoner.knowledge.ontologies.terms import ConceptReference
from typing_extensions import Optional

from cram_viz import paths
from cram_viz.hypothesis_knowledge import HypothesisKnowledge, activate

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

ANTHROPIC_API_KEY_NAME = "ANTHROPIC_API_KEY"
"""
The environment variable holding the key the default client authenticates with.
"""


def pouring_query() -> Match:
    """
    :return: A query for one pouring phase structure decomposing the pouring task, all
        else open.
    """
    return the(pouring.PouringPhaseStructure)(decomposes_task=POURING_TASK)


class MissingCredentialsError(RuntimeError):
    """
    Raised when no API key is available to authenticate the generation client.
    """

    def __init__(self, variable_name: str, searched: Optional[Path]) -> None:
        location = f" or in {searched}" if searched else ""
        super().__init__(
            f"{variable_name} is not set{location}. "
            f"Generation needs it; export it or add it to a .env file, then retry."
        )


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
    otherwise the same simple format is parsed directly, since :mod:`cram_viz` keeps its
    own dependencies to the standard library.
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
    Where admitted hypotheses are stored; :func:`~cram_viz.paths.hypothesis_store_dir`
    when unset.
    """

    client: Optional[LanguageModelClient] = None
    """
    The model client; an :class:`AnthropicClient` is built on demand when unset.
    """

    environment_file: Optional[Path] = None
    """
    The ``.env`` to read credentials from; searched for from the working directory
    upwards when unset.
    """

    model: str = GENERATION_MODEL
    """
    The model the default client generates with.
    """

    suite: AcceptanceSuite = field(default_factory=predictive_manipulation_suite)
    """
    The acceptance rules a generated hypothesis must meet.
    """

    consistency: Optional[ConsistencyCheck] = field(
        default_factory=HermitCheck.from_environment
    )
    """
    The open-world check against the ontology, CRAM-AICOR and DUL; configured from the
    environment variables naming the linked ontologies, and skipped without them.
    """

    linked_terms: Optional[LinkedOntologyTerms] = field(
        default_factory=LinkedOntologyTerms.from_environment
    )
    """
    The terms CRAM-AICOR and DUL declare, which the concepts of an answer named in their
    namespaces must be among; read from the environment variables naming the linked
    ontologies, and unchecked without them.
    """

    _service: Optional[KnowledgeService] = field(default=None, init=False)
    """
    The knowledge service, built on first use.
    """

    def load_credentials(self) -> None:
        """
        Make the API key visible to the client.

        The viewer is normally started from a shell that never sourced ``.env``, so the
        file the notebooks rely on is read here too. Values already exported win, and an
        injected client is assumed to carry its own credentials.

        :raises MissingCredentialsError: if no key is available.
        """
        if self.client is not None:
            return
        environment_file = self.environment_file or _find_environment_file()
        if environment_file is not None and environment_file.is_file():
            _load_environment_file(environment_file)
        if not os.environ.get(ANTHROPIC_API_KEY_NAME):
            raise MissingCredentialsError(ANTHROPIC_API_KEY_NAME, environment_file)

    def view_for(self, instruction: str) -> HypothesisView:
        return self._knowledge_service().view_for(instruction, pouring_query())

    def client_for_generation(self) -> LanguageModelClient:
        """
        :return: The given client, or an Anthropic client for the generation model.
        """
        return self.client or AnthropicClient(model=self.model)

    def _knowledge_service(self) -> KnowledgeService:
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
