"""
Generating a hypothesis about an action with a language model, as a krrood generative
backend.

The query is an underspecified match on a vocabulary class, such as
``the(PouringPhaseStructure)(decomposes_task=ConceptReference(...))``: the fields it fixes
are stated to the model as fixed values, everything else is for the model to fill. The
backend answers from the store when it holds a hypothesis for the request; otherwise a
:class:`~llm_reasoner.knowledge.generation.dialogue.GenerationDialogue` asks the model,
judging each answer and returning a rejected one for repair, and a
:class:`~llm_reasoner.knowledge.generation.recording.HypothesisRecorder` records and
stores the admitted hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from krrood.entity_query_language.backends import GenerativeBackend
from krrood.entity_query_language.query.match import Match
from krrood.symbol_graph.symbol_graph import SymbolGraph
from typing_extensions import Iterable, Optional, TypeVar

from llm_reasoner.clients.base import Exchange, LanguageModelClient
from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge.acceptance.rules import AcceptanceSuite
from llm_reasoner.knowledge.acceptance.suite import predictive_manipulation_suite
from llm_reasoner.knowledge.admission.consistency import ConsistencyCheck
from llm_reasoner.knowledge.generation.assessment import HypothesisAssessor
from llm_reasoner.knowledge.generation.dialogue import (
    GenerationDialogue,
    GenerationOutcome,
)
from llm_reasoner.knowledge.generation.prompts import GenerationPrompt
from llm_reasoner.knowledge.generation.recording import HypothesisRecorder
from llm_reasoner.knowledge.generation.request import HypothesisRequest
from llm_reasoner.knowledge.ontologies.linked import LinkedOntologyTerms
from llm_reasoner.knowledge.ontologies.vocabulary import Vocabulary
from llm_reasoner.knowledge.representation.rdf import GraphReader
from llm_reasoner.knowledge.storage.record import HypothesisRecord, qualified_name
from llm_reasoner.knowledge.storage.store import HypothesisStore

T = TypeVar("T")


@dataclass
class VocabularyUnknownToSymbolGraph(LLMReasonerError):
    """
    Krrood's symbol graph was created before the knowledge vocabulary was imported, so
    its class diagram lacks vocabulary classes and no hypothesis can be built or read.
    """

    unknown_classes: list[str]
    """
    The module and class name of every vocabulary class the graph lacks.
    """

    def __post_init__(self):
        super().__init__(
            f"krrood's symbol graph lacks {len(self.unknown_classes)} knowledge "
            f"vocabulary classes, such as {self.unknown_classes[0]}; create the graph "
            "after importing llm_reasoner.knowledge"
        )


@dataclass
class HypothesisBackend(GenerativeBackend):
    """
    Generates a hypothesis that fills an underspecified match on a vocabulary class.
    """

    client: LanguageModelClient
    """
    The language model.
    """

    instruction: str = ""
    """
    The action the hypothesis is generated for, in plain words, such as "pour milk from
    the carton into the bowl"; recorded as provenance and stated to the model, but not
    part of what the store finds a hypothesis by.

    The type being generated when empty.
    """

    suite: AcceptanceSuite = field(default_factory=predictive_manipulation_suite)
    """
    The acceptance rules an answer must pass.
    """

    max_repairs: int = field(kw_only=True, default=2)
    """
    How many times a rejected answer is returned to the model for correction.
    """

    prompt: GenerationPrompt = field(default_factory=GenerationPrompt)
    """
    Builds the prompts.
    """

    consistency: Optional[ConsistencyCheck] = field(kw_only=True, default=None)
    """
    The reasoner checking an answer against the ontology in the open world, once it
    passes the acceptance rules; no open-world check when absent.
    """

    linked_terms: Optional[LinkedOntologyTerms] = field(kw_only=True, default=None)
    """
    The terms CRAM-AICOR and DUL declare, which concepts named in their namespaces must
    be among; without them such concepts are not checked.
    """

    store: Optional[HypothesisStore] = field(kw_only=True, default=None)
    """
    Where admitted hypotheses are stored and looked up before asking the model; no
    storing when absent.
    """

    log: list[Exchange] = field(default_factory=list, init=False)
    """
    Every exchange of the most recent generation, in order.
    """

    last_record: Optional[HypothesisRecord] = field(default=None, init=False)
    """
    The record of the most recent hypothesis, generated or served from the store.
    """

    last_outcome: Optional[GenerationOutcome] = field(default=None, init=False)
    """
    The most recent admitted hypothesis, with its warnings and exchanges.
    """

    def _evaluate(self, expression: Match[T]) -> Iterable[T]:
        unknown = Vocabulary().classes_unknown_to(SymbolGraph().class_diagram)
        if unknown:
            raise VocabularyUnknownToSymbolGraph(
                unknown_classes=[qualified_name(generated) for generated in unknown]
            )
        request = HypothesisRequest.from_match(expression, self.instruction)
        stored = (
            self.store.find(
                qualified_name(request.root_type), request.written_fixed_values
            )
            if self.store is not None
            else None
        )
        if stored is not None:
            self.last_record = stored.record
            yield GraphReader().read(stored.graph).root_of(request.root_type)
            return
        dialogue = GenerationDialogue(
            client=self.client,
            assessor=HypothesisAssessor(suite=self.suite, consistency=self.consistency),
            prompt=self.prompt,
            max_repairs=self.max_repairs,
            linked_terms=self.linked_terms,
        )
        self.log = dialogue.exchanges
        self.last_outcome = dialogue.generate(request)
        self.last_record = HypothesisRecorder(
            client_identity=self.client.identity,
            consistency_checked=self.consistency is not None,
            store=self.store,
        ).record(self.last_outcome, request)
        yield self.last_outcome.hypothesis.root
