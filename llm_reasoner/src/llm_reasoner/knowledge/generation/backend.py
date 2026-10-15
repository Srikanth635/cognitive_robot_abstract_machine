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
from typing_extensions import Iterable, Optional, TypeVar

from llm_reasoner.clients.base import Exchange, LanguageModelClient
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
from llm_reasoner.knowledge.representation.rdf import GraphReader
from llm_reasoner.observation.events import HypothesisRequested, HypothesisServed
from llm_reasoner.observation.sinks import EventSink, NullSink
from llm_reasoner.knowledge.storage.record import HypothesisRecord, qualified_name
from llm_reasoner.knowledge.storage.store import HypothesisStore

T = TypeVar("T")


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

    observer: EventSink = field(kw_only=True, default_factory=NullSink)
    """
    Where a request, each attempt at the model's answer and the hypothesis served are
    reported to; nowhere by default.
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
        request = HypothesisRequest.from_match(expression, self.instruction)
        self.observer.emit(
            HypothesisRequested(
                root_type=qualified_name(request.root_type),
                fixed_values=request.written_fixed_values,
                instruction=self.instruction,
            )
        )
        stored = (
            self.store.find(
                qualified_name(request.root_type), request.written_fixed_values
            )
            if self.store is not None
            else None
        )
        if stored is not None:
            self.last_record = stored.record
            self.observer.emit(
                HypothesisServed(
                    record_id=stored.record.record_id, from_store=True, attempts=0
                )
            )
            yield GraphReader().read(stored.graph).root_of(request.root_type)
            return
        dialogue = GenerationDialogue(
            client=self.client,
            assessor=HypothesisAssessor(
                suite=self.suite,
                consistency=(
                    None
                    if self.consistency is None
                    else self.consistency.covering(request.module_files)
                ),
            ),
            prompt=self.prompt,
            max_repairs=self.max_repairs,
            linked_terms=self.linked_terms,
            observer=self.observer,
        )
        self.log = dialogue.exchanges
        self.last_outcome = dialogue.generate(request)
        self.last_record = HypothesisRecorder(
            client_identity=self.client.identity,
            consistency_checked=self.consistency is not None,
            store=self.store,
        ).record(self.last_outcome, request)
        self.observer.emit(
            HypothesisServed(
                record_id=self.last_record.record_id,
                from_store=False,
                attempts=len(self.last_outcome.exchanges),
                warnings=[
                    f"{warning.rule}: {warning.reason}"
                    for warning in self.last_outcome.warnings
                ],
            )
        )
        yield self.last_outcome.hypothesis.root
