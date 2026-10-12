"""
Generating a hypothesis about an action with a language model, as a krrood generative
backend.

The query is an underspecified match on a vocabulary class, such as
``the(PouringPhaseStructure)(decomposes_task=ConceptReference(...))``: the fields it fixes
are stated to the model as fixed values, everything else is for the model to fill. The
answer is decoded into vocabulary instances and checked by the acceptance suite; what is
wrong is shown to the model, which answers again, until the hypothesis is admitted or the
repair budget is spent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from krrood.entity_query_language.backends import GenerativeBackend
from krrood.entity_query_language.query.match import Match
from typing_extensions import Iterable, Optional, TypeVar

from llm_reasoner.knowledge.acceptance.rules import AcceptanceSuite, Violation
from llm_reasoner.knowledge.storage.citation import Citation
from llm_reasoner.knowledge.admission.consistency import ConsistencyCheck
from llm_reasoner.knowledge.representation.rdf import GraphReader, GraphWriter
from llm_reasoner.knowledge.storage.record import (
    ConsistencyVerdict,
    HypothesisRecord,
    current_module_versions,
    qualified_name,
)
from llm_reasoner.knowledge.storage.store import HypothesisStore
from llm_reasoner.knowledge.acceptance.suite import predictive_manipulation_suite
from llm_reasoner.clients.base import Exchange, LanguageModelClient
from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge.generation.prompts import (
    GenerationPrompt,
    described_violation,
)
from llm_reasoner.knowledge.generation.assessment import HypothesisAssessor
from llm_reasoner.knowledge.generation.request import HypothesisRequest
from llm_reasoner.knowledge.generation.schema import GenerationSchema
from llm_reasoner.knowledge.ontologies.linked import LinkedOntologyTerms
from llm_reasoner.knowledge.generation.wire import FragmentDecoder
from llm_reasoner.knowledge.representation.hypothesis import DecodedHypothesis

T = TypeVar("T")

# %% exceptions


@dataclass
class HypothesisRejected(LLMReasonerError):
    """
    No answer was admitted within the repair budget.
    """

    problems: list[str]
    """
    What was wrong with the last answer.
    """

    exchanges: list[Exchange]
    """
    Every exchange of the generation, in order.
    """

    def __post_init__(self):
        super().__init__(
            f"no admissible hypothesis after {len(self.exchanges)} answers; the last "
            f"had {len(self.problems)} problems, the first being: "
            + (self.problems[0] if self.problems else "none")
        )


# %% outcome


@dataclass
class GenerationOutcome:
    """
    An admitted hypothesis, with what it took to get it.
    """

    hypothesis: DecodedHypothesis
    """
    The admitted hypothesis.
    """

    warnings: list[Violation]
    """
    The violations of warning severity it was admitted with.
    """

    exchanges: list[Exchange]
    """
    Every exchange of the generation, repairs included.
    """


@dataclass
class AnswerUnderRepair:
    """
    The answer repairs are applied to: the one with the fewest problems so far.
    """

    answer: str
    """
    The answer, in full, with every repair applied to it.
    """

    problems: list[str]
    """
    What is wrong with it.
    """


# %% backend


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
    The action to describe, in plain words, such as "pour milk from the carton into the
    bowl"; the type being generated when empty.
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
        request = HypothesisRequest.from_match(expression, self.instruction)
        fixed = request.fixed_values
        instruction = request.instruction
        written_fixed = request.written_fixed_values
        stored = (
            self.store.find(instruction, qualified_name(expression.type), written_fixed)
            if self.store is not None
            else None
        )
        if stored is not None:
            self.last_record = stored.record
            yield GraphReader().read(stored.graph).root_of(expression.type)
            return
        schema = GenerationSchema.for_root(expression.type)
        decoder = FragmentDecoder(schema, linked_terms=self.linked_terms)
        assessor = HypothesisAssessor(suite=self.suite, consistency=self.consistency)
        system_prompt = self.prompt.system(schema)
        initial_prompt = self.prompt.initial(schema, instruction, fixed)
        prompt = initial_prompt
        self.log = []
        under_repair: Optional[AnswerUnderRepair] = None
        for attempt in range(self.max_repairs + 1):
            raw_response = self.client.complete_structured(
                system_prompt, prompt, schema.json_schema()
            )
            self.log.append(
                Exchange(
                    system=system_prompt,
                    user=prompt,
                    raw_response=raw_response,
                    client_identity=self.client.identity,
                    attempt=attempt,
                )
            )
            answer = (
                raw_response
                if under_repair is None
                else decoder.merged(under_repair.answer, raw_response)
            )
            assessment = assessor.assess(decoder.decode(answer), request)
            if assessment.hypothesis is not None:
                self.last_outcome = GenerationOutcome(
                    hypothesis=assessment.hypothesis,
                    warnings=assessment.warnings,
                    exchanges=list(self.log),
                )
                self.admit(
                    self.last_outcome, instruction, expression.type, written_fixed
                )
                yield assessment.hypothesis.root
                return
            if under_repair is None or len(assessment.problems) <= len(
                under_repair.problems
            ):
                under_repair = AnswerUnderRepair(
                    answer=answer, problems=assessment.problems
                )
            prompt = self.prompt.repair(
                initial_prompt, under_repair.answer, under_repair.problems
            )
        raise HypothesisRejected(
            problems=under_repair.problems, exchanges=list(self.log)
        )

    def admit(
        self,
        outcome: GenerationOutcome,
        instruction: str,
        root_type: type,
        fixed_values: dict[str, str],
    ) -> None:
        """
        Record an admitted hypothesis, and store it when there is a store.
        """
        hypothesis = outcome.hypothesis
        self.last_record = HypothesisRecord(
            instruction=instruction,
            root_type=qualified_name(root_type),
            fixed_values=fixed_values,
            client_identity=self.client.identity,
            exchanges=outcome.exchanges,
            warnings=[
                described_violation(warning, hypothesis.identifier_of)
                for warning in outcome.warnings
            ],
            consistency=(
                ConsistencyVerdict.CONSISTENT
                if self.consistency is not None
                else ConsistencyVerdict.NOT_CHECKED
            ),
            module_versions=current_module_versions(),
        )
        if self.store is not None:
            namespace = Citation.namespace_of(self.last_record.record_id)
            self.store.put(
                self.last_record, GraphWriter(namespace=namespace).write(hypothesis)
            )
