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
from operator import attrgetter

from krrood.entity_query_language.backends import GenerativeBackend
from krrood.entity_query_language.query.match import Match
from typing_extensions import Any, Iterable, Optional, TypeVar

from llm_reasoner.acceptance.rules import AcceptanceSuite, Severity, Violation
from llm_reasoner.admission.citation import Citation
from llm_reasoner.admission.consistency import (
    ConsistencyCheck,
    ContradictionLocaliser,
    OpenWorldCheck,
)
from llm_reasoner.admission.rdf import (
    HYPOTHESIS_NAMESPACE_ROOT,
    GraphReader,
    GraphWriter,
)
from llm_reasoner.admission.record import (
    ConsistencyVerdict,
    HypothesisRecord,
    current_module_versions,
    qualified_name,
)
from llm_reasoner.admission.store import HypothesisStore
from llm_reasoner.acceptance.suite import predictive_manipulation_suite
from llm_reasoner.clients.base import Exchange, LanguageModelClient
from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.generation.prompts import (
    GenerationPrompt,
    described_violation,
    written_value,
)
from llm_reasoner.generation.schema import GenerationSchema
from llm_reasoner.ontologies.linked import LinkedOntologyTerms
from llm_reasoner.generation.wire import (
    DecodedHypothesis,
    DecodingResult,
    FragmentDecoder,
)

T = TypeVar("T")

DRAFT_NAMESPACE = HYPOTHESIS_NAMESPACE_ROOT + "draft#"
"""
The namespace an answer is written in while the reasoner checks it.
"""

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


@dataclass
class UnsupportedFixedValue(LLMReasonerError):
    """
    The query fixes a field to a nested match, which cannot be stated to the model as
    a value.
    """

    field_name: str
    """
    The field the query fixes.
    """

    def __post_init__(self):
        super().__init__(
            f"field '{self.field_name}' is fixed to a nested match; fix it to a value "
            "or leave it open"
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


@dataclass
class Assessment:
    """
    The judgement of one answer: admitted with its warnings, or rejected for its
    problems.
    """

    problems: list[str] = field(default_factory=list)
    """
    What is wrong with the answer, as the model is told.
    """

    hypothesis: Optional[DecodedHypothesis] = None
    """
    The hypothesis, when the answer was admitted.
    """

    warnings: list[Violation] = field(default_factory=list)
    """
    The violations of warning severity, when the answer was admitted.
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
    The action to describe, in plain words, such as "pour milk from the carton into
    the bowl"; the type being generated when empty.
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
    The terms CRAM-AICOR and DUL declare, which concepts named in their namespaces
    must be among; without them such concepts are not checked.
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
        fixed = self.fixed_values(expression)
        instruction = self.instruction or expression.type.__name__
        written_fixed = {name: written_value(value) for name, value in fixed.items()}
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
            assessment = self.assess(decoder.decode(answer), fixed)
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

    @staticmethod
    def fixed_values(expression: Match[T]) -> dict[str, Any]:
        """
        :return: The values the query fixes, by field name; fields left open are absent.
        """
        fixed = {}
        for name, value in expression.kwargs.items():
            if value is Ellipsis:
                continue
            if isinstance(value, Match):
                raise UnsupportedFixedValue(field_name=name)
            fixed[name] = value
        return fixed

    def assess(self, decoded: DecodingResult, fixed: dict[str, Any]) -> Assessment:
        """
        :return: Whether a decoded answer is admitted, with its warnings, or what is
            wrong with it.
        """
        if decoded.hypothesis is None:
            return Assessment(problems=decoded.problems + self.rule_problems(decoded))
        hypothesis = decoded.hypothesis
        problems = [
            f"the root fragment '{hypothesis.identifier_of(hypothesis.root)}' must have "
            f"{name} = {written_value(value)}"
            for name, value in fixed.items()
            if not self.agrees(attrgetter(name)(hypothesis.root), value)
        ]
        violations = self.suite.violations(hypothesis.fragments)
        problems.extend(
            described_violation(violation, hypothesis.identifier_of)
            for violation in violations
            if violation.severity == Severity.ERROR
        )
        if not problems and self.consistency is not None:
            conflict = ContradictionLocaliser(self.consistency).conflict(
                GraphWriter(namespace=DRAFT_NAMESPACE).write(hypothesis)
            )
            if conflict:
                problems.append(
                    f"[{OpenWorldCheck.CONSISTENT_WITH_ONTOLOGY}] fragments "
                    + ", ".join(
                        f"'{str(node).removeprefix(DRAFT_NAMESPACE)}'"
                        for node in conflict
                    )
                    + " together contradict the ontology, CRAM-AICOR or DUL"
                )
        if problems:
            return Assessment(problems=problems)
        return Assessment(
            hypothesis=hypothesis,
            warnings=[
                violation
                for violation in violations
                if violation.severity == Severity.WARNING
            ],
        )

    def rule_problems(self, decoded: DecodingResult) -> list[str]:
        """
        :return: The rule errors of what could be linked of an answer with problems,
            so a repair corrects both at once; none when not even the root could be
            linked.
        """
        partial = decoded.partial
        if partial is None:
            return []
        return [
            described_violation(violation, partial.identifier_of)
            for violation in self.suite.violations(partial.fragments)
            if violation.severity == Severity.ERROR
        ]

    @staticmethod
    def agrees(stated: Any, fixed: Any) -> bool:
        """
        :return: Whether the value an answer states is the value the query fixes.
        """
        if isinstance(fixed, (list, tuple)):
            return list(stated) == list(fixed)
        return stated == fixed

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
