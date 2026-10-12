"""
Asking a language model for a hypothesis, and returning a rejected answer to it for
repair: each repair is a patch applied to the answer with the fewest problems so far,
until an answer is admitted or the repair budget is spent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Optional

from llm_reasoner.clients.base import Exchange, LanguageModelClient
from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge.acceptance.rules import Violation
from llm_reasoner.knowledge.generation.assessment import HypothesisAssessor
from llm_reasoner.knowledge.generation.prompts import GenerationPrompt
from llm_reasoner.knowledge.generation.request import HypothesisRequest
from llm_reasoner.knowledge.generation.schema import GenerationSchema
from llm_reasoner.knowledge.generation.wire import FragmentDecoder
from llm_reasoner.knowledge.ontologies.linked import LinkedOntologyTerms
from llm_reasoner.knowledge.representation.hypothesis import DecodedHypothesis

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


# %% dialogue


@dataclass
class GenerationDialogue:
    """
    Asks the model for a hypothesis answering a request, judging each answer and
    returning a rejected one for repair.
    """

    client: LanguageModelClient
    """
    The language model.
    """

    assessor: HypothesisAssessor
    """
    Judges each answer.
    """

    prompt: GenerationPrompt = field(default_factory=GenerationPrompt)
    """
    Builds the prompts.
    """

    max_repairs: int = 2
    """
    How many times a rejected answer is returned to the model for correction.
    """

    linked_terms: Optional[LinkedOntologyTerms] = None
    """
    The terms CRAM-AICOR and DUL declare, which concepts named in their namespaces must
    be among; without them such concepts are not checked.
    """

    exchanges: list[Exchange] = field(default_factory=list, init=False)
    """
    Every exchange of the dialogue, in order.
    """

    def generate(self, request: HypothesisRequest) -> GenerationOutcome:
        """
        :return: The admitted hypothesis, with its warnings and every exchange.
        :raises HypothesisRejected: When no answer is admitted within the repair budget.
        """
        schema = GenerationSchema.for_root(request.root_type)
        decoder = FragmentDecoder(schema, linked_terms=self.linked_terms)
        system_prompt = self.prompt.system(schema)
        initial_prompt = self.prompt.initial(
            schema, request.instruction, request.fixed_values
        )
        prompt = initial_prompt
        under_repair: Optional[AnswerUnderRepair] = None
        for attempt in range(self.max_repairs + 1):
            raw_response = self.client.complete_structured(
                system_prompt, prompt, schema.json_schema()
            )
            self.exchanges.append(
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
            assessment = self.assessor.assess(decoder.decode(answer), request)
            if assessment.hypothesis is not None:
                return GenerationOutcome(
                    hypothesis=assessment.hypothesis,
                    warnings=assessment.warnings,
                    exchanges=list(self.exchanges),
                )
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
            problems=under_repair.problems, exchanges=list(self.exchanges)
        )
