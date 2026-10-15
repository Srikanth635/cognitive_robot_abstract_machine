"""
Asking a model for the vocabulary of a perspective, and returning an answer that breaks
a rule to it for repair, until an answer follows every rule or the repair budget is
spent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from typing_extensions import List

from llm_reasoner.clients.base import Exchange, LanguageModelClient
from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge.vocabulary_proposal.checks import Problem, ProposalChecker
from llm_reasoner.knowledge.vocabulary_proposal.perspective import Perspective
from llm_reasoner.knowledge.vocabulary_proposal.prompts import (
    PROPOSAL_SCHEMA,
    ProposalPrompt,
)
from llm_reasoner.knowledge.vocabulary_proposal.proposal import VocabularyProposal
from llm_reasoner.knowledge.vocabulary_proposal.writing import ProposalWriter


@dataclass
class ProposalRejected(LLMReasonerError):
    """
    No answer followed every rule within the repair budget.
    """

    problems: List[Problem]
    """
    The rules the last answer broke.
    """

    exchanges: List[Exchange]
    """
    Every exchange, in order.
    """

    def __post_init__(self):
        super().__init__(
            f"no proposal followed every rule after {len(self.exchanges)} answers; the "
            f"last broke {len(self.problems)}, the first being: "
            + (self.problems[0].message if self.problems else "none")
        )


@dataclass(frozen=True)
class ProposalOutcome:
    """
    A proposed vocabulary that follows every rule, as a module to review.
    """

    proposal: VocabularyProposal
    """
    The vocabulary.
    """

    turtle: str
    """
    The vocabulary as the module of its perspective.
    """

    exchanges: List[Exchange]
    """
    Every exchange that led to it, repairs included.
    """


@dataclass
class VocabularyProposer:
    """
    Asks a model for the vocabulary of a perspective and has it repair what breaks a
    rule.
    """

    client: LanguageModelClient
    """
    The model asked.
    """

    prompt: ProposalPrompt
    """
    Writes what the model is asked.
    """

    checker: ProposalChecker
    """
    Checks an answer against the rules.
    """

    max_repairs: int = 2
    """
    How many times an answer breaking a rule is returned for repair.
    """

    def propose(
        self, perspective: Perspective, action: str, frame: str
    ) -> ProposalOutcome:
        """
        :param perspective: The perspective whose vocabulary is asked for.
        :param action: The CRAM-AICOR concept of the action, such as ``aicor:Pouring``.
        :param frame: The core's frame of the action, such as ``core:PouringFrame``.
        :return: The first answer that follows every rule, as a module.
        :raises ProposalRejected: When no answer does within the repair budget.
        """
        system = self.prompt.system()
        request = self.prompt.request(perspective, action, frame)
        user = request
        exchanges: List[Exchange] = []
        problems: List[Problem] = []
        for attempt in range(self.max_repairs + 1):
            answer = self.client.complete_structured(system, user, PROPOSAL_SCHEMA)
            exchanges.append(
                Exchange(
                    system=system,
                    user=user,
                    raw_response=answer,
                    client_identity=self.client.identity,
                    attempt=attempt,
                )
            )
            proposal = VocabularyProposal.from_json(json.loads(answer))
            problems = self.checker.problems(proposal)
            if not problems:
                return ProposalOutcome(
                    proposal=proposal,
                    turtle=ProposalWriter(
                        proposal=proposal, perspective=perspective
                    ).turtle(),
                    exchanges=exchanges,
                )
            user = self.prompt.repair(request, answer, problems)
        raise ProposalRejected(problems=problems, exchanges=exchanges)
