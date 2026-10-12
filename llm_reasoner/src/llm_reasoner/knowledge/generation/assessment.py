"""
Judging one decoded answer: whether it meets the values the request fixes and the
acceptance rules in the closed world, then whether it is consistent with the ontology in
the open world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from operator import attrgetter

from typing_extensions import Any, Optional

from llm_reasoner.knowledge.acceptance.rules import AcceptanceSuite, Severity, Violation
from llm_reasoner.knowledge.acceptance.suite import predictive_manipulation_suite
from llm_reasoner.knowledge.admission.consistency import (
    ConsistencyCheck,
    ContradictionLocaliser,
    OpenWorldCheck,
)
from llm_reasoner.knowledge.generation.prompts import (
    described_violation,
    written_value,
)
from llm_reasoner.knowledge.generation.request import HypothesisRequest
from llm_reasoner.knowledge.generation.wire import DecodingResult
from llm_reasoner.knowledge.representation.hypothesis import DecodedHypothesis
from llm_reasoner.knowledge.representation.rdf import (
    HYPOTHESIS_NAMESPACE_ROOT,
    GraphWriter,
)

DRAFT_NAMESPACE = HYPOTHESIS_NAMESPACE_ROOT + "draft#"
"""
The namespace an answer is written in while the reasoner checks it.
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


@dataclass
class HypothesisAssessor:
    """
    Judges one decoded answer against the request, the acceptance rules and, when
    given, the ontology in the open world.
    """

    suite: AcceptanceSuite = field(default_factory=predictive_manipulation_suite)
    """
    The acceptance rules an answer must pass.
    """

    consistency: Optional[ConsistencyCheck] = None
    """
    The reasoner checking an answer against the ontology in the open world, once it
    passes the acceptance rules; no open-world check when absent.
    """

    def assess(self, decoded: DecodingResult, request: HypothesisRequest) -> Assessment:
        """
        :return: Whether a decoded answer is admitted, with its warnings, or what is
            wrong with it.
        """
        fixed = request.fixed_values
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
