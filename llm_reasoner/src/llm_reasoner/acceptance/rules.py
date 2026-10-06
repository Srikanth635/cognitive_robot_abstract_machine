"""
The acceptance rules a hypothesis must pass, and the violations they report.

A rule names itself with a member of an enumeration whose documentation is the reason
the rule exists, so the rationale is written once beside the name. A violation names the
rule and the fragment at fault, which is what a repair prompt needs to tell a language
model what is wrong and where.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from krrood.entity_query_language.factories import an, entity, variable
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from typing_extensions import Any, Generic, Sequence, TypeVar

from llm_reasoner.ontologies.terms import DescriptionEntity

T = TypeVar("T", bound=DescriptionEntity)
"""
The type of fragment a rule is about.
"""


class Severity(StrEnum):
    """
    How a violation bears on admitting the hypothesis.
    """

    ERROR = "error"
    """
    The hypothesis is not admitted until the violation is repaired.
    """

    WARNING = "warning"
    """
    The hypothesis is admitted, and the violation is recorded with it.
    """


@dataclass(frozen=True)
class Violation:
    """
    One fragment failing one acceptance rule.
    """

    rule: StrEnum
    """
    The rule the fragment fails.
    """

    fragment: Any
    """
    The fragment at fault.
    """

    reason: str
    """
    What the fragment fails to meet, in the ontology's vocabulary.
    """

    severity: Severity = Severity.ERROR
    """
    How the violation bears on admitting the hypothesis.
    """

    def __str__(self) -> str:
        label = self.fragment.label or type(self.fragment).__name__
        return f"[{self.rule}] {label}: {self.reason}"


class AcceptanceRule(ABC):
    """
    A check every admitted hypothesis passes.
    """

    @property
    @abstractmethod
    def name(self) -> StrEnum:
        """
        The rule's name, whose documentation states why the rule exists.
        """

    @property
    def severity(self) -> Severity:
        """
        How a violation of the rule bears on admitting the hypothesis.
        """
        return Severity.ERROR

    @abstractmethod
    def violations(self, fragments: Sequence[DescriptionEntity]) -> list[Violation]:
        """
        :return: Every violation of the rule among one hypothesis's fragments.
        """


class TypedRule(AcceptanceRule, Generic[T], SubClassSafeGeneric, ABC):
    """
    A rule about the fragments of one type.
    """

    @property
    def subject_type(self) -> type[T]:
        """
        The type of fragment the rule is about.
        """
        return self.get_generic_type_parameters()[0]

    @property
    @abstractmethod
    def reason(self) -> str:
        """
        What a violating fragment fails to meet.
        """

    def subjects(self, fragments: Sequence[DescriptionEntity]) -> list[T]:
        """
        :return: The fragments the rule is about.
        """
        return [
            fragment
            for fragment in fragments
            if isinstance(fragment, self.subject_type)
        ]

    def violation(self, fragment: T) -> Violation:
        """
        :return: The violation of the rule by a fragment.
        """
        return Violation(
            rule=self.name,
            fragment=fragment,
            reason=self.reason,
            severity=self.severity,
        )


class FragmentRule(TypedRule[T], ABC):
    """
    A rule whose violators an EQL query finds among the fragments of one type.
    """

    @abstractmethod
    def violating(self, subject: T) -> Any:
        """
        :param subject: The EQL variable ranging over the rule's fragments.
        :return: The EQL condition a fragment meets when it violates the rule.
        """

    def matching(self, condition: Any, candidates: Sequence[T]) -> set[int]:
        """
        :return: The identities of the candidates an EQL condition holds for.
        """
        if not candidates:
            return set()
        subject = variable(self.subject_type, domain=list(candidates))
        return {
            id(fragment)
            for fragment in an(entity(subject).where(condition(subject))).evaluate()
        }

    def violations(self, fragments: Sequence[DescriptionEntity]) -> list[Violation]:
        subjects = self.subjects(fragments)
        found = self.matching(self.violating, subjects)
        return [
            self.violation(fragment) for fragment in subjects if id(fragment) in found
        ]


class RequirementRule(FragmentRule[T], ABC):
    """
    A rule requiring something of the fragments it applies to. The fragments meeting the
    requirement are found by an EQL query and the violators are the remaining ones, so a
    fragment with an empty collection counts as failing rather than being skipped.
    """

    @abstractmethod
    def applies(self, subject: T) -> Any:
        """
        :param subject: The EQL variable ranging over the rule's fragments.
        :return: The EQL condition a fragment meets when the requirement applies to it.
        """

    @abstractmethod
    def meets(self, subject: T) -> Any:
        """
        :param subject: The EQL variable ranging over the rule's fragments.
        :return: The EQL condition a fragment meets when it fulfils the requirement.
        """

    def violating(self, subject: T) -> Any:
        return self.applies(subject)

    def violations(self, fragments: Sequence[DescriptionEntity]) -> list[Violation]:
        subjects = self.subjects(fragments)
        applying = self.matching(self.applies, subjects)
        candidates = [fragment for fragment in subjects if id(fragment) in applying]
        meeting = self.matching(self.meets, candidates)
        return [
            self.violation(fragment)
            for fragment in candidates
            if id(fragment) not in meeting
        ]


class StructureRule(TypedRule[T], ABC):
    """
    A rule comparing the parts of one fragment with one another, such as the shape of a
    structure's phases as a graph, which no single query over fragments expresses.
    """

    @abstractmethod
    def is_violated_by(self, fragment: T) -> bool:
        """
        :return: Whether the fragment violates the rule.
        """

    def violations(self, fragments: Sequence[DescriptionEntity]) -> list[Violation]:
        return [
            self.violation(fragment)
            for fragment in self.subjects(fragments)
            if self.is_violated_by(fragment)
        ]


@dataclass
class AcceptanceSuite:
    """
    The acceptance rules of a microworld, checked together.
    """

    rules: Sequence[AcceptanceRule]
    """
    The rules.
    """

    def violations(self, fragments: Sequence[DescriptionEntity]) -> list[Violation]:
        """
        :return: Every violation of every rule among one hypothesis's fragments.
        """
        return [
            violation for rule in self.rules for violation in rule.violations(fragments)
        ]
