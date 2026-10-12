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

from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from typing_extensions import Any, Generic, Sequence, TypeVar

from llm_reasoner.knowledge.ontologies.terms import DescriptionEntity

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

    rule: str
    """
    The name of the rule the fragment fails: a rule's enumeration member, or the name
    of the shape stating it.
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

    involved: tuple[Any, ...] = ()
    """
    The other fragments the violation concerns, such as the events a structure leaves
    unmonitored.
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

    def violation(self, fragment: T, involved: Sequence[Any] = ()) -> Violation:
        """
        :param fragment: The fragment at fault.
        :param involved: The other fragments the violation concerns.
        :return: The violation of the rule by a fragment.
        """
        return Violation(
            rule=self.name,
            fragment=fragment,
            reason=self.reason,
            severity=self.severity,
            involved=tuple(involved),
        )


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
