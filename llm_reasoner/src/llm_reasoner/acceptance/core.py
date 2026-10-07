"""
Acceptance rules of the core module still written in Python: checks on comparisons
that are about to move into the ontology, and the closure of a structure's components.
"""

from __future__ import annotations

from dataclasses import fields
from enum import StrEnum
from operator import attrgetter

from krrood.entity_query_language.factories import (
    and_,
    flat_variable,
    or_,
)
from typing_extensions import Any, Sequence

from llm_reasoner.acceptance.rules import (
    FragmentRule,
    RequirementRule,
    TypedRule,
    Violation,
)
from llm_reasoner.acceptance.vocabulary import Vocabulary
from llm_reasoner.ontologies.predictive_manipulation import core
from llm_reasoner.ontologies.terms import DescriptionEntity, PropertyTerm

# %% rule names


class CoreRule(StrEnum):
    """
    The acceptance rules of the core module.
    """

    MATCH_HAS_NO_RESPONSE = "match_has_no_response"
    """
    Corrections and model updates answer mismatches; a match that invokes them claims
    both outcomes at once.
    """

    ABSENT_EVENT_HAS_NO_OBSERVATION = "absent_event_has_no_observation"
    """
    An error classifying a predicted event as absent contradicts an observation of it.
    """

    ERROR_HAS_OBSERVATION = "error_has_observation"
    """
    An error other than an absent event classifies an observation, so the comparison
    must name the observation.
    """

    COMPONENTS_DECLARED = "components_declared"
    """
    Every fragment a structure reaches that a structure can declare is declared by it,
    so the hypothesis is self-contained and every part can be bound and cited.
    """


# %% quantities


# %% transitions


# %% phases, subgoals and conditions


# %% structures


# %% comparisons


class MatchHasNoResponse(FragmentRule[core.PredictionComparison]):
    """
    A matching comparison invokes no correction and produces no model update.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.MATCH_HAS_NO_RESPONSE

    @property
    def reason(self) -> str:
        return "a match that invokes a correction or produces a model update"

    def violating(self, subject: Any) -> Any:
        return and_(
            subject.has_comparison_outcome == core.ComparisonOutcome.MATCH,
            or_(subject.invokes_correction != [], subject.produces_model_update != []),
        )


class AbsentEventHasNoObservation(FragmentRule[core.PredictionComparison]):
    """
    A comparison classifying an absent event names no observation.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.ABSENT_EVENT_HAS_NO_OBSERVATION

    @property
    def reason(self) -> str:
        return "an absent-event error alongside compares_observation"

    def violating(self, subject: Any) -> Any:
        error = flat_variable(subject.has_prediction_error)
        return and_(
            subject.compares_observation != None,
            error.has_error_kind == core.PredictionErrorKind.ABSENT_EVENT,
        )


class ErrorHasObservation(RequirementRule[core.PredictionComparison]):
    """
    A comparison classifying only errors other than an absent event names the
    observation they classify.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.ERROR_HAS_OBSERVATION

    @property
    def reason(self) -> str:
        return "an error other than an absent event without compares_observation"

    def applies(self, subject: Any) -> Any:
        error = flat_variable(subject.has_prediction_error)
        return and_(
            subject.compares_observation == None,
            error.has_error_kind != core.PredictionErrorKind.ABSENT_EVENT,
        )

    def meets(self, subject: Any) -> Any:
        error = flat_variable(subject.has_prediction_error)
        return error.has_error_kind == core.PredictionErrorKind.ABSENT_EVENT


# %% control points


# %% continuous predictions


# %% structure-wide control


# %% closure


class ComponentsDeclared(TypedRule[core.ActionPhaseStructure]):
    """
    Every fragment reachable from a structure, of a kind the structure can declare, is
    among the components the structure declares.
    """

    @property
    def name(self) -> StrEnum:
        return CoreRule.COMPONENTS_DECLARED

    @property
    def reason(self) -> str:
        return "reachable from the structure but not declared by it"

    @staticmethod
    def component_fields(
        structure: core.ActionPhaseStructure,
    ) -> dict[str, type[DescriptionEntity]]:
        """
        :return: The structure's fields declaring components, with the class each
            declares.
        """
        return Vocabulary().component_fields(type(structure))

    def declared(self, structure: core.ActionPhaseStructure) -> set[int]:
        """
        :return: The identities of the components the structure declares.
        """
        return {
            id(component)
            for name in self.component_fields(structure)
            for component in attrgetter(name)(structure)
        }

    @staticmethod
    def reachable(structure: core.ActionPhaseStructure) -> list[DescriptionEntity]:
        """
        :return: Every description fragment reachable from the structure through its
            fields, the structure excluded.
        """
        seen, reached, frontier = {id(structure)}, [], [structure]
        while frontier:
            current = frontier.pop()
            for declared in fields(current):
                if PropertyTerm.of_field(type(current), declared.name) is None:
                    continue
                value = attrgetter(declared.name)(current)
                for linked in value if isinstance(value, list) else [value]:
                    if isinstance(linked, DescriptionEntity) and id(linked) not in seen:
                        seen.add(id(linked))
                        reached.append(linked)
                        frontier.append(linked)
        return reached

    def violations(self, fragments: Sequence[DescriptionEntity]) -> list[Violation]:
        violations: list[Violation] = []
        for structure in self.subjects(fragments):
            declarable, declared = (
                tuple(self.component_fields(structure).values()),
                self.declared(structure),
            )
            violations.extend(
                self.violation(fragment)
                for fragment in self.reachable(structure)
                if isinstance(fragment, declarable) and id(fragment) not in declared
            )
        return violations
