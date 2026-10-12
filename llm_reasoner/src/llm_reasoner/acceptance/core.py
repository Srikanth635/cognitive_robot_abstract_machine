"""
The closure of a structure's components: everything a structure reaches that it can
declare, it declares.
"""

from __future__ import annotations

from dataclasses import fields
from enum import StrEnum
from operator import attrgetter

from typing_extensions import Sequence

from llm_reasoner.acceptance.rules import (
    TypedRule,
    Violation,
)
from llm_reasoner.acceptance.vocabulary import Vocabulary
from llm_reasoner.knowledge.ontologies.predictive_manipulation import core
from llm_reasoner.knowledge.ontologies.terms import DescriptionEntity, PropertyTerm

# %% rule names


class CoreRule(StrEnum):
    """
    The acceptance rules of the core module.
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
