"""
The acceptance rule derived from the ontology itself: every fragment meets the
restrictions of its classes, read closed-world.

It replaces the hand-declared required-content, exactly-one, fixed-value, allowed-value
and unit checks: the ontology already states them, so writing them again would only let
the two copies drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from operator import attrgetter

from krrood.class_diagrams.class_diagram import WrappedClass
from typing_extensions import Any, Sequence

from llm_reasoner.acceptance.restrictions import ClosedWorldMembership
from llm_reasoner.acceptance.rules import AcceptanceRule, Violation
from llm_reasoner.acceptance.vocabulary import Vocabulary
from llm_reasoner.knowledge.ontologies.terms import (
    ClassExpression,
    ComplementOf,
    DescriptionEntity,
    EnumeratedClass,
    IntersectionOf,
    NamedClass,
    PropertyRestriction,
    PropertyTerm,
    Quantifier,
    UnionOf,
    local_name,
    python_attribute_name,
)

# %% rule names


class DerivedRule(StrEnum):
    """
    The acceptance rules derived from the ontology.
    """

    RANGES_HOLD = "ranges_hold"
    """
    Every value a fragment holds is of the type its property's range allows. A language
    model links fragments by identifier, so nothing but this check stops it from
    linking a field to a fragment of the wrong kind.
    """

    SHAPES_HOLD = "shapes_hold"
    """
    Every fragment meets the structural rules the SHACL shapes beside the ontology
    state, such as the transitions of a structure connecting all its phases: rules
    relating several fragments, which no restriction on one class expresses.
    """

    RESTRICTIONS_HOLD = "restrictions_hold"
    """
    Every fragment meets the restrictions its classes state, including the classes its
    values classify it into, and what is absent counts as false. Under the open-world
    reading a phase without a terminating event is merely underspecified, which is
    exactly the failure of a language model that generates too little.
    """


# %% violations


@dataclass(frozen=True, kw_only=True)
class RestrictionViolation(Violation):
    """
    A fragment failing one restriction of one of its classes.
    """

    owner: type[DescriptionEntity]
    """
    The class stating the restriction.
    """

    expression: ClassExpression
    """
    The restriction the fragment fails.
    """


@dataclass(frozen=True, kw_only=True)
class RangeViolation(Violation):
    """
    A fragment holding a value of a type its field's property does not allow.
    """

    field_name: str
    """
    The field holding the value.
    """


def described(expression: ClassExpression, explains_definitions: bool = True) -> str:
    """
    :param expression: The class expression.
    :param explains_definitions: Whether a defined class is followed by its definition,
        which says what a fragment must state to belong to it. Definitions inside a
        definition are not explained again.
    :return: The class expression in words, naming properties by their field names.
    """
    explains = explains_definitions
    if isinstance(expression, NamedClass):
        named = Vocabulary().definition_named(expression.iri)
        definition = (
            named.term.definition
            if isinstance(named, type) and issubclass(named, DescriptionEntity)
            else None
        )
        if not explains or definition is None:
            return local_name(expression.iri)
        return f"{local_name(expression.iri)}, that is: {described(definition, False)}"
    if isinstance(expression, EnumeratedClass):
        return "one of " + ", ".join(
            local_name(member) for member in expression.members
        )
    if isinstance(expression, IntersectionOf):
        return " and ".join(
            f"({described(operand, explains)})" for operand in expression.operands
        )
    if isinstance(expression, UnionOf):
        return " or ".join(
            f"({described(operand, explains)})" for operand in expression.operands
        )
    if isinstance(expression, ComplementOf):
        return f"not ({described(expression.operand, explains)})"
    name = python_attribute_name(expression.property_iri)
    filler = (
        described(expression.filler, explains)
        if expression.filler is not None
        else "anything"
    )
    if expression.quantifier == Quantifier.SOME:
        return f"{name} has a {filler}"
    if expression.quantifier == Quantifier.ONLY:
        if isinstance(expression.filler, NamedClass):
            return f"every {name} is {filler}"
        return f"every {name} meets: {filler}"
    if expression.quantifier == Quantifier.VALUE:
        value = expression.value
        return f"{name} is {local_name(value) if isinstance(value, str) else value}"
    return f"{name} has {expression.quantifier} {expression.cardinality} {filler}"


# %% rule


@dataclass
class RestrictionsHold(AcceptanceRule):
    """
    Checks every fragment against the restrictions and definitions of its classes.
    """

    membership: ClosedWorldMembership = field(default_factory=ClosedWorldMembership)
    """
    Decides which fragments meet a restriction.
    """

    @property
    def name(self) -> StrEnum:
        return DerivedRule.RESTRICTIONS_HOLD

    def owners(
        self, fragments: Sequence[DescriptionEntity]
    ) -> dict[type[DescriptionEntity], list[DescriptionEntity]]:
        """
        :return: Every class whose restrictions apply, with the fragments they apply to:
            the classes a fragment instantiates and the defined classes its values
            classify it into.
        """
        owned: dict[type[DescriptionEntity], list[DescriptionEntity]] = {}
        for fragment in fragments:
            for owner in type(fragment).__mro__:
                if issubclass(owner, DescriptionEntity) and "term" in vars(owner):
                    owned.setdefault(owner, []).append(fragment)
        for defined in self.membership.vocabulary.defined_classes:
            unclassified = [
                fragment for fragment in fragments if not isinstance(fragment, defined)
            ]
            for fragment in self.membership.members(
                defined.term.definition, unclassified
            ):
                owned.setdefault(defined, []).append(fragment)
        return owned

    def violations(self, fragments: Sequence[DescriptionEntity]) -> list[Violation]:
        violations: list[Violation] = []
        for owner, owned in self.owners(fragments).items():
            expressions = list(owner.term.restrictions)
            if owner.term.definition is not None:
                expressions.append(owner.term.definition)
            for expression in expressions:
                meeting = {
                    id(member) for member in self.membership.members(expression, owned)
                }
                violations.extend(
                    RestrictionViolation(
                        rule=self.name,
                        fragment=fragment,
                        reason=f"{owner.__name__} requires: {described(expression)}",
                        owner=owner,
                        expression=expression,
                    )
                    for fragment in owned
                    if id(fragment) not in meeting
                )
        return violations


@dataclass
class RangesHold(AcceptanceRule):
    """
    Checks every value of every fragment against the type its field allows.
    """

    @property
    def name(self) -> StrEnum:
        return DerivedRule.RANGES_HOLD

    @staticmethod
    def allows(allowed: type, value: Any) -> bool:
        """
        :return: Whether a value is of the allowed type, an integer counting as a
            floating-point number.
        """
        if allowed is float and isinstance(value, int) and not isinstance(value, bool):
            return True
        return isinstance(value, allowed)

    def violations(self, fragments: Sequence[DescriptionEntity]) -> list[Violation]:
        vocabulary = Vocabulary()
        violations: list[Violation] = []
        for fragment in fragments:
            for wrapped in WrappedClass(type(fragment)).fields:
                if PropertyTerm.of_field(type(fragment), wrapped.field.name) is None:
                    continue
                value = attrgetter(wrapped.field.name)(fragment)
                values = value if isinstance(value, list) else [value]
                allowed = vocabulary.value_type_of(
                    type(fragment), wrapped.field.name, wrapped.type_endpoint
                )
                violations.extend(
                    RangeViolation(
                        rule=self.name,
                        fragment=fragment,
                        reason=f"{wrapped.field.name} holds a {type(held).__name__}, "
                        f"not a {allowed.__name__}",
                        field_name=wrapped.field.name,
                    )
                    for held in values
                    if held is not None and not self.allows(allowed, held)
                )
        return violations
