from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from typing import Any, Optional

import pytest
from llm_reasoner.knowledge_updated.ontological.acceptance import (
    AcceptanceSuite,
    AcceptanceViolation,
)
from llm_reasoner.knowledge_updated.ontological.derived import (
    DerivedChecks,
    DerivedRule,
)
from llm_reasoner.knowledge_updated.ontological.documentation import (
    DocumentationReader,
)
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.ontology import Ontology
from llm_reasoner.knowledge_updated.ontological.properties import (
    attribute,
    reference,
    relation,
)
from llm_reasoner.knowledge_updated.ontological.references import ConceptReference

# %% a small domain that exercises each derivation

NAMESPACE = "http://example.org/derived#"


class Colour(StrEnum):
    """
    The colour of a widget.
    """

    RED = "red"
    """
    The colour of blood.
    """

    GREEN = "green"
    """
    The colour of grass.
    """

    BLUE = "blue"
    """
    The colour of the sky.
    """


@dataclass(eq=False)
class Fastener(KnowledgeEntity):
    """
    Something that holds parts together.
    """


@dataclass(eq=False)
class Bolt(Fastener):
    """
    A threaded fastener.
    """


@dataclass(eq=False)
class Tag(KnowledgeEntity):
    """
    A note that belongs to whatever carries it and is never declared on its own.
    """


@dataclass(eq=False)
class Part(KnowledgeEntity):
    """
    A part of a widget.
    """

    weight: Optional[float] = attribute(functional=True, minimum_count=1)
    """
    How heavy it is.
    """


@dataclass(eq=False)
class Widget(KnowledgeEntity):
    """
    A thing built from parts.
    """

    colour: Optional[Colour] = attribute(functional=True)
    """
    Its colour.
    """

    count: Optional[int] = attribute(functional=True)
    """
    How many there are.
    """

    parts: list[Part] = relation(minimum_count=1)
    """
    The parts it is built from.
    """

    by_name: Optional[str] = attribute(
        functional=True, exclusive_group="identification"
    )
    """
    Identifies it by name.
    """

    by_number: Optional[int] = attribute(
        functional=True, exclusive_group="identification"
    )
    """
    Identifies it by number.
    """

    fastened_with: Optional[ConceptReference] = reference(
        functional=True, concept_root=Fastener
    )
    """
    The kind of fastener that holds it.
    """

    tag: Optional[Tag] = relation(functional=True)
    """
    A note it carries.
    """


@dataclass(eq=False)
class RedWidget(Widget):
    """
    A widget that is always red.
    """


@dataclass(eq=False)
class WarmWidget(Widget):
    """
    A widget that is red or green.
    """


@dataclass(eq=False)
class UndeclaredPart(Part):
    """
    A specialisation of a part that no ontology declares.
    """


@dataclass(eq=False)
class UndeclaredFastener(Fastener):
    """
    A specialisation of an abstract concept that no ontology declares.
    """


@dataclass(eq=False)
class Workshop(KnowledgeEntity):
    """
    Declares the widgets and parts of a hypothesis.
    """

    widgets: list[Widget] = relation(owns=True)
    """
    The widgets it declares.
    """

    parts: list[Part] = relation(owns=True)
    """
    The parts it declares.
    """


@cache
def new_ontology() -> Ontology:
    """
    Return the ontology declaring the domain, built once because a class can be declared
    only once per ontology.
    """
    ontology = Ontology(name="derived")
    module = ontology.module("core", version="1.0.0", namespace=NAMESPACE)
    module.value_partition()(Colour)
    module.concept(abstract=True)(Fastener)
    module.concept()(Bolt)
    module.concept()(Tag)
    module.concept()(Part)
    module.concept()(Widget)
    module.concept(fixed_values={"colour": Colour.RED})(RedWidget)
    module.concept(allowed_values={"colour": [Colour.RED, Colour.GREEN]})(WarmWidget)
    module.concept()(Workshop)
    return ontology


def evaluate(*instances: Any) -> tuple[AcceptanceViolation, ...]:
    """
    Run the derived checks over the given fragments.
    """
    ontology = new_ontology()
    suite = AcceptanceSuite(
        name="derived", ontology=ontology, rules=[], checks=[DerivedChecks(ontology)]
    )
    return suite.evaluate(instances)


def rules_fired(violations: tuple[AcceptanceViolation, ...]) -> set[DerivedRule]:
    """
    Return which derived rules a set of violations broke.
    """
    return {violation.rule for violation in violations}


def valid_widget(**overrides: Any) -> Widget:
    """
    Return a widget that satisfies every derivation, changed by the overrides.
    """
    fields = {"parts": [Part(weight=1.0)], "by_name": "gadget"}
    fields.update(overrides)
    return Widget(**fields)


# %% a consistent hypothesis


class TestConsistentHypothesis:
    """
    Fragments that follow every declaration raise nothing.
    """

    def test_valid_widget_raises_nothing(self) -> None:
        assert evaluate(valid_widget(), *valid_widget().parts) == ()

    def test_instance_of_an_undeclared_class_is_left_alone(self) -> None:
        assert evaluate(KnowledgeEntity()) == ()

    def test_undeclared_specialisation_is_checked_as_its_declared_ancestor(
        self,
    ) -> None:
        assert DerivedRule.MINIMUM_COUNT_NOT_MET in rules_fired(
            evaluate(UndeclaredPart(label="bare"))
        )

    def test_undeclared_specialisation_of_an_abstract_concept_is_concrete(self) -> None:
        assert evaluate(UndeclaredFastener()) == ()


# %% required counts


class TestRequiredCounts:
    """
    A property declared as required must hold at least that many values.
    """

    def test_missing_single_value(self) -> None:
        violations = evaluate(Part(label="bare"))

        assert rules_fired(violations) == {DerivedRule.MINIMUM_COUNT_NOT_MET}
        assert violations[0].subject == "bare"

    def test_empty_collection(self) -> None:
        violations = evaluate(valid_widget(label="empty", parts=[]))

        assert DerivedRule.MINIMUM_COUNT_NOT_MET in rules_fired(violations)

    def test_optional_property_may_be_empty(self) -> None:
        widget = valid_widget()

        assert widget.colour is None
        assert DerivedRule.MINIMUM_COUNT_NOT_MET not in rules_fired(
            evaluate(widget, *widget.parts)
        )


# %% alternatives


class TestExclusiveGroups:
    """
    Of a group of alternatives, exactly one is given.
    """

    def test_none_given(self) -> None:
        widget = valid_widget(by_name=None)

        assert DerivedRule.EXCLUSIVE_GROUP_NOT_EXACTLY_ONE in rules_fired(
            evaluate(widget, *widget.parts)
        )

    def test_two_given(self) -> None:
        widget = valid_widget(by_number=3)

        assert DerivedRule.EXCLUSIVE_GROUP_NOT_EXACTLY_ONE in rules_fired(
            evaluate(widget, *widget.parts)
        )

    def test_one_given_is_accepted(self) -> None:
        widget = valid_widget(by_name=None, by_number=3)

        assert DerivedRule.EXCLUSIVE_GROUP_NOT_EXACTLY_ONE not in rules_fired(
            evaluate(widget, *widget.parts)
        )


# %% value types


class TestValueTypes:
    """
    A value must be of the type its property is declared to hold.
    """

    def test_text_where_a_number_is_declared(self) -> None:
        violations = evaluate(Part(label="heavy", weight="heavy"))

        assert DerivedRule.VALUE_OF_WRONG_TYPE in rules_fired(violations)

    def test_number_where_an_enumeration_is_declared(self) -> None:
        widget = valid_widget(colour="red")

        assert DerivedRule.VALUE_OF_WRONG_TYPE in rules_fired(
            evaluate(widget, *widget.parts)
        )

    def test_boolean_where_an_integer_is_declared(self) -> None:
        widget = valid_widget(count=True)

        assert DerivedRule.VALUE_OF_WRONG_TYPE in rules_fired(
            evaluate(widget, *widget.parts)
        )

    def test_integer_where_a_number_is_declared(self) -> None:
        assert DerivedRule.VALUE_OF_WRONG_TYPE not in rules_fired(
            evaluate(Part(weight=2))
        )

    def test_wrong_kind_of_object_in_a_relation(self) -> None:
        widget = valid_widget(parts=[Tag()])

        assert DerivedRule.VALUE_OF_WRONG_TYPE in rules_fired(evaluate(widget))

    def test_declared_type_is_accepted(self) -> None:
        widget = valid_widget(colour=Colour.BLUE, count=2)

        assert DerivedRule.VALUE_OF_WRONG_TYPE not in rules_fired(
            evaluate(widget, *widget.parts)
        )


# %% fixed and allowed values


class TestRestrictedValues:
    """
    A concept can fix a property, or allow it only some values.
    """

    def test_fixed_value_that_is_missing(self) -> None:
        widget = RedWidget(parts=[Part(weight=1.0)], by_name="a")

        assert DerivedRule.FIXED_VALUE_NOT_HELD in rules_fired(evaluate(widget))

    def test_fixed_value_that_differs(self) -> None:
        widget = RedWidget(parts=[Part(weight=1.0)], by_name="a", colour=Colour.BLUE)

        assert DerivedRule.FIXED_VALUE_NOT_HELD in rules_fired(evaluate(widget))

    def test_fixed_value_that_is_held(self) -> None:
        widget = RedWidget(parts=[Part(weight=1.0)], by_name="a", colour=Colour.RED)

        assert DerivedRule.FIXED_VALUE_NOT_HELD not in rules_fired(evaluate(widget))

    def test_value_outside_the_allowed_values(self) -> None:
        widget = WarmWidget(parts=[Part(weight=1.0)], by_name="a", colour=Colour.BLUE)

        assert DerivedRule.VALUE_NOT_ALLOWED in rules_fired(evaluate(widget))

    def test_value_inside_the_allowed_values(self) -> None:
        widget = WarmWidget(parts=[Part(weight=1.0)], by_name="a", colour=Colour.GREEN)

        assert DerivedRule.VALUE_NOT_ALLOWED not in rules_fired(evaluate(widget))

    def test_absent_value_is_not_outside_the_allowed_values(self) -> None:
        widget = WarmWidget(parts=[Part(weight=1.0)], by_name="a")

        assert DerivedRule.VALUE_NOT_ALLOWED not in rules_fired(evaluate(widget))


# %% references into a local vocabulary


class TestReferenceRoots:
    """
    A reference closed over a concept must name a concrete descendant of it.
    """

    def reference_to(self, name: str) -> ConceptReference:
        return ConceptReference(iri=NAMESPACE + name)

    def test_concrete_descendant_is_accepted(self) -> None:
        widget = valid_widget(fastened_with=self.reference_to("Bolt"))

        assert DerivedRule.REFERENCE_OUTSIDE_CONCEPT_ROOT not in rules_fired(
            evaluate(widget, *widget.parts)
        )

    def test_abstract_root_itself_is_rejected(self) -> None:
        widget = valid_widget(fastened_with=self.reference_to("Fastener"))

        assert DerivedRule.REFERENCE_OUTSIDE_CONCEPT_ROOT in rules_fired(
            evaluate(widget, *widget.parts)
        )

    def test_unrelated_concept_is_rejected(self) -> None:
        widget = valid_widget(fastened_with=self.reference_to("Tag"))

        assert DerivedRule.REFERENCE_OUTSIDE_CONCEPT_ROOT in rules_fired(
            evaluate(widget, *widget.parts)
        )

    def test_unknown_term_is_rejected(self) -> None:
        widget = valid_widget(fastened_with=self.reference_to("Nowhere"))

        assert DerivedRule.REFERENCE_OUTSIDE_CONCEPT_ROOT in rules_fired(
            evaluate(widget, *widget.parts)
        )


# %% abstract concepts


class TestAbstractConcepts:
    """
    A concept declared abstract is never instantiated.
    """

    def test_abstract_concept_instantiated(self) -> None:
        violations = evaluate(Fastener(label="loose"))

        assert rules_fired(violations) == {DerivedRule.ABSTRACT_CONCEPT_INSTANTIATED}
        assert violations[0].subject == "loose"

    def test_concrete_specialisation_is_accepted(self) -> None:
        assert evaluate(Bolt()) == ()


# %% declaring fragments


class TestOwnershipClosure:
    """
    A structure may refer only to fragments that it declares.
    """

    def workshop(self, widget: Widget, declared_parts: list[Part]) -> Workshop:
        return Workshop(label="shop", widgets=[widget], parts=declared_parts)

    def test_reference_to_a_declared_fragment_is_accepted(self) -> None:
        part = Part(weight=1.0)
        widget = valid_widget(parts=[part])
        shop = self.workshop(widget, [part])

        assert DerivedRule.REFERENCE_NOT_DECLARED not in rules_fired(
            evaluate(shop, widget, part)
        )

    def test_reference_to_an_undeclared_fragment_is_reported_against_the_referrer(
        self,
    ) -> None:
        stray = Part(label="stray", weight=1.0)
        widget = valid_widget(label="widget", parts=[stray])
        shop = self.workshop(widget, [])

        violations = [
            v
            for v in evaluate(shop, widget, stray)
            if v.rule is DerivedRule.REFERENCE_NOT_DECLARED
        ]

        assert [v.subject for v in violations] == ["widget"]
        assert "stray" in violations[0].message
        assert "shop" in violations[0].message

    def test_fragment_no_collection_declares_may_be_nested(self) -> None:
        part = Part(weight=1.0)
        widget = valid_widget(parts=[part], tag=Tag())
        shop = self.workshop(widget, [part])

        assert DerivedRule.REFERENCE_NOT_DECLARED not in rules_fired(
            evaluate(shop, widget, part)
        )

    def test_an_undeclared_fragment_is_found_however_deep_it_is_nested(self) -> None:
        stray = Part(label="deep", weight=1.0)
        inner = valid_widget(label="inner", parts=[stray])
        outer_part = Part(weight=1.0)
        outer = valid_widget(label="outer", parts=[outer_part])
        shop = Workshop(label="shop", widgets=[outer, inner], parts=[outer_part])

        subjects = {
            v.subject
            for v in evaluate(shop, outer, inner, outer_part, stray)
            if v.rule is DerivedRule.REFERENCE_NOT_DECLARED
        }

        assert subjects == {"inner"}


# %% documentation


class TestDerivedRulesAreDocumented:
    """
    Every derived rule says why it exists.
    """

    def test_every_member_is_documented(self) -> None:
        assert DocumentationReader().find_undocumented_members(DerivedRule) == ()
