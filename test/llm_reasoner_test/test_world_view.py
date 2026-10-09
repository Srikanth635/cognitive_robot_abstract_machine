"""
Tests of the world injector's questions about the live world: which kinds exist, which
individuals, what they are, what holds them, what their fields reach, and what the world
model's own predicates say about them.
"""

from __future__ import annotations

import gc
from dataclasses import fields

import pytest

from krrood.entity_query_language.predicate import HasType, Symbol
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression
from krrood.entity_query_language.verbalization.rendering.source_documentation import (
    _attribute_docstrings,
)

from llm_reasoner.world.classes import DocumentedClass
from llm_reasoner.world.describer import WalkLimits
from llm_reasoner.world.exceptions import (
    AmbiguousKind,
    ExpiredHandle,
    MissingArgument,
    UnknownField,
    UnknownHandle,
    UnknownKind,
)
from llm_reasoner.world.finding import Comparison, Condition
from llm_reasoner.world.kinds import KindListing
from llm_reasoner.world.predicates import IndividualArgument, PlainArgument
from llm_reasoner.world.values import (
    IndividualReference,
    PlainValue,
    WorldStateDependentValue,
)
from llm_reasoner.world.holders import Holder
from llm_reasoner.world.view import WorldView

from . import world_mimics_sharing_a_name
from .world_mimics import (
    BodyAnnotation,
    InertialProperties,
    IsHeavierThan,
    SymbolFillingFieldsLater,
    UndocumentedAnnotation,
    VesselAnnotation,
    WorldBody,
    body_moved_by_a_readable_joint,
    body_with_geometry,
)

# %% helpers


@pytest.fixture
def view() -> WorldView:
    return WorldView()


def identifier_of(kind: type) -> str:
    return f"{kind.__module__}.{kind.__qualname__}"


# %% kinds


class TestKinds:
    def test_a_kind_is_listed_with_its_summary_and_live_individuals(self, view):
        bodies = [WorldBody(name="first"), WorldBody(name="second")]
        (summary,) = [
            kind for kind in view.kinds() if kind.identifier == identifier_of(WorldBody)
        ]
        assert summary.summary == WorldBody.__doc__
        assert summary.individual_count == len(bodies)

    def test_kinds_within_a_kind_are_the_kind_and_its_specialisations(self, view):
        names = {kind.name for kind in view.kinds(within=identifier_of(BodyAnnotation))}
        assert names == {
            BodyAnnotation.__name__,
            VesselAnnotation.__name__,
            UndocumentedAnnotation.__name__,
        }

    def test_present_kinds_are_those_with_live_individuals(self, view):
        annotation = VesselAnnotation(body=body_with_geometry())
        present = view.kinds(
            within=identifier_of(BodyAnnotation), listing=KindListing.PRESENT
        )
        assert {kind.name for kind in present} == {
            BodyAnnotation.__name__,
            type(annotation).__name__,
        }

    def test_instantiated_kinds_are_those_with_individuals_of_their_own(self, view):
        annotation = VesselAnnotation(body=body_with_geometry())
        (instantiated,) = view.kinds(
            within=identifier_of(BodyAnnotation), listing=KindListing.INSTANTIATED
        )
        assert instantiated.name == type(annotation).__name__
        assert instantiated.own_individual_count == 1

    def test_the_classes_of_krrood_itself_are_not_kinds_of_the_world(self, view):
        listed = {kind.identifier for kind in view.kinds()}
        predicates = {predicate.identifier for predicate in view.predicates()}
        assert identifier_of(Symbol) not in listed
        assert identifier_of(HasType) not in predicates
        assert identifier_of(IsHeavierThan) in predicates

    def test_a_kind_is_described_by_its_fields(self, view):
        description = view.describe_kind(identifier_of(InertialProperties))
        (mass,) = [entry for entry in description.fields if entry.name == "mass"]
        (declared,) = [
            entry for entry in fields(InertialProperties) if entry.name == "mass"
        ]
        assert mass.meaning == _attribute_docstrings(InertialProperties)["mass"]
        assert mass.type_name == declared.type
        assert mass.default == repr(declared.default)

    def test_a_kind_is_found_by_its_plain_name_when_that_is_unique(self, view):
        assert view.describe_kind(WorldBody.__name__).identifier == identifier_of(
            WorldBody
        )

    def test_a_plain_name_two_kinds_share_is_refused(self, view):
        with pytest.raises(AmbiguousKind):
            view.describe_kind(VesselAnnotation.__name__)
        assert world_mimics_sharing_a_name.VesselAnnotation is not VesselAnnotation

    def test_an_unknown_kind_is_refused(self, view):
        with pytest.raises(UnknownKind):
            view.describe_kind("NoSuchKind")


# %% handles


class TestHandles:
    def test_an_individual_keeps_its_handle(self, view):
        body = body_with_geometry()
        assert view.handles.handle_of(body) == view.handles.handle_of(body)
        assert view.handles.individual_of(view.handles.handle_of(body)) is body

    def test_the_handle_of_a_collected_individual_has_expired(self, view):
        handle = view.handles.handle_of(WorldBody(name="short-lived"))
        gc.collect()
        with pytest.raises(ExpiredHandle):
            view.handles.individual_of(handle)

    def test_a_handle_never_given_out_is_unknown(self, view):
        with pytest.raises(UnknownHandle):
            view.handles.individual_of("WorldBody#999")


# %% finding


class TestFinding:
    def test_finding_a_kind_finds_its_specialisations_too(self, view):
        vessel = VesselAnnotation(body=body_with_geometry())
        found = view.find(identifier_of(BodyAnnotation))
        assert [summary.handle for summary in found.individuals] == [
            view.handles.handle_of(vessel)
        ]

    def test_a_condition_on_a_field_path_selects_individuals(self, view):
        light, heavy = WorldBody(name="light"), body_with_geometry()
        heavy.inertial.mass = 2.0
        condition = Condition(
            path="inertial.mass", comparison=Comparison.GREATER, value=1.0
        )
        found = view.find(identifier_of(WorldBody), [condition])
        assert [summary.handle for summary in found.individuals] == [
            view.handles.handle_of(heavy)
        ]
        assert light.inertial.mass <= condition.value

    def test_a_limit_keeps_the_total_but_returns_fewer(self, view):
        bodies = [WorldBody(name=str(index)) for index in range(3)]
        found = view.find(identifier_of(WorldBody), limit=1)
        assert found.total == len(bodies)
        assert len(found.individuals) == 1

    def test_the_search_is_stated_in_english(self, view):
        condition = Condition(path="name", comparison=Comparison.EQUAL, value="jug")
        query = view.finder.query(WorldBody, [condition])
        found = view.find(identifier_of(WorldBody), [condition])
        assert found.query_in_english == verbalize_expression(query)


# %% describing


class TestDescribing:
    def test_a_description_names_its_individuals_by_their_handles(self, view):
        annotation = VesselAnnotation(body=body_with_geometry())
        description = view.describe(view.handles.handle_of(annotation))
        (body,) = [entry for entry in description.fields if entry.name == "body"]
        assert description.identifier == view.handles.handle_of(annotation)
        assert body.value.description.identifier == view.handles.handle_of(
            annotation.body
        )

    def test_an_individual_beyond_the_limits_is_left_as_a_reference_to_follow(
        self, view
    ):
        body = body_with_geometry()
        annotation = VesselAnnotation(body=body)
        description = view.describe(
            view.handles.handle_of(annotation), limits=WalkLimits(depth=0)
        )
        (held,) = [entry for entry in description.fields if entry.name == "body"]
        assert held.value == IndividualReference(
            identifier=view.handles.handle_of(body), type_name=type(body).__name__
        )

    def test_a_state_dependent_value_carries_its_current_numbers(self, view):
        joint_position = [0.25]
        body = body_moved_by_a_readable_joint(joint_position)
        position = view.follow(view.handles.handle_of(body), "position")
        assert isinstance(position, WorldStateDependentValue)
        assert position.current == tuple(body.position.evaluate().flatten().tolist())

    def test_a_field_the_class_fills_in_is_described_once_set(self, view):
        symbol = SymbolFillingFieldsLater()
        symbol.filled_later = 3
        description = view.describe(view.handles.handle_of(symbol))
        assert [entry.name for entry in description.fields] == ["filled_later"]


# %% holders and paths


class TestHoldersAndPaths:
    def test_the_holders_of_an_individual_are_given_by_handle(self, view):
        body = body_with_geometry()
        annotation = VesselAnnotation(body=body)
        assert view.holders(view.handles.handle_of(body)) == [
            Holder(
                handle=view.handles.handle_of(annotation),
                type_name=type(annotation).__name__,
                field_name="body",
                summary=DocumentedClass.of(VesselAnnotation).summary,
            )
        ]

    def test_following_a_path_reaches_a_value(self, view):
        body = body_with_geometry()
        assert view.follow(view.handles.handle_of(body), "inertial.mass") == PlainValue(
            value=body.inertial.mass, is_default=False
        )

    def test_following_a_path_to_an_individual_gives_its_handle(self, view):
        body = body_with_geometry()
        annotation = VesselAnnotation(body=body)
        assert view.follow(view.handles.handle_of(annotation), "body") == (
            IndividualReference(
                identifier=view.handles.handle_of(body), type_name=type(body).__name__
            )
        )

    def test_a_path_through_a_field_the_kind_lacks_is_refused(self, view):
        body = body_with_geometry()
        with pytest.raises(UnknownField):
            view.follow(view.handles.handle_of(body), "inertial.volume")


# %% predicates


class TestPredicates:
    def test_the_predicates_about_a_kind_are_listed_with_their_arguments(self, view):
        (definition,) = [
            predicate
            for predicate in view.predicates(about=identifier_of(WorldBody))
            if predicate.identifier == identifier_of(IsHeavierThan)
        ]
        assert [argument.name for argument in definition.arguments] == ["body", "mass"]
        assert definition.summary == IsHeavierThan.__doc__

    def test_a_predicate_is_evaluated_on_an_individual(self, view):
        body = body_with_geometry()
        threshold = body.inertial.mass / 2
        result = view.evaluate(
            identifier_of(IsHeavierThan),
            {
                "body": IndividualArgument(handle=view.handles.handle_of(body)),
                "mass": PlainArgument(value=threshold),
            },
        )
        assert result == PlainValue(value=True, is_default=False)

    def test_a_predicate_missing_an_argument_is_refused(self, view):
        body = body_with_geometry()
        with pytest.raises(MissingArgument):
            view.evaluate(
                identifier_of(IsHeavierThan),
                {"body": IndividualArgument(handle=view.handles.handle_of(body))},
            )
