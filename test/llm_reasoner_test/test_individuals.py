"""
Tests of describing world individuals to a language model from their dataclass fields,
knowing nothing about the world model they belong to.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import fields

import pytest

from krrood.entity_query_language.factories import the, variable
from krrood.entity_query_language.verbalization.rendering.source_documentation import (
    _attribute_docstrings,
)

from llm_reasoner.individuals.describer import IndividualDescriber, WalkLimits
from llm_reasoner.individuals.description import (
    CollectionValue,
    IndividualDescription,
    NestedIndividual,
    NumericValue,
    PlainValue,
    Referrer,
    RepeatedIndividual,
    UndescribedValue,
    WorldStateDependentValue,
)
from llm_reasoner.individuals.rendering import IndividualRenderer
from llm_reasoner.resolver.backend import LanguageModelBackend
from llm_reasoner.resolver.context import StructuralCandidateRenderer
from llm_reasoner.resolver.prompts import PromptBuilder
from llm_reasoner.resolver.reflection import reflect_slots

from .individual_world import (
    ActionWithBodyAndValueSlots,
    BodyAnnotation,
    BoxShape,
    InertialProperties,
    SymbolFillingFieldsLater,
    UndocumentedAnnotation,
    VesselAnnotation,
    WorldBody,
    body_moved_by_a_joint,
    body_with_geometry,
)

WORLD_MODEL_PACKAGE = "semantic_digital_twin"
"""
A world model built on krrood that describing individuals must never load.
"""

# %% helpers


@pytest.fixture
def describer() -> IndividualDescriber:
    return IndividualDescriber()


def value_of(description: IndividualDescription, name: str):
    (described,) = [entry for entry in description.fields if entry.name == name]
    return described.value


def nested(description: IndividualDescription, name: str) -> IndividualDescription:
    value = value_of(description, name)
    assert isinstance(value, NestedIndividual)
    return value.description


# %% lineage


class TestLineage:
    def test_lineage_runs_from_the_class_through_its_ancestors(self, describer):
        description = describer.describe(VesselAnnotation(body=body_with_geometry()))
        assert [summary.name for summary in description.lineage[:2]] == [
            VesselAnnotation.__name__,
            BodyAnnotation.__name__,
        ]

    def test_a_class_is_summarised_by_its_docstring(self, describer):
        description = describer.describe(VesselAnnotation(body=body_with_geometry()))
        assert description.lineage[0].summary == VesselAnnotation.__doc__


# %% fields


class TestFields:
    def test_a_field_means_what_its_docstring_says(self, describer):
        description = describer.describe(body_with_geometry())
        (mass,) = [
            entry
            for entry in nested(description, "inertial").fields
            if entry.name == "mass"
        ]
        assert mass.meaning == _attribute_docstrings(InertialProperties)["mass"]

    def test_an_inherited_field_keeps_the_meaning_its_declaring_class_gives_it(
        self, describer
    ):
        description = describer.describe(VesselAnnotation(body=body_with_geometry()))
        (body,) = [entry for entry in description.fields if entry.name == "body"]
        assert body.meaning == _attribute_docstrings(BodyAnnotation)["body"]

    def test_a_value_equal_to_the_declared_default_is_marked_as_default(
        self, describer
    ):
        (declared,) = [
            entry for entry in fields(InertialProperties) if entry.name == "mass"
        ]
        defaulted = value_of(
            nested(describer.describe(WorldBody(name="box")), "inertial"), "mass"
        )
        assert defaulted == PlainValue(value=declared.default, is_default=True)

    def test_a_value_set_explicitly_is_not_marked_as_default(self, describer):
        body = body_with_geometry()
        set_mass = value_of(nested(describer.describe(body), "inertial"), "mass")
        assert set_mass == PlainValue(value=body.inertial.mass, is_default=False)

    def test_a_field_the_class_fills_in_itself_is_left_out(self, describer):
        unset = SymbolFillingFieldsLater()
        description = describer.describe(body_with_geometry())
        assert [entry.name for entry in describer.describe(unset).fields] == []
        assert description.referrers == []

    def test_private_unserialised_and_empty_fields_are_left_out(self, describer):
        names = [
            entry.name for entry in describer.describe(body_with_geometry()).fields
        ]
        assert names == [
            "name",
            "shapes",
            "inertial",
            "position",
            "attached_to",
            "identifier",
        ]


# %% values


class TestValues:
    def test_a_constant_symbolic_value_is_given_as_numbers(self, describer):
        body = body_with_geometry()
        assert value_of(describer.describe(body), "position") == NumericValue(
            values=tuple(body.position.to_np().flatten().tolist())
        )

    def test_a_symbolic_value_with_free_variables_depends_on_the_world_state(
        self, describer
    ):
        body = body_moved_by_a_joint()
        assert value_of(
            describer.describe(body), "position"
        ) == WorldStateDependentValue(
            variables=tuple(
                variable.name for variable in body.position.free_variables()
            )
        )

    def test_a_collection_gives_its_size_and_describes_its_first_items(self):
        body = WorldBody(name="crate", shapes=[BoxShape(width=0.1, height=0.1)] * 5)
        limits = WalkLimits(items_per_collection=2)
        shapes = value_of(IndividualDescriber(limits=limits).describe(body), "shapes")
        assert isinstance(shapes, CollectionValue)
        assert shapes.size == len(body.shapes)
        assert len(shapes.items) == limits.items_per_collection

    def test_a_value_of_no_known_form_is_named_by_its_type(self, describer):
        body = body_with_geometry()
        assert value_of(describer.describe(body), "identifier") == UndescribedValue(
            type_name=type(body.identifier).__name__
        )


# %% the walk


class TestWalk:
    def test_an_individual_reached_again_is_referred_to_by_its_identifier(
        self, describer
    ):
        first = WorldBody(name="first")
        second = WorldBody(name="second", attached_to=first)
        first.attached_to = second
        description = describer.describe(first)
        assert value_of(nested(description, "attached_to"), "attached_to") == (
            RepeatedIndividual(identifier=description.identifier)
        )

    def test_individuals_below_the_depth_limit_are_named_by_their_type(self):
        description = IndividualDescriber(limits=WalkLimits(depth=0)).describe(
            body_with_geometry()
        )
        assert value_of(description, "inertial") == UndescribedValue(
            type_name=InertialProperties.__name__
        )

    def test_the_walk_stops_describing_at_the_limit_of_individuals(self):
        description = IndividualDescriber(limits=WalkLimits(individuals=1)).describe(
            body_with_geometry()
        )
        assert value_of(description, "inertial") == UndescribedValue(
            type_name=InertialProperties.__name__
        )


# %% referrers


class TestReferrers:
    def test_an_individual_is_described_with_the_individuals_holding_it(
        self, describer
    ):
        body = body_with_geometry()
        annotation = VesselAnnotation(body=body)
        WorldBody(name="unrelated")
        assert describer.describe(body).referrers == [
            Referrer(
                type_name=type(annotation).__name__,
                field_name="body",
                summary=VesselAnnotation.__doc__,
            )
        ]


# %% rendering


class TestRendering:
    def test_a_default_value_is_shown_as_one(self, describer):
        (declared,) = [
            entry for entry in fields(InertialProperties) if entry.name == "mass"
        ]
        text = IndividualRenderer().render(describer.describe(WorldBody(name="box")))
        assert f"mass = {declared.default!r} (default)" in text

    def test_the_text_stays_within_its_line_budget(self, describer):
        description = describer.describe(VesselAnnotation(body=body_with_geometry()))
        whole = IndividualRenderer().render(description).splitlines()
        budget = 4
        shortened = (
            IndividualRenderer(line_budget=budget).render(description).splitlines()
        )
        assert shortened[: budget - 1] == whole[: budget - 1]
        assert shortened[-1] == IndividualRenderer.omission(len(whole) - (budget - 1))

    def test_the_individuals_holding_it_come_right_after_the_heading(self, describer):
        body = body_with_geometry()
        annotation = VesselAnnotation(body=body)
        description = describer.describe(body)
        (referrer,) = description.referrers
        lines = IndividualRenderer().render(description).splitlines()
        assert lines[1] == IndividualRenderer.referrer_line(referrer)
        assert referrer.type_name == type(annotation).__name__

    def test_a_class_without_its_own_docstring_borrows_no_summary(self, describer):
        description = describer.describe(
            UndocumentedAnnotation(body=body_with_geometry())
        )
        assert IndividualRenderer.heading(description) == (
            f"{description.identifier} ["
            + " < ".join(summary.name for summary in description.lineage)
            + "]"
        )

    def test_rendering_is_deterministic(self, describer):
        description = describer.describe(VesselAnnotation(body=body_with_geometry()))
        assert IndividualRenderer().render(description) == IndividualRenderer().render(
            description
        )


# %% the resolver


class TestResolverPrompt:
    def test_a_candidate_is_described_by_its_structure_under_its_token(self):
        body = body_with_geometry()
        VesselAnnotation(body=body)
        query = the(ActionWithBodyAndValueSlots)(source=variable(WorldBody))
        prompt = PromptBuilder(renderer=StructuralCandidateRenderer()).build(
            query, reflect_slots(query), [], number_of_candidates=1
        )
        first, *rest = (
            IndividualRenderer()
            .render(IndividualDescriber().describe(body))
            .splitlines()
        )
        token = PromptBuilder.candidate_prefix(0)
        expected = [token + first] + [" " * len(token) + line for line in rest]
        assert "\n".join(expected) in prompt

    def test_the_resolver_describes_candidates_by_their_structure_by_default(self):
        (renderer,) = [
            entry
            for entry in fields(LanguageModelBackend)
            if entry.name == "candidate_renderer"
        ]
        assert renderer.default_factory is StructuralCandidateRenderer


# %% independence


def test_describing_individuals_loads_no_world_model():
    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, llm_reasoner.individuals.describer, llm_reasoner.resolver; "
            f"print({WORLD_MODEL_PACKAGE!r} in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert loaded.stdout.strip() == str(False)
