from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest
from llm_reasoner.knowledge_updated.ontological.documentation import (
    DocumentationReader,
)
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.ontology import Ontology
from llm_reasoner.knowledge_updated.ontological.paths import (
    FieldPathResolver,
    PathReference,
    PathResolution,
)
from llm_reasoner.knowledge_updated.ontological.properties import attribute, relation

# %% fixtures

NAMESPACE = "http://example.org/paths#"
UNKNOWN_FIELD = "no_such_field"
"""
A field that no fixture class declares.
"""


@dataclass(eq=False)
class Threshold(KnowledgeEntity):
    """
    A boundary value.
    """

    level: Optional[float] = attribute(functional=True)
    """
    Where the boundary lies.
    """


@dataclass(eq=False)
class Event(KnowledgeEntity):
    """
    Something that happens.
    """

    is_failure: bool = attribute(functional=True, default=False)
    """
    Whether it marks failure.
    """


@dataclass(eq=False)
class CrossingEvent(Event):
    """
    An event that occurs when a threshold is crossed.
    """

    crosses: Optional[Threshold] = relation(functional=True)
    """
    The threshold crossed.
    """


@dataclass(eq=False)
class Phase(KnowledgeEntity):
    """
    A stretch of an action.
    """

    ends_with: list[Event] = relation()
    """
    The events that end it.
    """

    next_phase: Optional[Phase] = relation(functional=True)
    """
    The phase that follows.
    """


def resolver() -> FieldPathResolver:
    """
    Return a resolver over an ontology declaring the fixture classes.
    """
    ontology = Ontology(name="paths")
    module = ontology.module("core", version="1.0.0", namespace=NAMESPACE)
    for cls in (Threshold, Event, CrossingEvent, Phase):
        module.concept()(cls)
    return FieldPathResolver(ontology)


# %% following paths


class TestFollowingPaths:
    """
    A path is followed through the declared classes, one field at a time.
    """

    def test_single_field_ends_on_its_range(self) -> None:
        resolution = resolver().resolve((Phase,), "ends_with")

        assert resolution.types == (Event,)
        assert resolution.unknown_step is None

    def test_path_of_several_fields_ends_on_the_last_range(self) -> None:
        resolution = resolver().resolve((Phase,), "next_phase.ends_with")

        assert resolution.types == (Event,)

    def test_field_declared_only_by_a_subtype_of_the_range_is_found(self) -> None:
        resolution = resolver().resolve((Phase,), "ends_with.crosses")

        assert resolution.types == (Threshold,)

    def test_path_may_end_on_a_plain_value(self) -> None:
        resolution = resolver().resolve((Phase,), "ends_with.is_failure")

        assert resolution.types == (bool,)

    def test_path_can_start_from_several_types_at_once(self) -> None:
        resolution = resolver().resolve((Event, Threshold), "level")

        assert resolution.types == (float,)


# %% unknown steps


class TestUnknownSteps:
    """
    A step no candidate type declares is reported, with the step named.
    """

    def test_unknown_first_step(self) -> None:
        resolution = resolver().resolve((Phase,), UNKNOWN_FIELD)

        assert resolution == PathResolution(unknown_step=UNKNOWN_FIELD)

    def test_unknown_later_step(self) -> None:
        resolution = resolver().resolve((Phase,), f"ends_with.{UNKNOWN_FIELD}")

        assert resolution.unknown_step == UNKNOWN_FIELD

    def test_step_taken_from_a_plain_value_is_unknown(self) -> None:
        resolution = resolver().resolve((Phase,), "ends_with.is_failure.something")

        assert resolution.unknown_step == "something"


# %% path references


class TestPathReference:
    """
    A reference says where a path starts.
    """

    def test_reference_starts_at_the_subject_unless_told_otherwise(self) -> None:
        assert PathReference("ends_with").relative_to == ""

    def test_reference_can_start_from_the_members_of_a_collection(self) -> None:
        assert PathReference("is_failure", relative_to="ends_with").relative_to == (
            "ends_with"
        )


# %% documentation


class TestPathVocabularyIsDocumented:
    """
    Every field of the path vocabulary carries documentation.
    """

    @pytest.mark.parametrize("dataclass_type", [PathReference, PathResolution])
    def test_every_field_is_documented(self, dataclass_type: type) -> None:
        assert DocumentationReader().find_undocumented_fields(dataclass_type) == ()
