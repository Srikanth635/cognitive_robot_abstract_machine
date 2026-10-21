"""
Tests for reflecting the pouring designator into its slot table, with referent slots
resolved against the roles of the active hypothesis.
"""

import pytest

krrood = pytest.importorskip("krrood", reason="EQL requires krrood")

from llm_reasoner.knowledge.access import (  # noqa: E402
    pouring,
)

from cramera_resolution import hypothesis_knowledge  # noqa: E402
from cramera_resolution.designator_knowledge import reflect_pouring_action  # noqa: E402
from cramera_resolution.scenes import SceneObject  # noqa: E402

from .resolutions import objects_given  # noqa: E402


def scene_objects() -> list[SceneObject]:
    """
    Return a carton, a bowl and a cup standing on a bench.
    """
    return [
        SceneObject(name=name, label=label)
        for name, label in [
            ("carton", "Milk Carton"),
            ("bowl", "Bowl"),
            ("cup", "Cup"),
        ]
    ]


def slot_named(knowledge, name):
    (slot,) = [slot for slot in knowledge.slots if slot.name == name]
    return slot


def role_label(role_type: type) -> str:
    """
    Return the label of the active hypothesis's role of the given type.
    """
    (role,) = [
        role
        for role in hypothesis_knowledge.active().view.root.defines_role
        if type(role) is role_type
    ]
    return role.label


class TestRoles:
    def test_container_slots_name_the_role_they_fill(self):
        knowledge = reflect_pouring_action(scene_objects(), None)
        assert slot_named(knowledge, "source_container").role is pouring.SourceContainer
        assert slot_named(knowledge, "target_container").role is (
            pouring.TargetReceptacle
        )

    def test_value_slot_fills_no_role(self):
        knowledge = reflect_pouring_action(scene_objects(), None)
        assert slot_named(knowledge, "tilt_angle").role is None


class TestResolution:
    def test_nothing_is_resolved_without_a_hypothesis(self):
        knowledge = reflect_pouring_action(scene_objects(), None)
        assert slot_named(knowledge, "source_container").resolved_candidate is None
        assert knowledge.resolved_for_instruction is None

    @pytest.mark.usefixtures("sample_hypothesis_knowledge")
    def test_container_resolves_to_the_object_named_like_its_role(self):
        view = hypothesis_knowledge.active().view
        knowledge = reflect_pouring_action(scene_objects(), view)
        assert "carton" in role_label(pouring.SourceContainer)
        assert slot_named(
            knowledge, "source_container"
        ).resolved_candidate.identifier == ("carton")
        assert "bowl" in role_label(pouring.TargetReceptacle)
        assert slot_named(
            knowledge, "target_container"
        ).resolved_candidate.identifier == ("bowl")
        assert knowledge.resolved_for_instruction == view.record.instruction

    @pytest.mark.usefixtures("sample_hypothesis_knowledge")
    def test_a_scene_without_objects_resolves_nothing(self):
        knowledge = reflect_pouring_action([], hypothesis_knowledge.active().view)
        assert slot_named(knowledge, "source_container").resolved_candidate is None


class TestResolvedByTheResolver:
    def test_a_resolved_slot_holds_the_value_the_resolver_chose(self):
        resolution = objects_given()
        knowledge = reflect_pouring_action(scene_objects(), None, resolution)
        (action,) = resolution.actions
        assert {
            slot.name: slot_named(knowledge, slot.name).lisp_value
            for slot in action.slots
        } == {slot.name: slot.value for slot in action.slots}

    def test_a_resolved_slot_offers_no_illustrative_candidates(self):
        knowledge = reflect_pouring_action(scene_objects(), None, objects_given())
        assert slot_named(knowledge, "target_container").candidates == []

    def test_a_slot_the_resolution_does_not_name_stays_as_reflected(self):
        resolved = reflect_pouring_action(scene_objects(), None, objects_given())
        reflected = reflect_pouring_action(scene_objects(), None)
        assert (
            slot_named(resolved, "pour_side").lisp_value
            == slot_named(reflected, "pour_side").lisp_value
        )
