"""
Tests that the slots of a pouring that cramera_resolution names are fields of coraplex's
``PouringAction``, so a rename there shows up here.
"""

from __future__ import annotations

from dataclasses import fields

from coraplex.robot_plans.actions.composite.tool_based import PouringAction

from cramera_resolution.resolution.pouring_slot import PouringSlot


class TestPouringSlot:
    def test_every_slot_is_a_field_of_the_pouring_action(self):
        assert {slot.value for slot in PouringSlot} <= {
            field.name for field in fields(PouringAction)
        }

    def test_the_container_slots_are_the_source_and_the_target(self):
        assert PouringSlot.containers() == (
            PouringSlot.SOURCE_CONTAINER,
            PouringSlot.TARGET_CONTAINER,
        )
