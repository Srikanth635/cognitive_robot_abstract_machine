"""
Tests for where Tracy's camera, on a pole above its table, looks.
"""

from __future__ import annotations

from semantic_digital_twin.reasoning.robot_predicates import VisibleTo
from semantic_digital_twin.robots.tracy import Tracy

# %% the camera above the table


def test_tracys_camera_sees_the_table_below_it(tracy_world):
    [tracy] = tracy_world.get_semantic_annotations_by_type(Tracy)
    table = tracy_world.get_body_by_name("table")
    assert VisibleTo(entity=table, camera=tracy.get_default_camera())()
