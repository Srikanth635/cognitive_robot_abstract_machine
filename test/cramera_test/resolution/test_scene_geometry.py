"""
Tests of the scene's geometry as the resolver reads it: every annotated object's origin,
bounding box and top in the world frame, and where the robot's base stands, stated as a
fact of the situation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from llm_reasoner.resolver.prompt_format import PromptSection
from semantic_digital_twin.robots.robot_parts import AbstractRobot, AbstractRobotPart
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl, Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

from cramera.resolution.resolver.scene_geometry import (
    COORDINATE_DIGITS,
    GEOMETRY_HEADING,
    GeometryKey,
    OpeningDirection,
    SceneGeometry,
)
from cramera.resolution.resolver.scene_world import SceneKey, SceneWorldBuilder

from .conftest import OBJECT_MESHES
from .test_scene_world import recorded_start

TOLERANCE = 10.0**-COORDINATE_DIGITS
"""
How far a stated coordinate may lie from the exact one: its rounding.
"""

# %% helpers

OPEN_MESH = "apartment_bowl.stl"
"""
A mesh of an object that has an open side.
"""

CLOSED_MESH = "bread.stl"
"""
A mesh of an object with no open side.
"""


def add_object(world, mesh: str, kind: type) -> None:
    """
    Add the mesh to the world as an object of the kind.
    """
    world.merge_world_at_pose(
        STLParser(str(OBJECT_MESHES / mesh)).parse(),
        HomogeneousTransformationMatrix.from_xyz_rpy(
            0.0, 0.0, 0.0, reference_frame=world.root
        ),
    )
    with world.modify_world():
        world.add_semantic_annotations([kind(root=world.get_body_by_name(mesh))])


def described(bundle: Path) -> dict:
    """
    :return: The geometry of the bundle's scene, as the resolver reads it.
    """
    return json.loads(
        SceneGeometry(world=SceneWorldBuilder().build(bundle)).as_json_text()
    )


def object_named(geometry: dict, name: str) -> dict:
    """
    :return: The geometry of the object whose body has the given name.
    """
    (entry,) = [
        entry
        for entry in geometry[GeometryKey.OBJECTS]
        if entry[GeometryKey.NAME] == name
    ]
    return entry


def spawned(bundle: Path, identifier: str) -> list:
    """
    :return: Where the scene spawns the object of the given identifier.
    """
    scene = json.loads((bundle / "scene.json").read_text())
    (entry,) = [
        entry
        for entry in scene[SceneKey.OBJECTS]
        if entry[SceneKey.IDENTIFIER] == identifier
    ]
    return entry[SceneKey.SPAWN][:3]


# %% the objects


class TestObjects:
    def test_every_annotated_object_is_described_by_its_kind(self, pouring_bundle):
        world = SceneWorldBuilder().build(pouring_bundle)
        geometry = json.loads(SceneGeometry(world=world).as_json_text())
        assert sorted(
            (entry[GeometryKey.KIND], entry[GeometryKey.NAME])
            for entry in geometry[GeometryKey.OBJECTS]
        ) == sorted(
            (type(annotation).__name__, annotation.root.name.name)
            for annotation in world.get_semantic_annotations_by_type(HasRootBody)
        )

    def test_an_object_stands_at_its_origin(self, pouring_bundle):
        pitcher = object_named(described(pouring_bundle), "Static_MilkPitcher.stl")
        assert np.allclose(
            pitcher[GeometryKey.ORIGIN],
            spawned(pouring_bundle, "Static_MilkPitcher"),
            atol=TOLERANCE,
        )

    def test_the_top_is_the_centre_of_the_bounding_box_top(self, pouring_bundle):
        bowl = object_named(described(pouring_bundle), "bowl.stl")
        lowest = np.array(bowl[GeometryKey.BOUNDING_BOX][GeometryKey.MINIMUM])
        highest = np.array(bowl[GeometryKey.BOUNDING_BOX][GeometryKey.MAXIMUM])
        assert np.allclose(
            bowl[GeometryKey.TOP_CENTRE],
            [*(0.5 * (lowest[:2] + highest[:2])), highest[2]],
            atol=TOLERANCE,
        )

    def test_an_upright_object_states_that_its_up_points_up(self, pouring_kitchen):
        geometry = json.loads(SceneGeometry(world=pouring_kitchen).as_json_text())
        assert np.allclose(
            object_named(geometry, "bowl.stl")[GeometryKey.UP],
            [0.0, 0.0, 1.0],
            atol=TOLERANCE,
        )

    def test_an_overturned_object_states_that_its_up_points_down(self, pouring_kitchen):
        bowl = pouring_kitchen.get_body_by_name("bowl.stl")
        with pouring_kitchen.modify_world():
            bowl.parent_connection.origin = (
                HomogeneousTransformationMatrix.from_xyz_rpy(
                    *bowl.global_pose.position.to_np()[:3],
                    roll=np.pi,
                    reference_frame=pouring_kitchen.root,
                )
            )
        geometry = json.loads(SceneGeometry(world=pouring_kitchen).as_json_text())
        assert np.allclose(
            object_named(geometry, "bowl.stl")[GeometryKey.UP],
            [0.0, 0.0, -1.0],
            atol=TOLERANCE,
        )

    def test_the_opening_of_an_upright_container_faces_up(self, pouring_kitchen):
        geometry = json.loads(SceneGeometry(world=pouring_kitchen).as_json_text())
        assert (
            object_named(geometry, "bowl.stl")[GeometryKey.OPENING_FACES]
            == OpeningDirection.UP
        )

    @pytest.mark.parametrize(
        ("roll", "facing"),
        [
            (np.pi, OpeningDirection.DOWN),
            (np.pi / 2, OpeningDirection.SIDEWAYS),
        ],
    )
    def test_the_opening_of_a_turned_container_faces_where_it_was_turned(
        self, pouring_kitchen, roll, facing
    ):
        bowl = pouring_kitchen.get_body_by_name("bowl.stl")
        with pouring_kitchen.modify_world():
            bowl.parent_connection.origin = (
                HomogeneousTransformationMatrix.from_xyz_rpy(
                    *bowl.global_pose.position.to_np()[:3],
                    roll=roll,
                    reference_frame=pouring_kitchen.root,
                )
            )
        geometry = json.loads(SceneGeometry(world=pouring_kitchen).as_json_text())
        assert object_named(geometry, "bowl.stl")[GeometryKey.OPENING_FACES] == facing

    def test_the_opening_of_the_pitcher_faces_up(self, pouring_kitchen):
        geometry = json.loads(SceneGeometry(world=pouring_kitchen).as_json_text())
        assert (
            object_named(geometry, "Static_MilkPitcher.stl")[GeometryKey.OPENING_FACES]
            == OpeningDirection.UP
        )

    def test_an_object_of_a_kind_that_is_no_container_states_no_opening(
        self, pouring_kitchen
    ):
        milk = pouring_kitchen.get_semantic_annotations_by_type(Milk)
        assert milk
        geometry = json.loads(SceneGeometry(world=pouring_kitchen).as_json_text())
        assert GeometryKey.OPENING_FACES not in object_named(
            geometry, milk[0].root.name.name
        )

    def test_the_kind_decides_not_the_mesh(self, pouring_kitchen):
        add_object(pouring_kitchen, OPEN_MESH, Milk)
        geometry = json.loads(SceneGeometry(world=pouring_kitchen).as_json_text())
        assert GeometryKey.OPENING_FACES not in object_named(geometry, OPEN_MESH)

    def test_a_container_whose_mesh_has_no_open_side_is_stated_closed(
        self, pouring_kitchen
    ):
        add_object(pouring_kitchen, CLOSED_MESH, Bowl)
        geometry = json.loads(SceneGeometry(world=pouring_kitchen).as_json_text())
        assert (
            object_named(geometry, CLOSED_MESH)[GeometryKey.OPENING_FACES]
            == OpeningDirection.CLOSED
        )

    def test_the_robot_and_its_parts_are_not_objects(self, pouring_kitchen):
        robot_bodies = {
            annotation.root.name.name
            for kind in (AbstractRobot, AbstractRobotPart)
            for annotation in pouring_kitchen.get_semantic_annotations_by_type(kind)
        }
        geometry = json.loads(SceneGeometry(world=pouring_kitchen).as_json_text())
        described = {entry[GeometryKey.NAME] for entry in geometry[GeometryKey.OBJECTS]}
        assert robot_bodies
        assert not described & robot_bodies


# %% the robot


class TestRobotBase:
    def test_the_robot_base_stands_where_the_recording_starts(self, pouring_bundle):
        (base,) = described(pouring_bundle)[GeometryKey.ROBOT_BASES]
        assert np.allclose(
            base[GeometryKey.POSITION],
            recorded_start(pouring_bundle)[:2],
            atol=TOLERANCE,
        )


# %% in the prompt


class TestSection:
    def test_the_geometry_is_a_fact_of_the_situation(self, pouring_bundle):
        geometry = SceneGeometry(world=SceneWorldBuilder().build(pouring_bundle))
        assert geometry.section is PromptSection.SITUATION

    def test_the_geometry_is_stated_with_its_frame_and_units(self, pouring_bundle):
        geometry = SceneGeometry(world=SceneWorldBuilder().build(pouring_bundle))
        text = geometry.context_for(None)
        assert text.startswith(GEOMETRY_HEADING)
        assert text.endswith(geometry.as_json_text())
