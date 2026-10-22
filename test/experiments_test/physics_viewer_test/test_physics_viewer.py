"""
Tests of the physics viewer: the scenes it puts under MuJoCo physics, and the run that
steps a scene in real time until a duration is over or the window is closed.

Every run here is headless, so nothing opens a window.
"""

from __future__ import annotations

import time
from datetime import timedelta

import mujoco
import pytest
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World

from experiments.physics_viewer.exceptions import HeadlessRunWithoutDuration
from experiments.physics_viewer.scenes import MjcfFile, PhysicsScene, UrdfFile
from experiments.physics_viewer.viewer import PhysicsViewer

# %% what the tests compile


def compiled(scene: PhysicsScene) -> mujoco.MjModel:
    """
    :return: The MuJoCo model of the scene, compiled while the files of its simulation exist.
    """
    simulation = MujocoSim(world=scene.value.build(), headless=True)
    return mujoco.MjModel.from_xml_path(str(simulation.file_path))


# %% the scenes


class TestScenes:
    @pytest.mark.parametrize("scene", [PhysicsScene.CUBE, PhysicsScene.CUPS])
    def test_an_mjcf_scene_is_the_world_its_file_describes(self, scene):
        described = MJCFParser(str(scene.value.file)).parse()
        assert len(scene.value.build().bodies) == len(described.bodies)

    def test_every_mjcf_file_of_the_scenes_exists(self):
        assert all(
            file.is_file() for file in MjcfFile if file is not MjcfFile.DIRECTORY
        )

    @pytest.mark.parametrize("scene", list(PhysicsScene))
    def test_every_scene_builds_a_world(self, scene):
        assert isinstance(scene.value.build(), World)

    @pytest.mark.parametrize("scene", [PhysicsScene.PR2, PhysicsScene.TRACY])
    def test_a_robot_scene_holds_more_than_the_floor(self, scene):
        assert len(scene.value.build().bodies) > len(
            PhysicsScene.FLOOR.value.build().bodies
        )

    def test_a_scene_is_named_on_the_command_line_by_its_lower_case_name(self):
        assert PhysicsScene.from_command_line("pr2") is PhysicsScene.PR2

    def test_a_name_no_scene_has_is_refused(self):
        with pytest.raises(KeyError):
            PhysicsScene.from_command_line("no-such-scene")


# %% furniture that does not push itself apart


class TestFurnitureAtRest:
    """
    The apartment's drawers and doors have collision shapes that overlap the cabinets
    they sit in, so a scene whose furniture collides with itself throws them open the
    moment the physics starts.
    """

    @staticmethod
    def contacts_at_the_start(scene: PhysicsScene) -> int:
        model = compiled(scene)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return data.ncon

    def test_the_apartment_has_no_contact_between_its_own_parts(self):
        assert self.contacts_at_the_start(PhysicsScene.APARTMENT) == 0

    def test_the_cube_still_touches_the_ground_it_stands_on(self):
        assert self.contacts_at_the_start(PhysicsScene.CUBE) > 0

    def test_the_robot_in_the_apartment_still_collides_with_the_furniture(self):
        model = compiled(PhysicsScene.PR2_IN_APARTMENT)
        furniture_category, default_category = 2, 1
        offered = model.geom_contype
        collidable = model.geom_conaffinity
        assert (
            (offered == furniture_category) & (collidable == default_category)
        ).any()
        assert ((offered == default_category) & (collidable == default_category)).any()


# %% robots that start parked


class TestParkedRobots:
    """
    A robot whose arms start where its joints happen to be 0, which a joint limit puts
    out of reach, has its servos pull the arms to the limit the moment the physics
    starts.
    """

    @pytest.mark.parametrize(
        "scene", [PhysicsScene.PR2, PhysicsScene.TRACY, PhysicsScene.PR2_IN_APARTMENT]
    )
    def test_the_arms_of_a_robot_start_parked(self, scene):
        world = scene.value.build()
        [robot] = world.get_semantic_annotations_by_type(AbstractRobot)
        for arm in robot.all_arms:
            parked = arm.get_joint_state_by_type(StaticJointState.PARK)
            positions = [connection.position for connection in parked.connections]
            assert positions == pytest.approx(parked.target_values)


# %% the run


class TestRun:
    def test_a_headless_run_advances_the_physics_by_the_duration(self):
        viewer = PhysicsViewer(
            scene=PhysicsScene.CUBE,
            headless=True,
            duration=timedelta(seconds=0.5),
            speed=100.0,
        )
        assert viewer.run().simulated_seconds == pytest.approx(
            0.5, abs=viewer.frame.total_seconds()
        )

    def test_the_run_takes_the_real_time_the_speed_asks_for(self):
        viewer = PhysicsViewer(
            scene=PhysicsScene.CUBE,
            headless=True,
            duration=timedelta(seconds=0.6),
            speed=2.0,
        )
        started = time.monotonic()
        viewer.run()
        assert time.monotonic() - started >= 0.6 / 2.0

    def test_a_faster_run_takes_less_real_time(self):
        def real_seconds(speed: float) -> float:
            viewer = PhysicsViewer(
                scene=PhysicsScene.CUBE,
                headless=True,
                duration=timedelta(seconds=0.6),
                speed=speed,
            )
            started = time.monotonic()
            viewer.run()
            return time.monotonic() - started

        assert real_seconds(10.0) < real_seconds(1.0)

    def test_the_run_reports_the_frames_it_stepped(self):
        viewer = PhysicsViewer(
            scene=PhysicsScene.CUBE,
            headless=True,
            duration=timedelta(seconds=0.5),
            speed=100.0,
        )
        assert viewer.run().frames == round(0.5 / viewer.frame.total_seconds())

    def test_a_run_without_a_window_needs_to_be_told_when_to_stop(self):
        with pytest.raises(HeadlessRunWithoutDuration):
            PhysicsViewer(scene=PhysicsScene.CUBE, headless=True).run()
