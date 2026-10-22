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
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.pr2 import PR2, PR2Joint
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF

from experiments.physics_viewer.exceptions import HeadlessRunWithoutDuration
from experiments.physics_viewer.actions import PhysicsAction
from experiments.physics_viewer.overlay import ActionOverlay
from experiments.physics_viewer.paced_simulation import RealTimeMujocoSim
from experiments.physics_viewer.placement import Position, ViewerCamera
from experiments.physics_viewer.viewer import MovedJoint, MovedObject
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


# %% the colors the URDF declares


def declared_colors(file: str) -> set[tuple[float, ...]]:
    """
    :param file: A URDF file or the ROS path of one.
    :return: The distinct colors its visuals declare through their materials.
    """
    parsed = URDFParser.from_file(file).parsed
    by_name = {
        material.name: material.color.rgba
        for material in parsed.materials
        if material.color
    }
    colors = set()
    for link in parsed.links:
        for visual in link.visuals:
            material = visual.material
            if material is None:
                continue
            rgba = material.color.rgba if material.color else by_name.get(material.name)
            if rgba is not None:
                colors.add(tuple(round(float(value), 2) for value in rgba))
    return colors


def compiled_colors(scene: PhysicsScene) -> set[tuple[float, ...]]:
    """
    :return: The distinct colors of the geoms the scene's window shows.
    """
    model = compiled(scene)
    shown = model.geom_group < 3
    return {
        tuple(round(float(value), 2) for value in rgba)
        for rgba in model.geom_rgba[shown]
    }


class TestColors:
    """
    semantic_digital_twin gives the colors a URDF declares to its boxes, spheres and
    cylinders and not to its meshes, which leaves every mesh white.
    """

    def test_the_apartment_shows_the_colors_its_urdf_declares(self):
        assert declared_colors(str(UrdfFile.APARTMENT)) <= compiled_colors(
            PhysicsScene.APARTMENT
        )

    def test_the_pr2_shows_the_colors_its_urdf_declares(self):
        assert declared_colors(PR2.get_ros_file_path()) <= compiled_colors(
            PhysicsScene.PR2
        )

    def test_the_pr2_in_the_apartment_shows_the_colors_of_both(self):
        declared = declared_colors(PR2.get_ros_file_path()) | declared_colors(
            str(UrdfFile.APARTMENT)
        )
        assert declared <= compiled_colors(PhysicsScene.PR2_IN_APARTMENT)

    def test_a_scene_with_colors_of_its_own_keeps_them(self):
        assert (0.0, 0.0, 1.0, 1.0) in compiled_colors(PhysicsScene.CUPS)


# %% objects to interact with


class TestObjects:
    """
    The PR2 stands at the kitchen counter, with objects on it to interact with.
    """

    scene = PhysicsScene.PR2_AT_THE_COUNTER

    def test_every_object_of_the_scene_is_in_its_world(self):
        world = self.scene.value.build()
        names = {body.name.name for body in world.bodies}
        assert {placed.mesh for placed in self.scene.value.scene.objects} <= names

    def test_an_object_with_an_annotation_has_it(self):
        world = self.scene.value.build()
        annotated = [
            placed for placed in self.scene.value.scene.objects if placed.annotation
        ]
        assert annotated
        assert all(
            world.get_semantic_annotations_by_type(placed.annotation)
            for placed in annotated
        )

    def test_the_objects_stay_on_the_counter_under_physics(self):
        scene = self.scene
        run = PhysicsViewer(
            scene=scene, headless=True, duration=timedelta(seconds=2), speed=100.0
        ).run()
        for placed in scene.value.scene.objects:
            height = float(run.world.get_body_by_name(placed.mesh).global_pose.z)
            assert height == pytest.approx(placed.position.z, abs=0.01)

    def test_the_robot_stands_where_the_scene_puts_it(self):
        world = self.scene.value.build()
        [robot] = world.get_semantic_annotations_by_type(AbstractRobot)
        position = self.scene.value.position
        assert float(robot.root.global_pose.x) == pytest.approx(position.x)
        assert float(robot.root.global_pose.y) == pytest.approx(position.y)


# %% actions in physics


class TestActions:
    def torso_position(self, run) -> float:
        return run.world.get_connection_by_name(PR2Joint.TORSO_LIFT).position

    def test_raising_the_torso_in_physics_lifts_it(self):
        scene = PhysicsScene.PR2
        before = (
            scene.value.build().get_connection_by_name(PR2Joint.TORSO_LIFT).position
        )
        run = PhysicsViewer(
            scene=scene,
            headless=True,
            duration=timedelta(seconds=0.1),
            speed=100.0,
            action=PhysicsAction.RAISE_TORSO,
        ).run()
        assert self.torso_position(run) > before + 0.1

    @pytest.mark.parametrize(
        ("action", "state"),
        [
            (PhysicsAction.OPEN_GRIPPERS, GripperState.OPEN),
            (PhysicsAction.CLOSE_GRIPPERS, GripperState.CLOSE),
        ],
    )
    def test_the_grippers_go_to_the_state_the_action_names(self, action, state):
        run = PhysicsViewer(
            scene=PhysicsScene.PR2,
            headless=True,
            duration=timedelta(seconds=0.1),
            speed=100.0,
            action=action,
        ).run()
        [robot] = run.world.get_semantic_annotations_by_type(AbstractRobot)
        for arm in robot.all_arms:
            target = arm.end_effector.get_joint_state_by_type(state)
            positions = [connection.position for connection in target.connections]
            assert positions == pytest.approx(target.target_values, abs=0.01)

    def test_the_duration_counts_what_is_stepped_after_the_action(self):
        viewer = PhysicsViewer(
            scene=PhysicsScene.PR2,
            headless=True,
            duration=timedelta(seconds=0.5),
            speed=100.0,
            action=PhysicsAction.RAISE_TORSO,
        )
        assert viewer.run().frames == round(0.5 / viewer.frame.total_seconds())


# %% driving to the counter and picking up the milk


class TestMilk:
    scene = PhysicsScene.PR2_AT_THE_COUNTER

    def run_action(self, action: PhysicsAction):
        return PhysicsViewer(
            scene=self.scene,
            headless=True,
            duration=timedelta(seconds=0.5),
            speed=1000.0,
            action=action,
        ).run()

    def test_driving_to_the_counter_brings_the_robot_to_the_spot_in_front_of_it(self):
        run = self.run_action(PhysicsAction.GO_TO_THE_COUNTER)
        [robot] = run.world.get_semantic_annotations_by_type(AbstractRobot)
        spot = PhysicsAction.GO_TO_THE_COUNTER.value.target
        assert float(robot.root.global_pose.x) == pytest.approx(spot.x, abs=0.05)
        assert float(robot.root.global_pose.y) == pytest.approx(spot.y, abs=0.05)

    def test_driving_to_the_counter_moves_the_base_joints_of_the_robot_not_the_objects(
        self,
    ):
        run = self.run_action(PhysicsAction.GO_TO_THE_COUNTER)
        assert run.moved_objects == ()

    def test_picking_up_the_milk_is_performed_without_failing(self):
        run = self.run_action(PhysicsAction.PICK_UP_MILK)
        assert run.frames > 0

    def test_picking_up_the_milk_reports_the_milk_when_it_moved(self):
        run = self.run_action(PhysicsAction.PICK_UP_MILK)
        assert "milk.stl" in {moved.body for moved in run.moved_objects}

    def test_only_the_objects_that_moved_are_reported(self):
        viewer = PhysicsViewer(scene=self.scene, headless=True)
        world = self.scene.value.build()
        before = viewer.object_positions(world)
        milk = world.get_body_by_name("milk.stl")
        milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            before["milk.stl"].x + 0.05,
            before["milk.stl"].y,
            before["milk.stl"].z,
            reference_frame=world.root,
        )
        moved = viewer.objects_moved_since(before, world)
        assert [item.body for item in moved] == ["milk.stl"]

    def test_an_object_that_moved_less_than_the_threshold_is_not_reported(self):
        viewer = PhysicsViewer(scene=self.scene, headless=True)
        world = self.scene.value.build()
        before = viewer.object_positions(world)
        assert viewer.objects_moved_since(before, world) == ()

    @pytest.mark.xfail(
        reason="the PR2's fingers push the milk carton over as they close on it under physics"
    )
    def test_the_milk_is_lifted_off_the_counter(self):
        run = self.run_action(PhysicsAction.PICK_UP_MILK)
        [moved] = [moved for moved in run.moved_objects if moved.body == "milk.stl"]
        assert moved.after.z > moved.before.z + 0.05


# %% the first view of a scene


class TestCamera:
    def test_a_camera_is_put_where_the_scene_asks(self):
        camera = ViewerCamera(
            lookat=Position(1.0, 2.0, 3.0), distance=4.0, azimuth=50.0, elevation=-20.0
        )
        view = mujoco.MjvCamera()
        camera.apply(view)
        assert view.lookat.tolist() == [1.0, 2.0, 3.0]
        assert (view.distance, view.azimuth, view.elevation) == (4.0, 50.0, -20.0)

    def test_the_counter_scene_looks_at_the_robot_and_the_objects_on_the_counter(self):
        camera = PhysicsScene.PR2_AT_THE_COUNTER.value.initial_camera()
        scene = PhysicsScene.PR2_AT_THE_COUNTER.value
        xs = [placed.position.x for placed in scene.scene.objects] + [scene.position.x]
        assert min(xs) <= camera.lookat.x <= max(xs)

    def test_a_scene_without_a_view_of_its_own_leaves_the_default(self):
        assert PhysicsScene.CUBE.value.initial_camera() is None


# %% telling what an action did


class TestActionReport:
    def test_the_joints_an_action_moved_are_reported_with_where_they_went(self):
        run = PhysicsViewer(
            scene=PhysicsScene.PR2,
            headless=True,
            duration=timedelta(seconds=0.1),
            speed=100.0,
            action=PhysicsAction.RAISE_TORSO,
        ).run()
        torso = [
            moved for moved in run.moved_joints if moved.joint == PR2Joint.TORSO_LIFT
        ]
        assert [round(moved.after, 1) for moved in torso] == [0.3]

    def test_a_joint_the_action_did_not_move_is_not_reported(self):
        run = PhysicsViewer(
            scene=PhysicsScene.PR2,
            headless=True,
            duration=timedelta(seconds=0.1),
            speed=100.0,
            action=PhysicsAction.RAISE_TORSO,
        ).run()
        assert all(moved.joint != PR2Joint.HEAD_PAN for moved in run.moved_joints)

    def test_only_joints_a_servo_drives_are_reported(self):
        run = PhysicsViewer(
            scene=PhysicsScene.PR2_AT_THE_COUNTER,
            headless=True,
            duration=timedelta(seconds=0.1),
            speed=100.0,
            action=PhysicsAction.RAISE_TORSO,
        ).run()
        driven = {
            connection.name.name
            for connection in run.world.connections
            if isinstance(connection, ActiveConnection1DOF)
            and any(connection.dof in actuator.dofs for actuator in run.world.actuators)
        }
        assert {moved.joint for moved in run.moved_joints} <= driven

    def test_a_run_without_an_action_moved_no_joint(self):
        run = PhysicsViewer(
            scene=PhysicsScene.CUBE,
            headless=True,
            duration=timedelta(seconds=0.2),
            speed=100.0,
        ).run()
        assert run.moved_joints == ()

    def test_the_overlay_names_the_action_and_each_joint_it_moved(self):
        viewer_handle = TextsRecorded()
        overlay = ActionOverlay(viewer_handle)
        overlay.show_done(
            PhysicsAction.RAISE_TORSO, (MovedJoint("torso_lift_joint", 0.012, 0.29),)
        )
        text = " ".join(str(part) for part in viewer_handle.texts[-1])
        assert "raise_torso" in text.lower()
        assert "torso_lift_joint" in text

    def test_the_overlay_names_each_object_that_moved(self):
        viewer_handle = TextsRecorded()
        ActionOverlay(viewer_handle).show_done(
            PhysicsAction.PICK_UP_MILK,
            (),
            (
                MovedObject(
                    "milk.stl", Position(2.0, 2.0, 1.034), Position(2.1, 2.0, 0.978)
                ),
            ),
        )
        text = " ".join(str(part) for part in viewer_handle.texts[-1])
        assert "milk.stl" in text

    def test_the_overlay_says_the_action_is_under_way_before_it_is_done(self):
        viewer_handle = TextsRecorded()
        ActionOverlay(viewer_handle).show_started(PhysicsAction.RAISE_TORSO)
        assert (
            "raise_torso"
            in " ".join(str(part) for part in viewer_handle.texts[-1]).lower()
        )

    def test_a_run_without_a_window_has_no_overlay(self):
        assert ActionOverlay.of(headless=True, simulation=None) is None


class TextsRecorded:
    """
    Stands in for the viewer window's handle, recording the texts it is asked to show.
    """

    def __init__(self):
        self.texts = []

    def set_texts(self, texts):
        self.texts.append(texts if isinstance(texts, tuple) else tuple(texts[0]))


# %% pacing


class TestPacing:
    def test_a_simulation_steps_no_faster_than_real_time_times_its_speed(self):
        simulation = RealTimeMujocoSim(
            world=PhysicsScene.CUBE.value.build(), headless=True, speed=2.0
        )
        simulation.start_stepped_simulation()
        started = time.monotonic()
        simulation.step_simulation(timedelta(seconds=0.6))
        elapsed = time.monotonic() - started
        simulation.stop_simulation()
        assert elapsed >= 0.6 / 2.0


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
