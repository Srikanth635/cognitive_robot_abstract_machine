"""
Tests for putting the physics viewer's scenes under Isaac Sim's PhysX instead of MuJoCo.

They need Isaac Sim, which starts once, headless, before its modules can be imported,
and are skipped where it is not installed.
"""

from __future__ import annotations

import json
import time
from datetime import timedelta
from pathlib import Path

import pytest

pytest.importorskip("isaacsim")

from physics_simulators.isaac_application import IsaacApplication

IsaacApplication(headless=True).start()

from semantic_digital_twin.adapters.mujoco_video_recording import VideoResolution

from experiments.physics_viewer.actions import PhysicsAction
from experiments.physics_viewer.isaac.__main__ import parse_arguments
from experiments.physics_viewer.isaac.cameras import SceneCamera
from experiments.physics_viewer.isaac.paced_simulation import RealTimeIsaacSim
from experiments.physics_viewer.isaac.player import PlayerTemplate, RecordingFile
from experiments.physics_viewer.isaac.viewer import IsaacPhysicsViewer
from experiments.physics_viewer.scenes import PhysicsScene

# %% the command line


def test_the_command_line_asks_for_the_run_it_names():
    viewer = parse_arguments(
        [
            "pr2_at_the_counter",
            "--action",
            "open_grippers",
            "--headless",
            "--speed",
            "4",
            "--duration",
            "1.5",
            "--camera",
            "robot_camera",
            "--record",
            "recordings",
        ]
    )
    assert viewer.scene is PhysicsScene.PR2_AT_THE_COUNTER
    assert viewer.action is PhysicsAction.OPEN_GRIPPERS
    assert viewer.headless
    assert viewer.speed == 4.0
    assert viewer.duration == timedelta(seconds=1.5)
    assert viewer.camera is SceneCamera.ROBOT_CAMERA
    assert viewer.record_directory == Path("recordings")


# %% stepping at the pace of the clock


def test_a_simulation_steps_no_faster_than_real_time_times_its_speed():
    simulation = RealTimeIsaacSim(
        world=PhysicsScene.CUBE.value.build(), headless=True, speed=2.0
    )
    simulation.start_stepped_simulation()
    started = time.monotonic()
    simulation.step_simulation(timedelta(seconds=0.6))
    elapsed = time.monotonic() - started
    simulation.stop_simulation()
    assert elapsed >= 0.6 / 2.0


# %% an action


def test_an_action_reports_the_joints_it_moved():
    run = IsaacPhysicsViewer(
        scene=PhysicsScene.PR2_AT_THE_COUNTER,
        action=PhysicsAction.OPEN_GRIPPERS,
        headless=True,
        speed=100.0,
        duration=timedelta(seconds=0.1),
    ).run()
    assert run.moved_joints
    assert all("gripper" in joint.joint for joint in run.moved_joints)


# %% recording


def test_a_recorded_run_writes_a_video_of_every_camera_and_a_page_that_plays_them(
    tmp_path,
):
    IsaacPhysicsViewer(
        scene=PhysicsScene.PR2_AT_THE_COUNTER,
        action=PhysicsAction.OPEN_GRIPPERS,
        headless=True,
        speed=100.0,
        duration=timedelta(seconds=0.2),
        record_directory=tmp_path,
        recording_frames_per_second=5,
        recording_resolution=VideoResolution(width=160, height=120),
    ).run()
    videos = {
        f"{camera.name.lower()}{RecordingFile.VIDEO_SUFFIX}" for camera in SceneCamera
    }
    assert {path.name for path in tmp_path.iterdir()} == videos | {RecordingFile.PLAYER}
    page = (tmp_path / RecordingFile.PLAYER).read_text()
    assert PlayerTemplate.RECORDING not in page
    description = json.loads(
        page.split('type="application/json">')[1].split("</script>")[0]
    )
    assert [camera["label"] for camera in description["cameras"]] == [
        camera.label for camera in SceneCamera
    ]
