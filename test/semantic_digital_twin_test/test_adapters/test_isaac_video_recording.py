"""
Tests for recording videos of a world simulated under Isaac Sim, from cameras on its
stage.

They need Isaac Sim, which starts once, headless, before its modules can be imported,
and are skipped where it is not installed.
"""

from __future__ import annotations

import math
from datetime import timedelta

import pytest

pytest.importorskip("isaacsim")

from physics_simulators.isaac_application import IsaacApplication

IsaacApplication(headless=True).start()

import omni.usd
from pxr import UsdGeom

from semantic_digital_twin.adapters.isaac.simulation import IsaacSim
from semantic_digital_twin.adapters.isaac.video_recording import IsaacVideoRecorder
from semantic_digital_twin.adapters.multi_sim import MujocoCamera
from semantic_digital_twin.adapters.mujoco_video_recording import VideoResolution
from semantic_digital_twin.exceptions import (
    InvalidVideoRecordingRateError,
    VideoRecordingAlreadyStartedError,
    VideoRecordingNotStartedError,
)

from .test_usd_builder import _hinge_world

# %% a simulation with a camera


@pytest.fixture
def simulation():
    """
    :return: The hinge world simulated under Isaac Sim, started stepped, with a camera on
        its root looking at the base from two metres away; stopped afterwards.
    """
    world = _hinge_world().world
    world.root.add_simulator_property(
        MujocoCamera(
            name="watching",
            body=world.root,
            position=[0.0, -2.0, 0.5],
            quaternion=[math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0],
        )
    )
    simulation = IsaacSim(world=world)
    simulation.start_stepped_simulation()
    yield simulation
    simulation.stop_simulation()


@pytest.fixture
def camera(simulation) -> str:
    """
    :return: The path of the camera's prim.
    """
    [path] = [
        str(prim.GetPath())
        for prim in omni.usd.get_context().get_stage().Traverse()
        if prim.IsA(UsdGeom.Camera) and prim.GetName() == "watching"
    ]
    return path


# %% recording


def test_a_recording_has_a_frame_for_each_period_of_simulated_time(simulation, camera):
    recorder = IsaacVideoRecorder(
        simulation=simulation, cameras=[camera], frames_per_second=20
    )
    recorder.start()
    simulation.step_simulation(timedelta(seconds=0.5))
    videos = recorder.stop()
    assert len(videos[camera].frames) == 1 + round(0.5 * recorder.frames_per_second)
    assert videos[camera].frames_per_second == recorder.frames_per_second


def test_every_frame_has_the_resolution_asked_for(simulation, camera):
    resolution = VideoResolution(width=160, height=120)
    recorder = IsaacVideoRecorder(
        simulation=simulation, cameras=[camera], resolution=resolution
    )
    recorder.start()
    simulation.step_simulation(timedelta(milliseconds=100))
    [frame, *_] = recorder.stop()[camera].frames
    assert frame.shape == (resolution.height, resolution.width, 3)


def test_starting_a_recording_twice_is_refused(simulation, camera):
    recorder = IsaacVideoRecorder(simulation=simulation, cameras=[camera])
    recorder.start()
    with pytest.raises(VideoRecordingAlreadyStartedError):
        recorder.start()
    recorder.stop()


def test_stopping_a_recording_that_never_started_is_refused(simulation, camera):
    with pytest.raises(VideoRecordingNotStartedError):
        IsaacVideoRecorder(simulation=simulation, cameras=[camera]).stop()


def test_a_frame_rate_that_is_not_positive_is_refused(simulation, camera):
    with pytest.raises(InvalidVideoRecordingRateError):
        IsaacVideoRecorder(simulation=simulation, cameras=[camera], frames_per_second=0)
