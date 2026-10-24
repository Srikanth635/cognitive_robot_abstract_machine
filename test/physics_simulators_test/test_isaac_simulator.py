"""
Tests for stepping a USD stage under Isaac Sim's PhysX through
:class:`physics_simulators.isaac_simulator.IsaacSimulator`.

They need Isaac Sim, which starts once, headless, before its modules can be imported,
and are skipped where it is not installed.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import pytest

pytest.importorskip("isaacsim")

from physics_simulators.isaac_application import IsaacApplication

IsaacApplication(headless=True).start()

from physics_simulators.base_simulator import SimulatorRenderer
from physics_simulators.exceptions import SteppedOnAnotherThreadError
from physics_simulators.isaac_simulator import IsaacRenderer, IsaacSimulator
from pxr import Usd, UsdPhysics

# %% the stage


class StagePrim(StrEnum):
    """
    The prims of the stage the tests step, by what they are.
    """

    FILE = str(Path(__file__).parent / "resources" / "hinge_and_free_box.usda")
    """
    The stage's file.
    """

    PHYSICS_SCENE = "/World/physicsScene"
    """
    The physics scene, which states gravity.
    """

    HINGE = "/World/joints/hinge"
    """
    A hinge of a fixed-base articulation, about the vertical, driven towards a target.
    """

    BOX = "/World/box"
    """
    A box free to fall.
    """


@pytest.fixture
def step_size() -> float:
    """
    :return: The simulated time, in seconds, each step of the tests' simulators advances.
    """
    return 1.0 / 200.0


def simulator_of(step_size: float, headless: bool = True) -> IsaacSimulator:
    """
    :return: A simulator of the stage, not started.
    """
    return IsaacSimulator(
        file_path=StagePrim.FILE, _headless=headless, _step_size=step_size
    )


@pytest.fixture
def gravity() -> float:
    """
    :return: The acceleration, in metres per square second, the stage's physics scene
        pulls bodies down with, read before a simulator has the stage open.
    """
    stage = Usd.Stage.Open(StagePrim.FILE.value)
    scene = UsdPhysics.Scene(stage.GetPrimAtPath(StagePrim.PHYSICS_SCENE.value))
    return scene.GetGravityMagnitudeAttr().Get()


@pytest.fixture
def simulator(step_size):
    """
    :return: A headless simulator of the stage, started stepped, stopped afterwards.
    """
    simulator = simulator_of(step_size)
    simulator.start(simulate_in_thread=False, render_in_thread=False)
    yield simulator
    simulator.stop()


def step_for(simulator: IsaacSimulator, seconds: float) -> None:
    """
    Step the simulator through the simulated time.
    """
    for _ in range(round(seconds / simulator.step_size)):
        simulator.step()


# %% starting


def test_a_start_that_would_step_on_another_thread_is_refused(step_size):
    with pytest.raises(SteppedOnAnotherThreadError):
        simulator_of(step_size).start()


def test_a_headless_simulator_draws_nothing(simulator):
    assert type(simulator.renderer) is SimulatorRenderer


def test_a_simulator_with_a_window_draws_it_through_isaac_sim(step_size):
    simulator = simulator_of(step_size, headless=False)
    simulator.start(simulate_in_thread=False, render_in_thread=False)
    try:
        renderer = simulator.renderer
        simulator.render()
    finally:
        simulator.stop()
    assert isinstance(renderer, IsaacRenderer)


# %% stepping


def test_each_step_advances_the_simulated_time_by_the_step_size(simulator):
    for _ in range(10):
        simulator.step()
    assert simulator.current_simulation_time == pytest.approx(10 * simulator.step_size)


def test_a_joint_driven_towards_a_target_gets_there(simulator):
    target = 0.5
    simulator.set_joint_position_targets({StagePrim.HINGE: target})
    step_for(simulator, 2.0)
    position = simulator.joint_positions([StagePrim.HINGE])[StagePrim.HINGE]
    assert position == pytest.approx(target, abs=0.01)


def test_a_joint_put_somewhere_is_there_at_once(simulator):
    simulator.set_joint_positions({StagePrim.HINGE: 0.3})
    position = simulator.joint_positions([StagePrim.HINGE])[StagePrim.HINGE]
    assert position == pytest.approx(0.3, abs=1e-4)


def test_a_free_body_falls_as_gravity_pulls_it(gravity, simulator):
    simulator.track_bodies([StagePrim.BOX])
    height = simulator.body_poses()[StagePrim.BOX].position[2]
    seconds = 0.5
    step_for(simulator, seconds)
    fallen = height - simulator.body_poses()[StagePrim.BOX].position[2]
    assert fallen == pytest.approx(gravity * seconds**2 / 2, abs=0.02)
