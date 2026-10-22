"""
A MuJoCo simulation that steps no faster than the clock, so what happens in the window
can be followed.
"""

from __future__ import annotations

import time
from datetime import timedelta

from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.world import World


class RealTimeMujocoSim(MujocoSim):
    """
    A :class:`~semantic_digital_twin.adapters.multi_sim.MujocoSim` whose stepped physics
    takes as long as the simulated time it advances, divided by a speed.

    A plan performed in a stepped simulation steps it once per control tick as fast as the
    controller computes, and a free-running scene steps as fast as the machine can; either
    ends before a window can show it.
    """

    def __init__(self, world: World, speed: float = 1.0, **keywords) -> None:
        """
        :param world: The world to simulate.
        :param speed: How many times faster than real time the simulated time passes.
        :param keywords: What :class:`~semantic_digital_twin.adapters.multi_sim.MujocoSim`
            takes besides the world.
        """
        super().__init__(world=world, **keywords)
        self.speed = speed
        self.clock_started = time.monotonic()
        self.simulated_seconds = 0.0

    def start_stepped_simulation(self) -> None:
        super().start_stepped_simulation()
        self.clock_started = time.monotonic()
        self.simulated_seconds = 0.0

    def step_simulation(self, duration: timedelta) -> None:
        """
        Advance the physics, then wait until the clock has caught up with the simulated
        time.

        :param duration: How much simulated time to advance.
        """
        super().step_simulation(duration)
        self.simulated_seconds += duration.total_seconds()
        due = self.clock_started + self.simulated_seconds / self.speed
        time.sleep(max(0.0, due - time.monotonic()))
