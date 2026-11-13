"""
What the simulators of this package refuse to do.
"""

from __future__ import annotations

from dataclasses import dataclass

# %% starting a simulator


@dataclass
class SteppedOnAnotherThreadError(RuntimeError):
    """
    Raised when a simulator that steps only on the thread that started it is started to
    step or draw on a thread of its own.
    """

    simulator_name: str
    """
    The name of the simulator.
    """

    def __post_init__(self) -> None:
        super().__init__(
            f"{self.simulator_name} steps and draws only on the thread that started it; "
            f"start it with simulate_in_thread=False and render_in_thread=False, and "
            f"step it from that thread."
        )


# %% reading a simulation


@dataclass
class ActuatorWithoutJointError(KeyError):
    """
    Raised when an actuator's length is asked for before the joint it drives has been
    named to the simulator.
    """

    actuator_name: str
    """
    The name of the actuator.
    """

    def __post_init__(self) -> None:
        super().__init__(
            f"no joint is bound to the actuator {self.actuator_name}; name every "
            f"actuator's joint with track_actuators before reading their lengths."
        )
