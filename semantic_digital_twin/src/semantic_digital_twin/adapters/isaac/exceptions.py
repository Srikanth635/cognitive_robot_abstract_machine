"""
What simulating a world under Isaac Sim cannot do.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException


@dataclass
class WorldStructureChangedUnderIsaacError(DataclassException):
    """
    Raised when bodies, connections or actuators are added to a world while Isaac Sim
    simulates it.
    """

    entity_name: str
    """
    The name of what was added.
    """

    def error_message(self) -> str:
        return (
            f"'{self.entity_name}' was added to the world while Isaac Sim simulates it, "
            f"and the simulation does not add it to its stage."
        )

    def suggest_correction(self) -> str:
        return "Add it before the simulation is built."


@dataclass
class ServoWithoutJointError(DataclassException):
    """
    Raised when a servo of the world drives a degree of freedom that no joint of the
    stage is commanded through.
    """

    servo_name: str
    """
    The name of the servo.
    """

    def error_message(self) -> str:
        return f"No joint of the stage is driven by the servo '{self.servo_name}'."

    def suggest_correction(self) -> str:
        return (
            "Drive a degree of freedom of a revolute, prismatic or wheeled connection."
        )


@dataclass
class DurationNotWholeStepsError(DataclassException):
    """
    Raised when the simulation is to advance by a duration that is not a whole number of
    its physics steps.
    """

    duration: float
    """
    The duration, in seconds.
    """

    step_size: float
    """
    The duration of one physics step, in seconds.
    """

    def error_message(self) -> str:
        return (
            f"{self.duration} s is not a whole number of physics steps of "
            f"{self.step_size} s."
        )

    def suggest_correction(self) -> str:
        return (
            "Choose a step size that divides every duration the simulation advances by."
        )
