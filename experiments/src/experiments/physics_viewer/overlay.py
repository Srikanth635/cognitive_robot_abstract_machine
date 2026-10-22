"""
Text over the window that says what the robot is doing and what it did.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
from typing_extensions import Sequence

from experiments.physics_viewer.actions import PhysicsAction


@dataclass(frozen=True)
class ActionOverlay:
    """
    Writes the action the robot performs, and the joints it moved, into the window.
    """

    window: mujoco.viewer.Handle
    """
    The handle of the window.
    """

    def show_started(self, action: PhysicsAction) -> None:
        """
        Say that an action is being performed.

        :param action: The action.
        """
        self.write(f"{action.name.lower()}: performing", "")

    def show_done(self, action: PhysicsAction, moved: Sequence[object]) -> None:
        """
        Say that an action is done, and which joints it moved from where to where.

        :param action: The action.
        :param moved: The joints it moved, each with its name and where it was and is.
        """
        self.write(
            f"{action.name.lower()}: done",
            "\n".join(
                f"{joint.joint}: {joint.before:.3f} -> {joint.after:.3f}"
                for joint in moved
            ),
        )

    def write(self, heading: str, body: str) -> None:
        """
        Replace the text over the window.

        :param heading: The left-hand line.
        :param body: The right-hand lines.
        """
        self.window.set_texts(
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                heading,
                body,
            )
        )

    @classmethod
    def of(cls, headless: bool, simulation) -> ActionOverlay | None:
        """
        :param headless: Whether the run has no window.
        :param simulation: The simulation being stepped.
        :return: The overlay of its window, or ``None`` when there is no window.
        """
        return None if headless else cls(simulation.simulator.renderer)
