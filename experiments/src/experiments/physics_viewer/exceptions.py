"""
What the physics viewer cannot do, because of how it was asked to run.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException


@dataclass
class HeadlessRunWithoutDuration(DataclassException):
    """
    A run without a window was asked to run until its window is closed, which it never
    is.
    """

    def error_message(self) -> str:
        return "a run without a window has no window to close, so it never ends"

    def suggest_correction(self) -> str:
        return "give the run a duration"


@dataclass
class NoSuchBody(DataclassException):
    """
    An action names a body the scene it is performed in does not have.
    """

    body: str
    """
    The name of the body that is missing.
    """

    def error_message(self) -> str:
        return f"the scene has no graspable body named {self.body!r}"

    def suggest_correction(self) -> str:
        return "perform the action in the scene that holds the body"


@dataclass
class NoToolInHand(DataclassException):
    """
    A tool-based action is performed with an arm whose hand holds no tool.
    """

    arm: str
    """
    The name of the arm whose hand is empty.
    """

    def error_message(self) -> str:
        return f"the hand of {self.arm} holds no tool"

    def suggest_correction(self) -> str:
        return "perform the action in a scene that puts a tool in that hand"
