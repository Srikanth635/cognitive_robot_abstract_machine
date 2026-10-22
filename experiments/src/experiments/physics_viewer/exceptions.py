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
