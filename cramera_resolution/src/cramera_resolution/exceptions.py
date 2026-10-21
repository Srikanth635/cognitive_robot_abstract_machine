"""
What the add-on cannot do, because of what it was asked for or because the viewer it
extends is not as it expects.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException


@dataclass
class IndexPageHasNoBootScript(DataclassException):
    """
    The viewer's page has no script that boots its panels, so there is nowhere to insert
    the add-on's panels before it runs.
    """

    marker: str
    """
    The text the boot script was looked for by.
    """

    def error_message(self) -> str:
        return f"the viewer's index page has no script containing {self.marker!r}"

    def suggest_correction(self) -> str:
        return "check what the installed cramera calls to mount its panels"


@dataclass
class NoSuchScene(DataclassException):
    """
    No scene bundle is held under the requested name, and none is the default.
    """

    requested: str | None
    """
    The name that was asked for, or ``None`` when the default scene was wanted.
    """

    def error_message(self) -> str:
        wanted = "the default scene" if self.requested is None else repr(self.requested)
        return f"no scene bundle for {wanted}"

    def suggest_correction(self) -> str:
        return "pick one of the scenes the viewer's scene index lists"


@dataclass
class EmptyInstruction(DataclassException):
    """
    An instruction to generate knowledge for was asked for, and it was empty.
    """

    def error_message(self) -> str:
        return "empty instruction"

    def suggest_correction(self) -> str:
        return "state what the robot is to do"
