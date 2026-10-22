"""
What the add-on cannot do, because of what it was asked for or because the viewer it
extends is not as it expects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from krrood.exceptions import DataclassException


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


@dataclass
class SceneArchiveHasNoIndex(DataclassException):
    """
    A directory to convert as a scene archive has no index of the scenes it holds.
    """

    directory: Path
    """
    The directory that was to be converted.
    """

    def error_message(self) -> str:
        return f"{self.directory} has no scene index, so it is not a scene archive"

    def suggest_correction(self) -> str:
        return "name the directory holding the archive's index.json"


@dataclass
class TargetInsideSceneArchive(DataclassException):
    """
    A scene archive was to be converted into a directory inside itself.
    """

    source: Path
    """
    The archive to convert.
    """

    target: Path
    """
    The directory the conversion was to write to.
    """

    def error_message(self) -> str:
        return (
            f"{self.target} lies inside the archive {self.source} it is converted from"
        )

    def suggest_correction(self) -> str:
        return "write the converted archive outside the one it is converted from"
