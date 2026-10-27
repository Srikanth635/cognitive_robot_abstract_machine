"""
What the add-on cannot do, because of what it was asked for, because the viewer it
extends is not as it expects, or because the environment does not say what it needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


@dataclass
class NoPerformanceWaiting(DataclassException):
    """
    A performance was asked to start, and none is waiting for it.
    """

    def error_message(self) -> str:
        return "no performance is waiting to start"

    def suggest_correction(self) -> str:
        return "press Execute on a run first, and start it while it is waiting"


# %% the models the add-on asks


@dataclass
class MissingCredentialsError(DataclassException):
    """
    Raised when the key a model's client authenticates with is not available.
    """

    variable: str
    """
    The environment variable that should hold the key.
    """

    searched: Optional[Path] = None
    """
    The ``.env`` file that was read for it, if any.
    """

    def error_message(self) -> str:
        location = f" or in {self.searched}" if self.searched else ""
        return f"{self.variable} is not set{location}"

    def suggest_correction(self) -> str:
        return "export it or add it to a .env file, then retry"


@dataclass
class UnreadableModelChoice(DataclassException):
    """
    Raised when the environment chooses a model in words that name no provider and
    model.
    """

    stated: str
    """
    The choice as the environment states it.
    """

    providers: str
    """
    The providers a model can be asked through, listed.
    """

    def error_message(self) -> str:
        return f"{self.stated!r} names no provider and model"

    def suggest_correction(self) -> str:
        return f"write it as provider:model, with a provider among {self.providers}"


@dataclass
class UnreadablePrices(DataclassException):
    """
    Raised when the prices the environment states for a model are not one price for each
    kind of token.
    """

    stated: str
    """
    The prices as the environment states them.
    """

    kinds: str
    """
    The kinds of token a price is stated for, listed.
    """

    def error_message(self) -> str:
        return f"{self.stated!r} is not one price for each kind of token"

    def suggest_correction(self) -> str:
        return (
            "state a JSON object of dollars per million tokens with exactly the keys "
            f"{self.kinds}"
        )
