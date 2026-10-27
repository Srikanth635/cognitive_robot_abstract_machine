"""
Errors in what a scene bundle states about its objects, in what is derived from it while
resolving designators against the scene, and in reading the runs kept.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException
from llm_reasoner.knowledge.access import pouring_frame

# %% container contents


@dataclass
class FillFractionOutOfRange(DataclassException):
    """
    Raised when a container is said to be filled to a share of its capacity it cannot
    be filled to: less than empty or more than full, or empty where something must flow.
    """

    fill_fraction: float
    """
    The share of the container's capacity it was said to be filled to.
    """

    allows_empty: bool = True
    """
    Whether an empty container was acceptable where the fill fraction was given.
    """

    def error_message(self) -> str:
        return (
            f"a container cannot be filled to a fraction of {self.fill_fraction} here"
        )

    def suggest_correction(self) -> str:
        lowest = "0 (empty)" if self.allows_empty else "above 0"
        return (
            f"state the fill fraction from {lowest} to 1 (filled to the container's "
            "lowest lip)"
        )


@dataclass
class UnknownMaterialRegime(DataclassException):
    """
    Raised when a container's contents are stated to be of a material regime the
    ontology does not name.
    """

    stated_regime: str
    """
    The material regime stated.
    """

    def error_message(self) -> str:
        return f"the ontology names no material regime {self.stated_regime!r}"

    def suggest_correction(self) -> str:
        regimes = ", ".join(pouring_frame.MaterialRegime.__members__)
        return f"state one of the material regimes {regimes}"


# %% runs kept


@dataclass
class StaleRun(DataclassException):
    """
    Raised when what a run resolved is read although the run is kept in a format other
    than the current one, which this version cannot read.
    """

    run: str
    """
    The name of the run.
    """

    def error_message(self) -> str:
        return f"run {self.run!r} is kept in a format this version cannot read"

    def suggest_correction(self) -> str:
        return "resolve the designator again, or open the run with the version that kept it"
