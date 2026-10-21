"""
The perspectives on the action core a viewer's resolution may consult beside the phase
structure: every perspective the knowledge component registers, offered for choosing,
none consulted unless chosen.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from llm_reasoner.attribute_docstrings import attribute_docstrings
from llm_reasoner.knowledge.access import PERSPECTIVE_ROOTS, ActionPerspectiveFile, core
from typing_extensions import Any, Dict, List, Sequence, Tuple

from cram_viz.resolution.resolver import DesignatorResolver


class PerspectiveKey(StrEnum):
    """
    The keys a perspective is offered with.
    """

    NAME = "name"
    LABEL = "label"
    SUMMARY = "summary"


@dataclass
class UnknownPerspective(LookupError):
    """
    A request chose a perspective the knowledge component does not register.
    """

    name: str
    """
    The name the request gave.
    """

    def __post_init__(self):
        super().__init__(f"no perspective named {self.name!r}")


@dataclass(frozen=True)
class OfferedPerspective:
    """
    A perspective a resolution may consult, with the action core it is about.
    """

    module: ActionPerspectiveFile
    """
    The perspective's ontology module.
    """

    @classmethod
    def every(cls) -> List[OfferedPerspective]:
        """
        :return: Every perspective the knowledge component registers, in its order.
        """
        return [cls(module=module) for module in PERSPECTIVE_ROOTS]

    @classmethod
    def named(cls, names: Sequence[str]) -> Tuple[ActionPerspectiveFile, ...]:
        """
        :param names: The modules' file names, as :meth:`payload` gives them.
        :return: The perspectives of those names, in the order given.
        :raises UnknownPerspective: When a name is no registered perspective's.
        """
        by_name = {offered.module.value: offered.module for offered in cls.every()}
        unknown = [name for name in names if name not in by_name]
        if unknown:
            raise UnknownPerspective(name=unknown[0])
        return tuple(by_name[name] for name in names)

    def payload(self) -> Dict[str, Any]:
        """
        :return: The perspective as the viewer offers it: its name, a label and what it
            says of itself.
        """
        return {
            PerspectiveKey.NAME: self.module.value,
            PerspectiveKey.LABEL: Path(self.module.value).stem.replace("-", " "),
            PerspectiveKey.SUMMARY: self.module.summary,
        }


WORD_START = re.compile(r"(?<!^)(?=[A-Z])")
"""
Where a new word begins in a class name written in capitals.
"""


@dataclass(frozen=True)
class OfferedGeometryAdjustment:
    """
    Letting a miss open the pour geometry the query leaves out, offered for choosing
    beside the knowledge Generate consults.
    """

    label: str = "adjust the pour geometry"
    """
    What the viewer calls the choice.
    """

    def payload(self) -> Dict[str, Any]:
        """
        :return: The choice as the viewer offers it: a label and what the resolver does
            when it is made.
        """
        return {
            PerspectiveKey.LABEL: self.label,
            PerspectiveKey.SUMMARY: attribute_docstrings(DesignatorResolver)[
                "adjusts_pour_geometry"
            ],
        }


@dataclass(frozen=True)
class OfferedPhaseStructure:
    """
    The phase structure describing an action, offered for choosing like a perspective.
    """

    structure: type = core.ActionPhaseStructure
    """
    The vocabulary class every phase structure specialises.
    """

    def payload(self) -> Dict[str, Any]:
        """
        :return: The phase structure as the viewer offers it: a label and what its
            ontology says it is.
        """
        return {
            PerspectiveKey.LABEL: WORD_START.sub(" ", self.structure.__name__).lower(),
            PerspectiveKey.SUMMARY: inspect.getdoc(self.structure),
        }
