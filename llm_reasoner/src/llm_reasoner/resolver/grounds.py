"""
What an answer rests on, as data: each element of an answer names the world individuals
and knowledge fragments its choices rest on, each name is checked against its source,
and the answer keeps the stable identifiers they stand for.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import Any, List, Optional

from llm_reasoner.world.access import HANDLE_SEPARATOR, WorldAccess


class GroundKey(StrEnum):
    """
    The sources an answer may rest on, each the key an answer names one of its parts
    with.
    """

    WORLD = "world"
    """
    The world, its individuals named by handle.
    """

    KNOWLEDGE = "knowledge"
    """
    The knowledge about the requested action, its fragments named as the knowledge tools
    name them.
    """


class GroundSource(ABC):
    """
    Something an answer may name as what its choices rest on.
    """

    @property
    @abstractmethod
    def key(self) -> GroundKey:
        """
        The key an answer names a part of this source with.
        """

    @property
    @abstractmethod
    def name_form(self) -> str:
        """
        How a part of this source is named, as the rules state it.
        """

    @abstractmethod
    def identify(self, name: str) -> Optional[str]:
        """
        :param name: A name an answer gave.
        :return: The stable identifier of what the name names, ``None`` when it names
            nothing in this source.
        """


@dataclass(frozen=True)
class WorldGrounds(GroundSource):
    """
    The world's individuals, named by the handles the world tools gave out.
    """

    world: WorldAccess
    """
    The world giving out the handles.
    """

    @property
    def key(self) -> GroundKey:
        return GroundKey.WORLD

    @property
    def name_form(self) -> str:
        return f"<Kind>{HANDLE_SEPARATOR}<n>, a handle a tool returned"

    def identify(self, name: str) -> Optional[str]:
        return name if self.world.is_live(name) else None


@dataclass(frozen=True)
class Ground:
    """
    One thing an answer rests on.
    """

    source: GroundKey
    """
    The source it belongs to.
    """

    identifier: str
    """
    Its stable identifier: a world individual's handle, or the IRI citing a knowledge
    fragment.
    """


@dataclass
class Justification:
    """
    Why an answer chose what it chose, as the model stated it.
    """

    instance: Any
    """
    The instance built from the answer.
    """

    why: str
    """
    The model's one-sentence explanation.
    """

    grounds: List[Ground] = field(default_factory=list)
    """
    What the choices rest on, each checked against its source.
    """
