"""
How the world injector names individuals: within one description, or by handles that
stay the same across questions.
"""

from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field

from typing_extensions import Any, Dict, Optional

from llm_reasoner.world.exceptions import ExpiredHandle, UnknownHandle

HANDLE_SEPARATOR = "#"
"""
What separates the kind of an individual from its number in a handle.
"""


class IndividualNaming(ABC):
    """
    Gives individuals the identifiers references name them by.
    """

    @abstractmethod
    def identifier_of(self, individual: Any) -> str:
        """
        :return: The identifier of an individual, the same each time it is asked for.
        """


@dataclass
class LocalNumbering(IndividualNaming):
    """
    Numbers individuals in the order they are first named, for a description standing on
    its own.
    """

    identifiers: Dict[int, str] = field(default_factory=dict)
    """
    The identifier of every individual named so far, by object identity.
    """

    def identifier_of(self, individual: Any) -> str:
        if id(individual) not in self.identifiers:
            self.identifiers[id(individual)] = (
                f"{type(individual).__name__} {len(self.identifiers) + 1}"
            )
        return self.identifiers[id(individual)]


@dataclass
class HandleRegistry(IndividualNaming):
    """
    Gives individuals handles that stay the same across questions, such as ``Cup#1``,
    without keeping an individual alive: a handle outlives its individual only to say
    that it has expired.
    """

    handles: Dict[int, str] = field(default_factory=dict)
    """
    The handle of every live individual given one, by object identity.
    """

    individuals: Dict[str, weakref.ref] = field(default_factory=dict)
    """
    The individual every handle was given to, held weakly.
    """

    counts: Counter = field(default_factory=Counter)
    """
    How many handles each kind has been given.
    """

    def identifier_of(self, individual: Any) -> str:
        return self.handle_of(individual)

    def handle_of(self, individual: Any) -> str:
        """
        :return: The handle of an individual, given out on first asking.
        """
        if id(individual) in self.handles:
            return self.handles[id(individual)]
        kind = type(individual).__name__
        self.counts[kind] += 1
        handle = f"{kind}{HANDLE_SEPARATOR}{self.counts[kind]}"
        identity = id(individual)
        self.handles[identity] = handle
        self.individuals[handle] = weakref.ref(
            individual, lambda _: self.handles.pop(identity, None)
        )
        return handle

    def is_live(self, handle: str) -> bool:
        """
        :return: Whether a handle was given out to an individual that still exists.
        """
        return handle in self.individuals and self.individuals[handle]() is not None

    @staticmethod
    def kind_written_in(handle: str) -> Optional[str]:
        """
        :return: The name of the kind a handle is written for, or ``None`` when it is not
            written as handles are.
        """
        kind_name, separator, _ = handle.partition(HANDLE_SEPARATOR)
        return kind_name if separator and kind_name else None

    def individual_of(self, handle: str) -> Any:
        """
        :return: The individual a handle was given to.
        :raises UnknownHandle: When the handle was never given out.
        :raises ExpiredHandle: When its individual no longer exists.
        """
        if handle not in self.individuals:
            raise UnknownHandle(handle=handle, kind_name=self.kind_written_in(handle))
        individual = self.individuals[handle]()
        if individual is None:
            raise ExpiredHandle(handle=handle)
        return individual
