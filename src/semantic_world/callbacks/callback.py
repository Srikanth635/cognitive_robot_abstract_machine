from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ..world import World


@dataclass
class Callback(ABC):

    world: World
    _callback_function: Callable = field(init=False)

    def __post_init__(self):
        self._callback_function = lambda: self.notify()

    @abstractmethod
    def notify(self):
        """
        Notify the callback of a change in the world.
        """
        raise NotImplementedError


@dataclass
class StateChangeCallback(Callback, ABC):

    def __post_init__(self):
        super().__post_init__()
        self.world.state_change_callbacks.append(self._callback_function)

    def __del__(self):
        try:
            self.world.state_change_callbacks.remove(self._callback_function)
        except Exception:
            ...


@dataclass
class ModelChangeCallback(Callback, ABC):

    def __post_init__(self):
        super().__post_init__()
        self.world.model_change_callbacks.append(self._callback_function)

    def __del__(self):
        try:
            self.world.model_change_callbacks.remove(self._callback_function)
        except Exception:
            ...
