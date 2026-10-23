"""
The one Isaac Sim application a process runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from isaacsim import SimulationApp
from typing_extensions import ClassVar


@dataclass
class IsaacApplication:
    """
    Isaac Sim's application, started once for the whole process; the modules of Isaac
    Sim can be imported only once it runs.
    """

    headless: bool = True
    """
    Whether the application runs without a window, if it is not running yet.
    """

    _running: ClassVar[SimulationApp | None] = None
    """
    The application the process runs, shared by every instance because a process can run
    only one.
    """

    def start(self) -> SimulationApp:
        """
        :return: The application the process runs, started if it was not running.
        """
        if IsaacApplication._running is None:
            IsaacApplication._running = SimulationApp({"headless": self.headless})
        return IsaacApplication._running

    def update(self) -> None:
        """
        Let the running application handle one frame of its window.
        """
        self.start().update()
