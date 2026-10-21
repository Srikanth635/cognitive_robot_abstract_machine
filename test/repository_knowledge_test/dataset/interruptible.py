"""
Work that outlasts the notebook interrupt deadline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

# %% interruptible work


@dataclass
class PausingWork:
    """
    Remain in an interruptible operation for a bounded interval.
    """

    seconds: float = 8.0
    """
    The uninterrupted work duration.
    """

    def wait(self) -> None:
        """
        Remain active until interrupted or the work interval elapses.
        """
        time.sleep(self.seconds)


if __name__ == "__main__":
    PausingWork().wait()
