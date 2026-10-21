"""
What the add-on cannot do because the viewer it extends is not as it expects.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IndexPageHasNoBootScript(RuntimeError):
    """
    The viewer's page has no script that boots its panels, so there is nowhere to insert
    the add-on's panels before it runs.
    """

    marker: str
    """
    The text the boot script was looked for by.
    """

    def __post_init__(self):
        super().__init__(
            f"the viewer's index page has no script containing {self.marker!r}"
        )
