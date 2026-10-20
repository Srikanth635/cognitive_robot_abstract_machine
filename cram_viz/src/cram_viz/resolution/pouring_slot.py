"""
The slots of a pouring that cram_viz refers to by name.
"""

from __future__ import annotations

from enum import StrEnum

from typing_extensions import Tuple


class PouringSlot(StrEnum):
    """
    A field of coraplex's ``PouringAction`` that cram_viz reads or writes by name.
    """

    SOURCE_CONTAINER = "source_container"
    """
    The container poured from.
    """

    TARGET_CONTAINER = "target_container"
    """
    The container poured into.
    """

    ARM = "arm"
    """
    The arm holding the source container.
    """

    @classmethod
    def containers(cls) -> Tuple[PouringSlot, ...]:
        """
        :return: The slots that hold a container: the source, then the target.
        """
        return (cls.SOURCE_CONTAINER, cls.TARGET_CONTAINER)
