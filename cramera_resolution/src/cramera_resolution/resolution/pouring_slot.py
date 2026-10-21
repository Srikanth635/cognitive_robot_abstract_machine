"""
The slots of a pouring that cramera_resolution refers to by name.
"""

from __future__ import annotations

from enum import StrEnum

from typing_extensions import Tuple


class PouringSlot(StrEnum):
    """
    A field of coraplex's ``PouringAction`` that cramera_resolution reads or writes by name.
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

    POUR_SIDE_OFFSET = "pour_side_offset"
    """
    How far the pour point is shifted sideways from the target container's centre.
    """

    POUR_APPROACH_OFFSET = "pour_approach_offset"
    """
    How far the pour point is shifted away from the target container along the approach.
    """

    POUR_HEIGHT = "pour_height"
    """
    How high the tool is held above the target container.
    """

    @classmethod
    def geometry(cls) -> Tuple[PouringSlot, ...]:
        """
        :return: The slots that place the pour point and the tool over the target.
        """
        return (cls.POUR_SIDE_OFFSET, cls.POUR_APPROACH_OFFSET, cls.POUR_HEIGHT)

    @classmethod
    def containers(cls) -> Tuple[PouringSlot, ...]:
        """
        :return: The slots that hold a container: the source, then the target.
        """
        return (cls.SOURCE_CONTAINER, cls.TARGET_CONTAINER)
