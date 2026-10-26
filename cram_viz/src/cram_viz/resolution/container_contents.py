"""
What a container holds, as a scene bundle states it: the world fact the flow onset of a
pour depends on, which no recording can observe.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from llm_reasoner.knowledge.access import pouring_frame
from typing_extensions import Any, Mapping

from cram_viz.resolution.exceptions import (
    FillFractionOutOfRange,
    UnknownMaterialRegime,
)

# %% container contents


class ContentsKey(StrEnum):
    """
    The keys of a container's contents in a scene description.
    """

    MATERIAL_REGIME = "materialRegime"
    """
    How the substance held flows, by the name of its member of
    :class:`~llm_reasoner.knowledge.ontologies.action_perspectives.pouring_frame.MaterialRegime`.
    """

    FILL_FRACTION = "fillFraction"
    """
    How full the container is, from 0 (empty) to 1 (filled to its lowest lip).
    """


@dataclass(frozen=True)
class ContainerContents:
    """
    What a container holds and how full it is.

    :raises FillFractionOutOfRange: When the fill fraction is not between 0 and 1.
    """

    material_regime: pouring_frame.MaterialRegime
    """
    How the substance held flows.
    """

    fill_fraction: float
    """
    The volume held over the volume the container holds when upright and filled to its
    lowest lip.
    """

    def __post_init__(self) -> None:
        if not 0.0 <= self.fill_fraction <= 1.0:
            raise FillFractionOutOfRange(fill_fraction=self.fill_fraction)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ContainerContents:
        """
        :param data: The contents as a scene description states them, the material
            regime by the name of its member.
        :return: The contents stated.
        :raises UnknownMaterialRegime: When the ontology names no such material regime.
        """
        regime = data[ContentsKey.MATERIAL_REGIME]
        if regime not in pouring_frame.MaterialRegime.__members__:
            raise UnknownMaterialRegime(stated_regime=regime)
        return cls(
            material_regime=pouring_frame.MaterialRegime[regime],
            fill_fraction=data[ContentsKey.FILL_FRACTION],
        )

    def to_json(self) -> dict[str, Any]:
        """
        :return: The contents as a scene description states them.
        """
        return {
            ContentsKey.MATERIAL_REGIME: self.material_regime.name,
            ContentsKey.FILL_FRACTION: self.fill_fraction,
        }
