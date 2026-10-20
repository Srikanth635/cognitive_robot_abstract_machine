"""
The tilt at which the generated knowledge says a poured substance starts to flow, read
from the hypotheses the resolver consulted.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import radians

from llm_reasoner.knowledge.access import core, pouring
from typing_extensions import Callable, List, Optional, Protocol, Sequence

ANGLE_UNIT_IN_RADIANS = {core.MeasurementUnit.DEGREE: radians(1.0)}
"""
How many radians one unit of each angle unit the ontology has stands for.
"""


class StatesFlowOnset(Protocol):
    """
    A hypothesis that may state flow onset events.
    """

    def instances_of(self, kind: type) -> list:
        """
        :return: The fragments of the hypothesis that are instances of the vocabulary
            class.
        """


class ReadsHypotheses(Protocol):
    """
    A knowledge component that gives the hypotheses of stored records.
    """

    def views_read_with(self, record_ids: Sequence[str]) -> List[StatesFlowOnset]:
        """
        :return: The hypotheses of the records and of the hypotheses they build on.
        """


@dataclass
class FlowOnsetKnowledge:
    """
    The tilt, stated by the hypotheses the resolver consulted, below which a pour pours
    nothing.
    """

    knowledge: ReadsHypotheses
    """
    The knowledge component holding the hypotheses.
    """

    consulted_records: Callable[[], Sequence[str]]
    """
    The records of the hypotheses the resolver consulted, asked for when the tilt is.
    """

    def tilt(self) -> Optional[float]:
        """
        :return: The lowest tilt in radians that a consulted hypothesis states for the
            substance to start flowing, as the lower bound of the threshold its flow
            onset crosses; ``None`` when none states one as an angle.
        """
        lower_bounds = [
            lower_bound
            for hypothesis in self.knowledge.views_read_with(self.consulted_records())
            for event in hypothesis.instances_of(pouring.FlowOnset)
            if (lower_bound := self.lower_bound_in_radians(event.crosses_threshold))
            is not None
        ]
        return min(lower_bounds, default=None)

    @staticmethod
    def lower_bound_in_radians(
        threshold: Optional[core.MechanicalThreshold],
    ) -> Optional[float]:
        """
        :return: The lower bound of the threshold's range in radians; ``None`` when it
            states no range, no lower bound, or one that is no angle.
        """
        parameter_range = None if threshold is None else threshold.has_parameter_range
        if parameter_range is None or parameter_range.lower_bound is None:
            return None
        radians_per_unit = ANGLE_UNIT_IN_RADIANS.get(parameter_range.has_unit)
        if radians_per_unit is None:
            return None
        return parameter_range.lower_bound * radians_per_unit
