"""
The tilt at which the generated knowledge says a poured substance starts to flow, read
from the hypotheses the resolver consulted.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import radians

from llm_reasoner.knowledge.access import core, pouring
from typing_extensions import List, Optional, Protocol


class StatesFlowOnsetThresholds(Protocol):
    """
    A knowledge component holding hypotheses that may state flow onset thresholds.
    """

    def instances_of(self, kind: type) -> list:
        """
        :return: The fragments of every stored hypothesis that are instances of the
            vocabulary class.
        """


@dataclass
class FlowOnsetKnowledge:
    """
    The tilt, stated by the stored hypotheses, below which a pour pours nothing.
    """

    knowledge: StatesFlowOnsetThresholds
    """
    The knowledge component whose hypotheses state the thresholds.
    """

    def tilt(self) -> Optional[float]:
        """
        :return: The lowest tilt in radians any stored hypothesis states for the
            substance to start flowing, or ``None`` when none states one in degrees.
        """
        lower_bounds: List[float] = [
            threshold.has_parameter_range.lower_bound
            for threshold in self.knowledge.instances_of(pouring.FlowOnsetThreshold)
            if self.states_degrees(threshold)
        ]
        if not lower_bounds:
            return None
        return radians(min(lower_bounds))

    @staticmethod
    def states_degrees(threshold: pouring.FlowOnsetThreshold) -> bool:
        """
        :return: Whether the threshold states a lower bound in degrees.
        """
        parameter_range = threshold.has_parameter_range
        return (
            parameter_range is not None
            and parameter_range.lower_bound is not None
            and parameter_range.has_unit is core.MeasurementUnit.DEGREE
        )
