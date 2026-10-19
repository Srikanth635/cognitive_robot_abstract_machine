"""
Tests of the pour check requiring the tilt at which the generated knowledge says the
substance starts to flow: a pour tilted less than that lands its mouth over the opening
and pours nothing, so it is refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import radians

import pytest
from coraplex.datastructures.enums import Arms
from llm_reasoner.knowledge.access import core, pouring
from llm_reasoner.resolver.exceptions import OutcomeFellShort
from typing_extensions import List, Optional

from cram_viz.resolution.flow_onset import FlowOnsetKnowledge
from cram_viz.resolution.trial import OutcomeKey

from .resolutions import LANDS
from .test_pouring_trial import PourAnswer, checking

# %% helpers


def threshold(
    lower_bound: Optional[float],
    unit: core.MeasurementUnit = core.MeasurementUnit.DEGREE,
) -> pouring.FlowOnsetThreshold:
    """
    :return: A flow onset threshold stating the lowest tilt, in the unit, or stating no
        range at all when there is no lower bound.
    """
    if lower_bound is None:
        return pouring.FlowOnsetThreshold()
    return pouring.FlowOnsetThreshold(
        has_parameter_range=core.MagnitudeRange(
            lower_bound=lower_bound, upper_bound=lower_bound + 50, has_unit=unit
        )
    )


@dataclass
class StatedThresholds:
    """
    The knowledge component, as far as the flow onset reads it: it holds the thresholds
    its hypotheses state.
    """

    thresholds: List[pouring.FlowOnsetThreshold]
    """
    The thresholds the stored hypotheses state.
    """

    def instances_of(self, kind: type) -> list:
        """
        :return: The stated thresholds when asked for flow onset thresholds.
        """
        assert kind is pouring.FlowOnsetThreshold
        return self.thresholds


def knowledge_stating(*thresholds: pouring.FlowOnsetThreshold) -> FlowOnsetKnowledge:
    """
    :return: Flow onset knowledge over hypotheses stating the thresholds.
    """
    return FlowOnsetKnowledge(knowledge=StatedThresholds(thresholds=list(thresholds)))


SHALLOW_TILT = 0.2
"""
A tilt, in radians, below the onset the knowledge states in these tests.
"""

STATED_ONSET_DEGREES = 60.0
"""
The lowest tilt the knowledge states in these tests, in degrees.
"""


# %% reading the tilt from the knowledge


class TestFlowOnsetKnowledge:
    def test_the_lowest_stated_tilt_is_read_in_radians(self):
        flow_onset = knowledge_stating(threshold(STATED_ONSET_DEGREES))
        assert flow_onset.tilt() == radians(STATED_ONSET_DEGREES)

    def test_the_lowest_of_several_stated_tilts_is_read(self):
        flow_onset = knowledge_stating(threshold(90.0), threshold(STATED_ONSET_DEGREES))
        assert flow_onset.tilt() == radians(STATED_ONSET_DEGREES)

    def test_no_stated_threshold_leaves_the_tilt_unknown(self):
        assert knowledge_stating().tilt() is None

    def test_a_threshold_stating_no_range_is_ignored(self):
        flow_onset = knowledge_stating(threshold(None), threshold(STATED_ONSET_DEGREES))
        assert flow_onset.tilt() == radians(STATED_ONSET_DEGREES)

    def test_a_range_in_another_unit_is_ignored(self):
        flow_onset = knowledge_stating(
            threshold(1.0, unit=core.MeasurementUnit.DEGREE_PER_SECOND)
        )
        assert flow_onset.tilt() is None


# %% the check


class TestPourFlows:
    def answer(self, tilt_angle: float) -> PourAnswer:
        return PourAnswer(arm=Arms.RIGHT, tilt_angle=tilt_angle)

    def check_stating(self, allowed_trials: int = 3):
        check = checking([LANDS] * allowed_trials, allowed_trials=allowed_trials)
        check.flow_onset = knowledge_stating(threshold(STATED_ONSET_DEGREES))
        return check

    def test_a_pour_landing_in_the_opening_below_the_onset_is_refused(self):
        check = self.check_stating()
        with pytest.raises(OutcomeFellShort) as refusal:
            check.check([self.answer(SHALLOW_TILT)])
        assert refusal.value.outcome[OutcomeKey.TILT_REACHES_FLOW_ONSET] is False

    def test_the_refusal_states_the_tilt_the_substance_needs(self):
        check = self.check_stating()
        with pytest.raises(OutcomeFellShort) as refusal:
            check.check([self.answer(SHALLOW_TILT)])
        assert refusal.value.outcome[OutcomeKey.FLOW_ONSET_TILT_RADIANS] == radians(
            STATED_ONSET_DEGREES
        )

    def test_a_pour_landing_in_the_opening_at_the_onset_is_accepted(self):
        check = self.check_stating()
        check.check([self.answer(radians(STATED_ONSET_DEGREES))])
        assert len(check.trials) == 1

    def test_a_pour_below_the_onset_is_accepted_as_the_last_allowed_trial(self):
        check = self.check_stating(allowed_trials=1)
        check.check([self.answer(SHALLOW_TILT)])
        assert len(check.trials) == 1

    def test_without_a_stated_onset_a_shallow_pour_is_accepted(self):
        check = checking([LANDS])
        check.check([self.answer(SHALLOW_TILT)])
        assert len(check.trials) == 1

    def test_a_stated_onset_leaves_the_tilt_unchecked_when_knowledge_states_none(self):
        check = checking([LANDS])
        check.flow_onset = knowledge_stating()
        check.check([self.answer(SHALLOW_TILT)])
        assert len(check.trials) == 1
