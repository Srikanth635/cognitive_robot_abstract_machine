"""
Tests of the pour check requiring the tilt at which the generated knowledge says the
substance starts to flow: a pour tilted less than that lands its mouth over the opening
and pours nothing, so it is refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import radians

import pytest
from llm_reasoner.knowledge.access import core, pouring
from llm_reasoner.resolver.exceptions import OutcomeFellShort
from typing_extensions import Dict, List, Optional, Sequence

from cramera_resolution.resolution.flow_onset import FlowOnsetKnowledge
from cramera_resolution.resolution.trial import OutcomeKey

from .resolutions import LANDS, HandSide
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
class StatedHypothesis:
    """
    A stored hypothesis, as far as the flow onset reads it: the flow onset events it
    states.
    """

    events: List[pouring.FlowOnset]
    """
    The flow onset events the hypothesis states.
    """

    def instances_of(self, kind: type) -> list:
        """
        :return: The stated events when asked for flow onset events.
        """
        assert kind is pouring.FlowOnset
        return self.events


@dataclass
class StoredHypotheses:
    """
    The knowledge component, as far as the flow onset reads it: hypotheses by record.
    """

    by_record: Dict[str, StatedHypothesis]
    """
    The stored hypotheses, by their record identifier.
    """

    def views_read_with(self, record_ids: Sequence[str]) -> list:
        """
        :return: The hypotheses of the records asked for, and only those.
        """
        return [self.by_record[record_id] for record_id in record_ids]


CONSULTED = "consulted-record"
"""
The record of the hypothesis the resolver consulted.
"""

ELSEWHERE = "other-record"
"""
The record of a stored hypothesis the resolver did not consult.
"""


def events_crossing(*thresholds: pouring.FlowOnsetThreshold) -> StatedHypothesis:
    """
    :return: A hypothesis stating one flow onset event crossing each threshold.
    """
    return StatedHypothesis(
        events=[pouring.FlowOnset(crosses_threshold=found) for found in thresholds]
    )


def knowledge_stating(
    *thresholds: pouring.FlowOnsetThreshold,
    elsewhere: Sequence[pouring.FlowOnsetThreshold] = (),
    consulted: Sequence[str] = (CONSULTED,),
) -> FlowOnsetKnowledge:
    """
    :return: Flow onset knowledge over a consulted hypothesis stating the thresholds and
        another, not consulted, stating the ones given as elsewhere.
    """
    return FlowOnsetKnowledge(
        knowledge=StoredHypotheses(
            by_record={
                CONSULTED: events_crossing(*thresholds),
                ELSEWHERE: events_crossing(*elsewhere),
            }
        ),
        consulted_records=lambda: list(consulted),
    )


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
    def test_the_tilt_is_read_in_radians(self):
        flow_onset = knowledge_stating(threshold(STATED_ONSET_DEGREES))
        assert flow_onset.tilt() == radians(STATED_ONSET_DEGREES)

    def test_the_lowest_of_several_thresholds_the_consulted_hypothesis_states_is_read(
        self,
    ):
        flow_onset = knowledge_stating(threshold(90.0), threshold(STATED_ONSET_DEGREES))
        assert flow_onset.tilt() == radians(STATED_ONSET_DEGREES)

    def test_a_threshold_of_a_hypothesis_that_was_not_consulted_is_not_read(self):
        flow_onset = knowledge_stating(threshold(90.0), elsewhere=[threshold(30.0)])
        assert flow_onset.tilt() == radians(90.0)

    def test_no_consulted_hypothesis_leaves_the_tilt_unknown(self):
        flow_onset = knowledge_stating(threshold(90.0), consulted=[])
        assert flow_onset.tilt() is None

    def test_a_consulted_hypothesis_stating_no_threshold_leaves_the_tilt_unknown(self):
        assert knowledge_stating().tilt() is None

    def test_a_flow_onset_crossing_no_threshold_is_ignored(self):
        flow_onset = FlowOnsetKnowledge(
            knowledge=StoredHypotheses(
                by_record={
                    CONSULTED: StatedHypothesis(events=[pouring.FlowOnset()]),
                }
            ),
            consulted_records=lambda: [CONSULTED],
        )
        assert flow_onset.tilt() is None

    def test_a_threshold_stating_no_range_is_ignored(self):
        flow_onset = knowledge_stating(threshold(None), threshold(STATED_ONSET_DEGREES))
        assert flow_onset.tilt() == radians(STATED_ONSET_DEGREES)

    def test_a_range_in_a_unit_that_is_no_angle_is_ignored(self):
        flow_onset = knowledge_stating(
            threshold(1.0, unit=core.MeasurementUnit.DEGREE_PER_SECOND)
        )
        assert flow_onset.tilt() is None


# %% the check


class TestPourFlows:
    def answer(self, tilt_angle: float) -> PourAnswer:
        return PourAnswer(arm=HandSide.RIGHT, tilt_angle=tilt_angle)

    def check_stating(self):
        check = checking([LANDS])
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

    def test_a_pour_below_the_onset_is_never_kept_as_a_last_resort(self):
        check = self.check_stating()
        with pytest.raises(OutcomeFellShort) as refusal:
            check.check([self.answer(SHALLOW_TILT)])
        assert refusal.value.fallback_rank is None

    def test_without_a_stated_onset_a_shallow_pour_is_accepted(self):
        check = checking([LANDS])
        check.check([self.answer(SHALLOW_TILT)])
        assert len(check.trials) == 1

    def test_a_stated_onset_leaves_the_tilt_unchecked_when_knowledge_states_none(self):
        check = checking([LANDS])
        check.flow_onset = knowledge_stating()
        check.check([self.answer(SHALLOW_TILT)])
        assert len(check.trials) == 1
