"""
Tests that the situation an action is performed in is said in the pouring frame's terms:

the material poured and why the pour ends are the pouring frame's, the phase structure
and the frame of pouring both accept them, and each value defines a variant of the
frame.
"""

from __future__ import annotations

import pytest
from typing_extensions import Optional, get_type_hints

from llm_reasoner.knowledge.ontologies.action_perspectives import (
    action_core,
    pouring_frame,
)
from llm_reasoner.knowledge.ontologies.described_actions import DescribedActions
from llm_reasoner.knowledge.ontologies.predictive_manipulation import pouring
from llm_reasoner.knowledge.vocabulary.situation import SituationFeatures

POURING_ACTION = "coraplex.robot_plans.actions.composite.tool_based.PouringAction"
"""
The designator whose knowledge the mapping kept as data names.
"""

SITUATION_FIELDS = [
    ("has_material_regime", pouring_frame.MaterialRegime),
    ("has_termination_reason", pouring_frame.TransferTerminationReason),
]
"""
Each situation field with the core's enumeration of its values.
"""

# %% the terms


class TestTheTerms:
    @pytest.mark.parametrize("module", [action_core, pouring])
    def test_the_situation_terms_are_the_pouring_frames_alone(self, module):
        assert {"MaterialRegime", "TransferTerminationReason"} <= set(
            vars(pouring_frame)
        )
        assert {"MaterialRegime", "TransferTerminationReason"} & set(
            vars(module)
        ) == set()

    @pytest.mark.parametrize("field_name, values", SITUATION_FIELDS)
    @pytest.mark.parametrize(
        "structure", [pouring.PouringPhaseStructure, pouring_frame.PouringFrame]
    )
    def test_both_families_state_the_situation_in_the_pouring_frames_terms(
        self, structure, field_name, values
    ):
        assert get_type_hints(structure)[field_name] == Optional[values]


# %% the features


class TestTheFramesFeatures:
    def test_the_frame_accepts_the_features_the_phase_structure_does(self):
        features = SituationFeatures()
        assert [
            feature.field_name for feature in features.of(pouring_frame.PouringFrame)
        ] == [
            feature.field_name for feature in features.of(pouring.PouringPhaseStructure)
        ]

    @pytest.mark.parametrize("field_name, values", SITUATION_FIELDS)
    def test_every_value_defines_a_variant_of_the_frame(self, field_name, values):
        (feature,) = [
            feature
            for feature in SituationFeatures().of(pouring_frame.PouringFrame)
            if feature.field_name == field_name
        ]
        assert feature.values == tuple(values)
        assert set(feature.variants) == set(values)
        for variant in feature.variants.values():
            assert issubclass(variant, pouring_frame.PouringFrame)


# %% asking for the frame of a situation


class TestFrameQuery:
    def test_the_frame_is_asked_for_with_the_situation(self):
        described = DescribedActions().describing(POURING_ACTION)
        query = described.frame_query(
            {"has_material_regime": pouring_frame.MaterialRegime.GRANULAR}
        )
        frame = query.construct_instance()
        assert frame.has_material_regime is pouring_frame.MaterialRegime.GRANULAR
        assert frame.frames_action == described.concept
