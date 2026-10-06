from __future__ import annotations

from dataclasses import fields

from llm_reasoner.knowledge.generation.schema import derive_schema
from llm_reasoner.knowledge.microworlds.flanagan.constraints import SUITE
from llm_reasoner.knowledge.microworlds.flanagan.core import ScientificStatus
from llm_reasoner.knowledge.microworlds.flanagan.gaze import (
    GazeControlProfile,
    GazeEpisode,
    GazeMode,
    GazeTarget,
    MultimodalAlignment,
)
from llm_reasoner.knowledge.microworlds.flanagan.learning import (
    LearningProcessKind,
    LearningStage,
    SensorimotorMap,
    SensorimotorMapUpdate,
    VisuomotorLearningEpisode,
    VisuomotorLearningProfile,
)
from llm_reasoner.knowledge.microworlds.flanagan.observation import (
    ActionObservationProfile,
    ActorObserverPredictionMatch,
    HandVisibility,
    ObservedActionPrediction,
    Observer,
)
from llm_reasoner.knowledge.microworlds.flanagan.perception import (
    AttenuationControlRelationship,
    EventOrigin,
    SensoryAttenuation,
    SensoryAttenuationProfile,
)
from llm_reasoner.knowledge.microworlds.registry import (
    FLANAGAN_CORE,
    FLANAGAN_GAZE,
    FLANAGAN_LEARNING,
    FLANAGAN_OBJECT_DYNAMICS,
    FLANAGAN_OBSERVATION,
    FLANAGAN_PERCEPTION,
    FLANAGAN_POURING,
    POURING_PROFILE,
    Capability,
    select_modules,
)

# %% gaze control


class TestGazeControlCoverage:
    """
    Gaze control represents targeting, timing, fixation, and alignment.
    """

    def test_gaze_episode_exposes_each_control_component(self) -> None:
        assert {
            "mode",
            "target",
            "lead_time",
            "fixation_until",
            "monitors_contact",
            "scientific_status",
        } <= {field.name for field in fields(GazeEpisode)}
        assert set(GazeMode) == {GazeMode.PROACTIVE, GazeMode.REACTIVE}

    def test_gaze_target_and_multimodal_alignment_are_structured(self) -> None:
        assert {"contact_location", "target_for_phase"} <= {
            field.name for field in fields(GazeTarget)
        }
        assert {
            "control_point",
            "visual_observation",
            "aligned_observations",
            "scientific_status",
        } <= {field.name for field in fields(MultimodalAlignment)}
        assert "episodes" in {field.name for field in fields(GazeControlProfile)}


# %% action observation


class TestActionObservationCoverage:
    """
    Observation keeps actor, observer, access, prediction, and inference explicit.
    """

    def test_observation_prediction_exposes_observer_evidence(self) -> None:
        assert Observer.__name__ != "Actor"
        assert set(HandVisibility) == {
            HandVisibility.VISIBLE,
            HandVisibility.PARTIALLY_OCCLUDED,
            HandVisibility.HIDDEN,
        }
        assert {
            "actor",
            "observer",
            "observed_hand_visibility",
            "predicted_contact_site",
            "available_modalities",
            "inferred_hidden_properties",
            "observer_gaze",
        } <= {field.name for field in fields(ObservedActionPrediction)}
        assert "predictions" in {
            field.name for field in fields(ActionObservationProfile)
        }

    def test_actor_observer_matching_remains_an_author_hypothesis(self) -> None:
        matching = ActorObserverPredictionMatch(
            label="Covert actor-observer matching",
            scientific_status=ScientificStatus.EMPIRICAL_FINDING,
        )

        violations = SUITE.evaluate((matching,))

        assert any(violation.check == "OBS-C60" for violation in violations)


# %% visuomotor learning


class TestVisuomotorLearningCoverage:
    """
    Learning distinguishes stages, processes, maps, and map updates.
    """

    def test_learning_vocabulary_covers_stage_and_process_distinctions(self) -> None:
        assert set(LearningStage) == {
            LearningStage.EXPLORATION,
            LearningStage.SKILL_ACQUISITION,
            LearningStage.SKILL_REFINEMENT,
        }
        assert set(LearningProcessKind) == {
            LearningProcessKind.NOVEL_RULE_DISCOVERY,
            LearningProcessKind.EXISTING_MAPPING_ADAPTATION,
        }
        assert {"input_frame", "output_frame", "mapping_rule"} <= {
            field.name for field in fields(SensorimotorMap)
        }
        assert {"updates_map", "from_stage", "to_stage"} <= {
            field.name for field in fields(SensorimotorMapUpdate)
        }
        assert {"stage", "gaze_mode", "learning_process", "sensorimotor_map"} <= {
            field.name for field in fields(VisuomotorLearningEpisode)
        }
        assert "episodes" in {field.name for field in fields(VisuomotorLearningProfile)}


# %% sensory attenuation


class TestSensoryAttenuationCoverage:
    """
    Attenuation remains a perceptual phenomenon with an unresolved control link.
    """

    def test_attenuation_exposes_origin_timing_prediction_and_response(self) -> None:
        assert set(EventOrigin) == {
            EventOrigin.SELF_GENERATED,
            EventOrigin.EXTERNAL,
        }
        assert {
            "predicted_contact",
            "event_origin",
            "attenuation_window",
            "attenuated_response",
            "scientific_status",
        } <= {field.name for field in fields(SensoryAttenuation)}
        assert "attenuations" in {
            field.name for field in fields(SensoryAttenuationProfile)
        }

    def test_attenuation_control_link_must_remain_an_open_question(self) -> None:
        relationship = AttenuationControlRelationship(
            label="Shared prediction mechanism",
            scientific_status=ScientificStatus.SUPPORTED_INTERPRETATION,
        )

        violations = SUITE.evaluate((relationship,))

        assert any(violation.check == "PERCEPTION-C64" for violation in violations)


# %% optional module selection


class TestOptionalScientificModules:
    """
    Scientific extensions load only when their capabilities are requested.
    """

    def test_pouring_does_not_load_unrequested_phase_seven_modules(self) -> None:
        selection = select_modules(POURING_PROFILE.task)

        assert FLANAGAN_POURING in selection.modules
        assert FLANAGAN_GAZE not in selection.modules
        assert FLANAGAN_OBSERVATION not in selection.modules
        assert FLANAGAN_LEARNING not in selection.modules
        assert FLANAGAN_PERCEPTION not in selection.modules

    def test_requested_observation_capability_closes_over_gaze(self) -> None:
        selection = select_modules(
            POURING_PROFILE.task,
            additional_capabilities=frozenset({Capability.ACTION_OBSERVATION}),
        )

        assert FLANAGAN_OBSERVATION in selection.modules
        assert FLANAGAN_GAZE in selection.modules
        assert FLANAGAN_LEARNING not in selection.modules
        assert FLANAGAN_PERCEPTION not in selection.modules


# %% generation schemas


class TestExtendedGenerationSchemas:
    """
    Each optional scientific module exposes an independently selectable root.
    """

    def test_each_phase_seven_root_has_required_generation_content(self) -> None:
        cases = (
            (
                (FLANAGAN_CORE.name, FLANAGAN_GAZE.name),
                GazeControlProfile,
                "episodes",
            ),
            (
                (
                    FLANAGAN_CORE.name,
                    FLANAGAN_GAZE.name,
                    FLANAGAN_OBSERVATION.name,
                ),
                ActionObservationProfile,
                "predictions",
            ),
            (
                (
                    FLANAGAN_CORE.name,
                    FLANAGAN_OBJECT_DYNAMICS.name,
                    FLANAGAN_GAZE.name,
                    FLANAGAN_LEARNING.name,
                ),
                VisuomotorLearningProfile,
                "episodes",
            ),
            (
                (FLANAGAN_CORE.name, FLANAGAN_PERCEPTION.name),
                SensoryAttenuationProfile,
                "attenuations",
            ),
        )

        for modules, root_type, required_field in cases:
            schema = derive_schema(modules=modules, root=root_type.__name__)
            root_schema = schema.get(root_type.__name__)

            assert root_schema is not None
            assert required_field in {
                field.name for field in root_schema.required_fields
            }
