"""
Tests of comparing the resolver with and without the action frame: what is recorded of a
run, from a resolved designator with its pours and events or from a resolution in the
kitchen, and how the arms are summarised.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from llm_reasoner.clients.usage import SONNET_5_5_PRICES
from llm_reasoner.knowledge.access import (
    ActionPerspectiveFile,
    Citation,
    HypothesisStore,
    KnowledgeToolName,
)
from llm_reasoner.resolver.grounds import GroundKey

from coraplex.datastructures.enums import Arms
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import PouringCup
from semantic_digital_twin.world_description.world_entity import Body

from cram_viz.resolution.resolver import (
    ResolvedAction,
    ResolvedDesignator,
    ResolvedSlot,
    ShownGround,
)
from cram_viz.resolution.trial import OutcomeKey, TriedPour
from experiments.frame_comparison.__main__ import main as summarise_directory
from experiments.frame_comparison.output import ComparisonOutput
from experiments.frame_comparison.records import (
    ComparisonArm,
    Measurement,
    RunRecord,
    chosen_values,
    summarise,
    written_object,
)
from experiments.llm_reasoner_kitchen.resolution import Resolution

from ...llm_reasoner_test.test_resolver_knowledge import (
    tilt_of,
    world_backend_offering_the_frame,
)
from ...llm_reasoner_test.world_mimics import body_with_geometry

CASE = "pour_geometry_open"
"""
The case the scripted runs are of.
"""

FRAME_RECORD = "frame-record"
"""
The record of the frame hypothesis the scripted runs offering the frame consulted.
"""

# %% helpers


def knowledge_ground(cited: Citation) -> ShownGround:
    """
    :return: A ground resting on a cited knowledge fragment, named as cram_viz names it.
    """
    return ShownGround(GroundKey.KNOWLEDGE, cited.fragment_name, cited.iri)


def tried(lands: bool, distance: float) -> TriedPour:
    """
    :return: A pour tried in the simulation, landing or not, its mouth the given
        distance from the opening's centre.
    """
    return TriedPour(
        values={"pour_side_offset": 0.0},
        outcome={
            OutcomeKey.LANDS_IN_OPENING: lands,
            OutcomeKey.DISTANCE_FROM_OPENING_CENTRE: distance,
        },
    )


def resolved(trials: list[TriedPour], grounds: list[ShownGround]) -> ResolvedDesignator:
    """
    :return: A resolution of the pour, the arm and the tilt chosen and the containers
        given, with the pours it tried and what it rests on.
    """
    return ResolvedDesignator(
        query="the(PouringAction)(...)",
        designator="PouringAction",
        actions=[
            ResolvedAction(
                slots=[
                    ResolvedSlot(
                        "source_container", "cup", given=True, serialized="cup"
                    ),
                    ResolvedSlot("arm", "RIGHT", given=False, serialized="RIGHT"),
                    ResolvedSlot("tilt_angle", "1.85", given=False, serialized=1.85),
                ],
                why="",
                grounds=grounds,
            )
        ],
        situation={},
        record_id=None,
        cost=0.1,
        run="a-run",
        trials=trials,
        perspective_record_ids=["core-record", FRAME_RECORD],
    )


def events_file(directory: Path, tools: list[str]) -> Path:
    """
    :return: A run's events, the model calling the given tools in turn.
    """
    path = directory / "events.jsonl"
    path.write_text(
        "\n".join(json.dumps({"event": "ToolCalled", "tool": tool}) for tool in tools)
    )
    return path


def pour_record(tmp_path, arm, trials, grounds=(), tools=()) -> RunRecord:
    """
    :return: The record of a scripted run of the pour.
    """
    return RunRecord.of_designator(
        resolved(trials, list(grounds)),
        events_file(tmp_path, list(tools)),
        case=CASE,
        arm=arm,
        run=0,
    )


# %% the arms


class TestArms:
    def test_the_arm_without_the_frame_consults_no_perspective(self):
        assert ComparisonArm.WITHOUT_FRAME.perspectives == ()

    def test_only_the_arm_with_the_prediction_predicts_pours(self):
        assert [arm for arm in ComparisonArm if arm.predicts_pours] == [
            ComparisonArm.WITH_PREDICTION
        ]

    def test_the_arm_with_the_prediction_consults_no_perspective(self):
        assert ComparisonArm.WITH_PREDICTION.perspectives == ()

    def test_the_arm_with_the_frame_consults_the_action_frame(self):
        assert ComparisonArm.WITH_FRAME.perspectives == (
            ActionPerspectiveFile.ACTION_FRAME,
        )

    def test_the_arm_with_the_image_schemas_consults_only_them(self):
        assert ComparisonArm.WITH_IMAGE_SCHEMAS.perspectives == (
            ActionPerspectiveFile.IMAGE_SCHEMAS,
        )

    def test_the_arm_with_the_failures_consults_only_them(self):
        assert ComparisonArm.WITH_FAILURES.perspectives == (
            ActionPerspectiveFile.FAILURES,
        )

    def test_the_arm_with_both_consults_the_image_schemas_and_the_failures(self):
        assert ComparisonArm.WITH_IMAGE_SCHEMAS_AND_FAILURES.perspectives == (
            ActionPerspectiveFile.IMAGE_SCHEMAS,
            ActionPerspectiveFile.FAILURES,
        )

    def test_no_new_arm_predicts_pours(self):
        assert [arm for arm in ComparisonArm if arm.predicts_pours] == [
            ComparisonArm.WITH_PREDICTION
        ]


# %% recording a run of the pour


class TestPourRecord:
    def test_the_values_chosen_are_recorded_and_the_given_ones_left_out(self, tmp_path):
        record = pour_record(tmp_path, ComparisonArm.WITH_FRAME, [tried(True, 0.01)])
        assert record.values == {"arm": "RIGHT", "tilt_angle": "1.85"}

    def test_a_first_pour_landing_is_a_first_try_success(self, tmp_path):
        record = pour_record(tmp_path, ComparisonArm.WITH_FRAME, [tried(True, 0.01)])
        assert record.first_try_success is True

    def test_a_first_pour_missing_is_no_first_try_success(self, tmp_path):
        record = pour_record(
            tmp_path,
            ComparisonArm.WITH_FRAME,
            [tried(False, 0.188), tried(True, 0.005)],
        )
        assert record.first_try_success is False
        assert [trial.distance for trial in record.trials] == [0.188, 0.005]

    def test_a_ground_citing_a_consulted_perspective_cites_the_frame(self, tmp_path):
        record = pour_record(
            tmp_path,
            ComparisonArm.WITH_FRAME,
            [tried(True, 0.01)],
            grounds=[
                knowledge_ground(Citation(FRAME_RECORD, "lip_over_opening")),
                knowledge_ground(Citation("phase-record", "ph_tilt")),
                ShownGround(GroundKey.WORLD, "Body#1 (bowl.stl)", "Body#1"),
            ],
        )
        assert [ground.cites_perspective for ground in record.grounds] == [
            True,
            False,
            False,
        ]

    def test_only_the_knowledge_tools_called_are_recorded(self, tmp_path):
        record = pour_record(
            tmp_path,
            ComparisonArm.WITH_FRAME,
            [tried(True, 0.01)],
            tools=[
                "list_kinds",
                KnowledgeToolName.KNOWLEDGE_OVERVIEW,
                KnowledgeToolName.DESCRIBE_FRAGMENT,
            ],
        )
        assert record.knowledge_tools == [
            KnowledgeToolName.KNOWLEDGE_OVERVIEW,
            KnowledgeToolName.DESCRIBE_FRAGMENT,
        ]


# %% summarising the arms


class TestSummary:
    def test_each_arm_of_a_case_is_summarised_apart(self, tmp_path):
        records = [
            pour_record(tmp_path, ComparisonArm.WITHOUT_FRAME, [tried(False, 0.188)]),
            pour_record(tmp_path, ComparisonArm.WITH_FRAME, [tried(True, 0.01)]),
        ]
        assert [(summary.arm, summary.runs) for summary in summarise(records)] == [
            (ComparisonArm.WITHOUT_FRAME, 1),
            (ComparisonArm.WITH_FRAME, 1),
        ]

    def test_an_arm_counts_its_first_try_successes_and_trials(self, tmp_path):
        records = [
            pour_record(tmp_path, ComparisonArm.WITH_FRAME, [tried(True, 0.01)]),
            pour_record(
                tmp_path,
                ComparisonArm.WITH_FRAME,
                [tried(False, 0.188), tried(True, 0.005)],
            ),
        ]
        (summary,) = summarise(records)
        assert summary.measurement is Measurement.SIMULATED_POUR
        assert summary.first_try_successes == 1
        assert summary.mean_trials == 1.5

    def test_an_arm_counts_the_runs_resting_on_the_frame(self, tmp_path):
        records = [
            pour_record(
                tmp_path,
                ComparisonArm.WITH_FRAME,
                [tried(True, 0.01)],
                grounds=[knowledge_ground(Citation(FRAME_RECORD, "lip_pos"))],
            ),
            pour_record(tmp_path, ComparisonArm.WITH_FRAME, [tried(True, 0.01)]),
        ]
        (summary,) = summarise(records)
        assert summary.runs_resting_on_a_perspective == 1


# %% recording a run in the kitchen


class TestKitchenRecord:
    def test_a_kitchen_run_records_the_plain_values_chosen(self, tmp_path):
        backend = world_backend_offering_the_frame(HypothesisStore(directory=tmp_path))
        (action,) = tilt_of(body_with_geometry()).evaluate(backend=backend)
        record = RunRecord.of_kitchen(
            Resolution(actions=[action], backend=backend),
            case="K1",
            arm=ComparisonArm.WITH_FRAME,
            run=0,
            prices=SONNET_5_5_PRICES,
        )
        assert record.values == {"angle": str(action.angle)}
        assert record.first_try_success is None

    def test_a_kitchen_run_costs_what_its_conversations_cost(self, tmp_path):
        backend = world_backend_offering_the_frame(HypothesisStore(directory=tmp_path))
        resolution = Resolution(
            actions=list(tilt_of(body_with_geometry()).evaluate(backend=backend)),
            backend=backend,
        )
        record = RunRecord.of_kitchen(
            resolution,
            case="K1",
            arm=ComparisonArm.WITH_FRAME,
            run=0,
            prices=SONNET_5_5_PRICES,
        )
        assert record.cost == resolution.cost(SONNET_5_5_PRICES)


# %% keeping the records


class TestOutput:
    def test_the_records_kept_are_read_back(self, tmp_path):
        output = ComparisonOutput(directory=tmp_path / "comparison")
        kept = [
            pour_record(tmp_path, ComparisonArm.WITHOUT_FRAME, [tried(False, 0.188)]),
            pour_record(tmp_path, ComparisonArm.WITH_FRAME, [tried(True, 0.01)]),
        ]
        for record in kept:
            output.add(record)
        assert output.records() == kept

    def test_the_summary_written_summarises_the_records_kept(self, tmp_path):
        output = ComparisonOutput(directory=tmp_path / "comparison")
        record = pour_record(tmp_path, ComparisonArm.WITH_FRAME, [tried(True, 0.01)])
        output.add(record)
        output.write_summary()
        assert json.loads(output.summary_path.read_text()) == [
            asdict(summary) for summary in summarise([record])
        ]

    def test_summarising_a_directory_writes_its_summary(self, tmp_path):
        output = ComparisonOutput(directory=tmp_path / "comparison")
        output.add(pour_record(tmp_path, ComparisonArm.WITH_FRAME, [tried(True, 0.01)]))
        summarise_directory([str(output.directory)])
        assert output.summary_path.is_file()


# %% the objects chosen


@dataclass
class PourChoosingObjects:
    """
    A pour whose containers, arm and tilt have been chosen.
    """

    source_container: PouringCup
    target_container: Body
    arm: Arms
    tilt_angle: float


class TestChosenValues:
    def test_a_body_chosen_is_recorded_by_its_name(self):
        assert written_object(Body(name=PrefixedName("bowl.stl"))) == "bowl.stl"

    def test_an_annotated_object_chosen_is_recorded_by_its_kind_and_body(self):
        cup = PouringCup(root=Body(name=PrefixedName("cup.stl")))
        assert written_object(cup) == "PouringCup (cup.stl)"

    def test_every_object_and_value_chosen_is_recorded(self):
        pour = PourChoosingObjects(
            source_container=PouringCup(root=Body(name=PrefixedName("cup.stl"))),
            target_container=Body(name=PrefixedName("bowl.stl")),
            arm=Arms.RIGHT,
            tilt_angle=1.85,
        )
        assert chosen_values(pour) == {
            "source_container": "PouringCup (cup.stl)",
            "target_container": "bowl.stl",
            "arm": "RIGHT",
            "tilt_angle": "1.85",
        }

    def test_a_kitchen_record_keeps_whether_its_case_was_met(self, tmp_path):
        backend = world_backend_offering_the_frame(HypothesisStore(directory=tmp_path))
        (action,) = tilt_of(body_with_geometry()).evaluate(backend=backend)
        record = RunRecord.of_kitchen(
            Resolution(actions=[action], backend=backend),
            case="P3",
            arm=ComparisonArm.WITH_FRAME,
            run=0,
            prices=SONNET_5_5_PRICES,
            meets=False,
        )
        assert record.meets is False
