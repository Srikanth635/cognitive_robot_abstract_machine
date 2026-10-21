"""
Run a varied batch of PR2 action plans and persist every execution as a NEEM.

Run with the ``experiments`` package importable::

    python -m experiments.neem_action_sweep --count 100
    python -m experiments.neem_action_sweep --count 100 --database /tmp/action_sweep.sqlite

Each trial is saved as a separate plan in one SQLite database. A SQL dump and a CSV
manifest are written beside that database.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import coraplex.orm.ormatic_interface  # type: ignore  # noqa: F401
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    Arms,
    ApproachDirection,
    DetectionTechnique,
    VerticalAlignment,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot
from coraplex.orm.ormatic_interface import Base
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.composite.facing import FaceAtAction
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.misc import DetectAction
from coraplex.robot_plans.actions.core.navigation import (
    LookAtAction,
    NavigateAction,
    PathPlanningNavigateAction,
)
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    MoveManipulatorAction,
    ParkArmsAction,
    SetGripperAction,
)
from coraplex.testing import setup_world
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Cereal, Milk
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from sqlalchemy.orm import Session, sessionmaker

DEFAULT_DATABASE_PATH = Path("action_sweep_neems.sqlite")
DEFAULT_TRIAL_COUNT = 100
ARMS = (Arms.LEFT, Arms.RIGHT, Arms.BOTH)
PICKUP_ARMS = (Arms.LEFT, Arms.RIGHT)
TORSO_STATES = (TorsoState.HIGH, TorsoState.MID, TorsoState.LOW)
GRIPPER_STATES = (GripperState.OPEN, GripperState.CLOSE, GripperState.MEDIUM)
APPROACH_DIRECTIONS = tuple(ApproachDirection)
VERTICAL_ALIGNMENTS = tuple(VerticalAlignment)
NEARBY_POSES = (
    (1.45, 2.50, 0.0),
    (1.50, 2.50, 0.15),
    (1.55, 2.50, -0.15),
    (1.50, 2.45, 0.30),
    (1.50, 2.55, -0.30),
)
TARGET_POSES = (
    (2.37, 2.00, 1.05),
    (2.37, 1.80, 1.05),
    (2.30, 2.10, 1.05),
    (2.45, 1.90, 1.05),
)


class ActionKind(StrEnum):
    """
    Action types included in the apartment-scene sweep.
    """

    TORSO = "torso"
    PARK_ARMS = "park_arms"
    GRIPPER = "gripper"
    NAVIGATE = "navigate"
    LOOK_AT = "look_at"
    FACE_AT = "face_at"
    PICK_UP = "pick_up"
    TRANSPORT = "transport"
    DETECT = "detect"
    PATH_NAVIGATE = "path_navigate"
    MOVE_MANIPULATOR = "move_manipulator"


class TrialOutcome(StrEnum):
    """
    Execution result recorded in the batch manifest.
    """

    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class SimulationCase:
    """
    One action instance and the label used to describe its parameter setting.
    """

    action: ActionDescription
    """
    The CRAM action executed in this trial.
    """

    kind: ActionKind
    """
    The action family used to build the trial.
    """

    parameters: str
    """
    A readable description of the action parameters.
    """


@dataclass
class TrialRecord:
    """
    The outcome metadata for one persisted plan.
    """

    trial_number: int
    """
    One-based position in the requested batch.
    """

    action_kind: ActionKind
    """
    The action family recorded by the plan.
    """

    parameters: str
    """
    The selected action parameter values.
    """

    outcome: TrialOutcome
    """
    Whether the plan completed or raised during execution.
    """

    failure: str
    """
    The captured failure description, or an empty string on success.
    """


def build_case(
    trial_number: int, world: World, robot: PR2, milk: Milk, cereal: Cereal
) -> SimulationCase:
    """
    Build a deterministic action and parameter combination for a trial.
    """
    action_kinds = tuple(ActionKind)
    kind = action_kinds[(trial_number - 1) % len(action_kinds)]
    variation = (trial_number - 1) // len(action_kinds)
    target_pose = TARGET_POSES[variation % len(TARGET_POSES)]
    target = Pose.from_xyz_rpy(*target_pose, reference_frame=world.root)

    match kind:
        case ActionKind.TORSO:
            state = TORSO_STATES[variation % len(TORSO_STATES)]
            return SimulationCase(MoveTorsoAction(state), kind, f"state={state.name}")
        case ActionKind.PARK_ARMS:
            arm = ARMS[variation % len(ARMS)]
            return SimulationCase(ParkArmsAction(arm), kind, f"arm={arm.name}")
        case ActionKind.GRIPPER:
            combination_count = len(ARMS) * len(GRIPPER_STATES)
            combination = variation % combination_count
            arm = ARMS[combination // len(GRIPPER_STATES)]
            state = GRIPPER_STATES[combination % len(GRIPPER_STATES)]
            return SimulationCase(
                SetGripperAction(gripper=arm, motion=state),
                kind,
                f"arm={arm.name},state={state.name}",
            )
        case ActionKind.NAVIGATE:
            pose = NEARBY_POSES[variation % len(NEARBY_POSES)]
            keep_joint_states = variation % 2 == 0
            location = Pose.from_xyz_rpy(*pose, reference_frame=world.root)
            return SimulationCase(
                NavigateAction(location, keep_joint_states),
                kind,
                f"x={pose[0]},y={pose[1]},yaw={pose[2]},keep_joint_states={keep_joint_states}",
            )
        case ActionKind.LOOK_AT:
            return SimulationCase(LookAtAction(target), kind, f"target={target_pose}")
        case ActionKind.FACE_AT:
            return SimulationCase(FaceAtAction(target), kind, f"target={target_pose}")
        case ActionKind.PICK_UP:
            arm = PICKUP_ARMS[variation % len(PICKUP_ARMS)]
            object_annotation = (milk, cereal)[variation % 2]
            end_effector = (
                robot.left_arm.end_effector
                if arm is Arms.LEFT
                else robot.right_arm.end_effector
            )
            grasp = GraspDescription(
                APPROACH_DIRECTIONS[variation % len(APPROACH_DIRECTIONS)],
                VERTICAL_ALIGNMENTS[variation % len(VERTICAL_ALIGNMENTS)],
                end_effector,
                rotate_gripper=variation % 2 == 0,
                manipulation_offset=(0.03, 0.07, 0.10)[variation % 3],
            )
            return SimulationCase(
                PickUpAction(object_annotation, arm, grasp),
                kind,
                f"object={type(object_annotation).__name__},arm={arm.name},"
                f"approach={grasp.approach_direction.name},"
                f"vertical={grasp.vertical_alignment.name},"
                f"rotate={grasp.rotate_gripper},offset={grasp.manipulation_offset}",
            )
        case ActionKind.TRANSPORT:
            arm = PICKUP_ARMS[variation % len(PICKUP_ARMS)]
            object_annotation = (milk, cereal)[variation % 2]
            return SimulationCase(
                TransportAction(object_annotation, target, arm),
                kind,
                f"object={type(object_annotation).__name__},arm={arm.name},"
                f"target={target_pose}",
            )
        case ActionKind.DETECT:
            object_annotation = (milk, cereal)[variation % 2]
            return SimulationCase(
                DetectAction(
                    DetectionTechnique.TYPES,
                    object_sem_annotation=type(object_annotation),
                    accept_first_if_multiple=variation % 2 == 0,
                ),
                kind,
                f"object_type={type(object_annotation).__name__},"
                f"accept_first={variation % 2 == 0}",
            )
        case ActionKind.PATH_NAVIGATE:
            pose = NEARBY_POSES[variation % len(NEARBY_POSES)]
            location = Pose.from_xyz_rpy(*pose, reference_frame=world.root)
            return SimulationCase(
                PathPlanningNavigateAction(location),
                kind,
                f"target={pose}",
            )
        case ActionKind.MOVE_MANIPULATOR:
            arm = PICKUP_ARMS[variation % len(PICKUP_ARMS)]
            end_effector = (
                robot.left_arm.end_effector
                if arm is Arms.LEFT
                else robot.right_arm.end_effector
            )
            x, y, _ = NEARBY_POSES[variation % len(NEARBY_POSES)]
            pose = Pose.from_xyz_rpy(x, y, 0.95, reference_frame=world.root)
            return SimulationCase(
                MoveManipulatorAction(
                    target_pose=pose,
                    end_effector=end_effector,
                    allow_gripper_collision=True,
                ),
                kind,
                f"arm={arm.name},target={pose.to_position()}",
            )


def prepare_scene() -> tuple[World, PR2, Context, Milk, Cereal]:
    """
    Build the apartment scene and the PR2 context shared by the action sweep.
    """
    world = setup_world()
    with world.modify_world():
        world.add_semantic_annotation(
            Cereal(root=world.get_body_by_name("breakfast_cereal.stl"))
        )
    robot = PR2.from_world(world)
    context = Context(world=world, robot=robot, _debug=False, ros_node=None)
    context.evaluate_conditions = False
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    cereal = world.get_semantic_annotations_by_type(Cereal)[0]
    return world, robot, context, milk, cereal


def persist_plan(plan: Plan, session: Session) -> int:
    """
    Persist one executed plan and return its NEEM database identifier.
    """
    neem = to_dao(plan)
    session.add(neem)
    session.commit()
    return neem.database_id


def export_database(database_path: Path) -> tuple[Path, Path]:
    """
    Write SQL and CSV exports beside the batch database.
    """
    sql_path = database_path.with_suffix(".sql")
    with (
        sqlite3.connect(database_path) as connection,
        sql_path.open("w", encoding="utf-8") as sql_file,
    ):
        sql_file.writelines(f"{statement}\n" for statement in connection.iterdump())

    manifest_path = database_path.with_suffix(".csv")
    return sql_path, manifest_path


def run_batch(trial_count: int, database_path: Path) -> list[TrialRecord]:
    """
    Run each action case and persist its execution, including plan failures.
    """
    database_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{database_path.resolve()}")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(engine)()
    records: list[TrialRecord] = []
    try:
        world, robot, context, milk, cereal = prepare_scene()
        for trial_number in range(1, trial_count + 1):
            case = build_case(trial_number, world, robot, milk, cereal)
            plan = sequential([case.action], context=context).plan
            outcome = TrialOutcome.SUCCEEDED
            failure = ""
            try:
                with simulated_robot:
                    plan.perform()
            except Exception as exception:
                outcome = TrialOutcome.FAILED
                failure = f"{type(exception).__name__}: {exception}"

            database_id = persist_plan(plan, session)
            record = TrialRecord(
                trial_number=trial_number,
                action_kind=case.kind,
                parameters=case.parameters,
                outcome=outcome,
                failure=failure,
            )
            records.append(record)
            print(
                f"[{trial_number}/{trial_count}] {case.kind.value} "
                f"{outcome.value}; database_id={database_id}"
            )
    finally:
        session.close()
        engine.dispose()

    sql_path, manifest_path = export_database(database_path)
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(
            ["trial_number", "action_kind", "parameters", "outcome", "failure"]
        )
        writer.writerows(
            [
                record.trial_number,
                record.action_kind.value,
                record.parameters,
                record.outcome.value,
                record.failure,
            ]
            for record in records
        )
    print(f"Persisted {len(records)} NEEMs to {database_path}")
    print(f"Exported SQL to {sql_path}")
    print(f"Wrote trial manifest to {manifest_path}")
    return records


def parse_arguments() -> argparse.Namespace:
    """
    Read the trial count and output database path.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=DEFAULT_TRIAL_COUNT)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    arguments = parser.parse_args()
    if arguments.count < 1:
        parser.error("--count must be at least 1")
    return arguments


def main() -> None:
    """
    Run the requested batch and export its persisted NEEMs.
    """
    arguments = parse_arguments()
    run_batch(arguments.count, arguments.database)


if __name__ == "__main__":
    main()
