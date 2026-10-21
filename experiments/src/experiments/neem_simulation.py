"""
Run a short PR2 simulation and save its execution as a SQL NEEM.

Run with the ``experiments`` package importable::

    python -m experiments.neem_simulation
    python -m experiments.neem_simulation --database /tmp/pickup-neem.sqlite

The command writes both the SQLite database and a plain SQL dump beside it.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import coraplex.orm.ormatic_interface  # type: ignore  # noqa: F401
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot
from coraplex.orm.ormatic_interface import Base
from coraplex.plans.factories import sequential
from coraplex.plans.plan import Plan
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from coraplex.testing import setup_world
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types.spatial_types import Pose
from sqlalchemy.orm import sessionmaker

DEFAULT_DATABASE_PATH = Path("pickup_neem.sqlite")


def run_plan() -> Plan:
    """
    Execute the torso, arm, navigation, and milk pick-up plan in simulation.
    """
    world = setup_world()
    robot = PR2.from_world(world)
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    context = Context(world=world, robot=robot, _debug=False, ros_node=None)
    context.evaluate_conditions = False
    pick_up_pose = Pose.from_xyz_rpy(2.0, 2.0, reference_frame=world.root)

    plan = sequential(
        [
            MoveTorsoAction(TorsoState.HIGH),
            ParkArmsAction(Arms.BOTH),
            NavigateAction(pick_up_pose),
            PickUpAction(
                milk,
                Arms.RIGHT,
                GraspDescription(
                    ApproachDirection.FRONT,
                    VerticalAlignment.TOP,
                    robot.right_arm.end_effector,
                ),
            ),
        ],
        context=context,
    ).plan
    with simulated_robot:
        plan.perform()
    return plan


def persist_neem(plan: Plan, database_path: Path) -> Path:
    """
    Persist the executed plan to SQLite and write a matching SQL dump.
    """
    database_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{database_path.resolve()}")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(engine)()
    try:
        neem = to_dao(plan)
        session.add(neem)
        session.commit()
        database_id = neem.database_id
    finally:
        session.close()
        engine.dispose()

    sql_path = database_path.with_suffix(".sql")
    with sqlite3.connect(database_path) as connection, sql_path.open(
        "w", encoding="utf-8"
    ) as sql_file:
        sql_file.writelines(f"{statement}\n" for statement in connection.iterdump())
    print(f"Persisted NEEM database_id={database_id} to {database_path}")
    print(f"Exported SQL dump to {sql_path}")
    return sql_path


def parse_arguments() -> argparse.Namespace:
    """
    Read the destination SQLite database path from the command line.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help="SQLite output path; the SQL dump is written beside it.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Run the simulated plan, persist its NEEM, and export SQL.
    """
    arguments = parse_arguments()
    persist_neem(run_plan(), arguments.database)


if __name__ == "__main__":
    main()
