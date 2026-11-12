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

import cramph.orm.ormatic_interface  # type: ignore  # noqa: F401
from coraplex.orm.ormatic_interface import Base
from coraplex.plans.context_extensions import RobotAccess, StatementGrounding
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from coraplex.testing import setup_world
from cramph.composites import Sequence
from cramph.context import StatechartContext
from cramph.statechart import Statechart
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from krrood_spatial_math.spatial_types import Pose
from sqlalchemy.orm import sessionmaker

DEFAULT_DATABASE_PATH = Path("pickup_neem.sqlite")


def run_plan() -> Statechart:
    """
    Execute the torso, arm, navigation, and milk pick-up plan in simulation.

    :return: The statechart the plan was performed as.
    """
    world = setup_world()
    robot = PR2.from_world(world)
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    pick_up_pose = Pose.from_xyz_rpy(x=2.0, y=2.0, reference_frame=world.root)
    executor = KinematicallySimulatedPlanExecutor(
        StatechartContext.with_extensions(
            world, [RobotAccess(robot), StatementGrounding()]
        )
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(
        Sequence(
            nodes=[
                MoveTorsoAction(torso_state=TorsoState.HIGH),
                ParkArmsAction(arms=robot.all_arms),
                NavigateAction(target_location=pick_up_pose),
                PickUpAction(grasp=milk.grasp_candidates()[0], arm=robot.right_arm),
            ]
        )
    )
    executor.compile(statechart)
    executor.execute()
    return statechart


def persist_neem(statechart: Statechart, database_path: Path) -> Path:
    """
    Persist the statechart of an executed plan to SQLite and write a matching SQL dump.
    """
    database_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{database_path.resolve()}")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(engine)()
    try:
        neem = to_dao(statechart)
        session.add(neem)
        session.commit()
        database_id = neem.database_id
    finally:
        session.close()
        engine.dispose()

    sql_path = database_path.with_suffix(".sql")
    with (
        sqlite3.connect(database_path) as connection,
        sql_path.open("w", encoding="utf-8") as sql_file,
    ):
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
