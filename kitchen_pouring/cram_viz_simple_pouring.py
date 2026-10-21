"""
Run a compact pouring task suitable for cram-viz browser recording.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import coraplex
import numpy as np
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.composite.tool_based import PouringAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from coraplex.testing import setup_world
from coraplex.view_manager import ViewManager
from krrood.entity_query_language.factories import a
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    PouringCup,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Self


# %% Pour alignment
@dataclass(frozen=True)
class PouringAlignment:
    """
    Final spatial relation between the pitcher mouth and bowl rim.
    """

    mouth_position: tuple[float, float, float]
    """
    World position of the center of the pitcher opening.
    """

    bowl_rim_center: tuple[float, float, float]
    """
    World position of the center of the bowl rim.
    """

    @classmethod
    def measure(cls, pitcher: Body, bowl: Body) -> Self:
        """
        Measure the pitcher opening relative to the bowl rim.
        """
        pitcher_bounding_box = pitcher.visual.as_bounding_box_collection_in_frame(
            pitcher
        ).bounding_box()
        bowl_bounding_box = bowl.visual.as_bounding_box_collection_in_frame(
            bowl
        ).bounding_box()
        pitcher_P_mouth = np.array(
            [
                0.5 * (pitcher_bounding_box.min_x + pitcher_bounding_box.max_x),
                0.5 * (pitcher_bounding_box.min_y + pitcher_bounding_box.max_y),
                pitcher_bounding_box.max_z,
                1.0,
            ]
        )
        bowl_P_rim_center = np.array(
            [
                0.5 * (bowl_bounding_box.min_x + bowl_bounding_box.max_x),
                0.5 * (bowl_bounding_box.min_y + bowl_bounding_box.max_y),
                bowl_bounding_box.max_z,
                1.0,
            ]
        )
        world_P_mouth = pitcher.global_transform.to_np() @ pitcher_P_mouth
        world_P_rim_center = bowl.global_transform.to_np() @ bowl_P_rim_center
        return cls(
            mouth_position=tuple(float(value) for value in world_P_mouth[:3]),
            bowl_rim_center=tuple(float(value) for value in world_P_rim_center[:3]),
        )

    @property
    def horizontal_error(self) -> float:
        """
        Return the horizontal distance from the pitcher mouth to the bowl center.
        """
        return float(
            np.linalg.norm(
                np.asarray(self.mouth_position[:2])
                - np.asarray(self.bowl_rim_center[:2])
            )
        )

    @property
    def vertical_clearance(self) -> float:
        """
        Return the pitcher mouth height above the bowl rim.
        """
        return self.mouth_position[2] - self.bowl_rim_center[2]


# %% Compact pouring task
def execute_simple_pouring() -> PouringAlignment:
    """
    Pick up a nearby pitcher and pour it into a nearby bowl.
    """
    world = setup_world()
    robot = PR2.from_world(world)

    coraplex_root = Path(coraplex.__file__).resolve().parents[2]
    object_directory = coraplex_root / "resources" / "objects"
    pitcher_world = STLParser(str(object_directory / "Static_MilkPitcher.stl")).parse()
    bowl_world = STLParser(str(object_directory / "bowl.stl")).parse()

    milk = world.get_body_by_name("milk.stl")
    island_height = float(milk.global_pose.z + milk.combined_mesh.bounds[0][2])
    pitcher_height = island_height - float(
        pitcher_world.root.combined_mesh.bounds[0][2]
    )
    bowl_height = island_height - float(bowl_world.root.combined_mesh.bounds[0][2])

    with world.modify_world():
        robot.root.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1.75,
                y=2.32,
                yaw=-0.19,
                reference_frame=world.root,
            )
        )
        world.merge_world_at_pose(
            pitcher_world,
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=2.37,
                y=2.2,
                z=pitcher_height,
                reference_frame=world.root,
            ),
        )
        world.merge_world_at_pose(
            bowl_world,
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=2.4,
                y=2.45,
                z=bowl_height,
                reference_frame=world.root,
            ),
        )

    pitcher = world.get_body_by_name("Static_MilkPitcher.stl")
    bowl = world.get_body_by_name("bowl.stl")
    pouring_source = PouringCup(root=pitcher)
    with world.modify_world():
        world.add_semantic_annotations([pouring_source, Bowl(root=bowl)])

    context = Context(
        world=world,
        robot=robot,
        evaluate_conditions=False,
    )
    grasp_description = GraspDescription(
        approach_direction=ApproachDirection.FRONT,
        vertical_alignment=VerticalAlignment.NoAlignment,
        end_effector=ViewManager.get_end_effector_view(Arms.RIGHT, robot),
    )
    pouring_task = sequential(
        [
            ParkArmsAction(Arms.BOTH),
            MoveTorsoAction(TorsoState.HIGH),
            PickUpAction(pitcher, Arms.RIGHT, grasp_description),
            a(PouringAction)(
                target_container=bowl,
                source_container=pouring_source,
                arm=Arms.RIGHT,
            ),
        ],
        context=context,
    )

    with simulated_robot:
        pouring_task.perform()

    tool_frame = ViewManager.get_end_effector_view(Arms.RIGHT, robot).tool_frame
    assert pitcher.parent_connection.parent == tool_frame
    return PouringAlignment.measure(pitcher, bowl)


def main() -> None:
    """
    Execute the compact pouring example and report its final alignment.
    """
    alignment = execute_simple_pouring()
    print(
        "Simple pouring task completed with "
        f"{alignment.horizontal_error:.3f} m horizontal mouth error and "
        f"{alignment.vertical_clearance:.3f} m rim clearance."
    )


if __name__ == "__main__":
    main()
