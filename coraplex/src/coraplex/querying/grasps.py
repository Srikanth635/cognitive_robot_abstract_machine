from __future__ import annotations

from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import List, Set, Tuple

from coraplex.datastructures.enums import (
    ApproachDirection,
    AxisIdentifier,
    VerticalAlignment,
)
from coraplex.datastructures.grasp import GraspDescription, PreferredGraspAlignment

# %% enumerating what the geometry allows


GraspOption = Tuple[ApproachDirection, VerticalAlignment]
"""One approach direction paired with one vertical alignment."""


def grasp_options_for(
    end_effector: EndEffector, target: Body, with_vertical_alignment: bool
) -> List[GraspOption]:
    """
    Ask coraplex which grasps the object's orientation actually permits.

    Approach directions are named in the *object's* frame, so which of them faces the
    robot depends on how the object is turned. That relation is geometry the planner
    already computes, and it cannot be inferred from a schema or from coordinates alone.

    :param end_effector: The end effector the grasp would be performed with.
    :param target: The body to be grasped.
    :param with_vertical_alignment: Whether to consider top and bottom alignments as
        well as the horizontal faces.
    :return: The permitted approach and alignment pairs.
    """
    alignment = PreferredGraspAlignment(
        preferred_axis=AxisIdentifier.Undefined,
        with_vertical_alignment=with_vertical_alignment,
        with_rotated_gripper=False,
    )
    descriptions = GraspDescription.calculate_grasp_descriptions(
        end_effector=end_effector,
        pose=Pose(reference_frame=target),
        grasp_alignment=alignment,
    )
    return [
        (description.approach_direction, description.vertical_alignment)
        for description in descriptions
    ]


def permitted_grasp_options(
    end_effectors: List[EndEffector], target: Body
) -> List[GraspOption]:
    """
    Collect every grasp the geometry permits, across alignments and end effectors.

    The computation depends on where the object sits relative to the robot's base
    rather than on which end effector is used, so results are gathered without
    repetition.

    :param end_effectors: The end effectors available.
    :param target: The body to be grasped.
    :return: The permitted approach and alignment pairs, first-computed order kept.
    """
    collected: List[GraspOption] = []
    seen: Set[GraspOption] = set()
    for end_effector in end_effectors:
        for with_vertical_alignment in (False, True):
            for option in grasp_options_for(
                end_effector, target, with_vertical_alignment
            ):
                if option in seen:
                    continue
                seen.add(option)
                collected.append(option)
    return collected


# %% describing them


def describe_grasp_options(
    end_effectors: List[EndEffector], target: Body
) -> str:
    """
    State the permitted grasps in the form the slots are answered in.

    The first option listed is the face most turned toward the robot, which is the one
    the planner would pick if left to itself.

    :param end_effectors: The end effectors available.
    :param target: The body to be grasped.
    :return: The description, empty when nothing could be computed.
    """
    options = permitted_grasp_options(end_effectors, target)
    if not options:
        return ""
    lines = [
        f"- approach_direction {approach.name}, "
        f"vertical_alignment {alignment.name}"
        for approach, alignment in options
    ]
    return (
        f"Grasps permitted by how {target.name.name} is turned relative to the robot.\n"
        "Approach directions name faces of the object, not of the robot, and the first "
        "listed is the face most turned toward the robot:\n" + "\n".join(lines)
    )