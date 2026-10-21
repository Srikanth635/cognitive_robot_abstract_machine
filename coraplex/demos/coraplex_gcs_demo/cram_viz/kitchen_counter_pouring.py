"""Run a short GCS kitchen-counter pouring task in CRAM-Viz Live."""

import math
import time
from dataclasses import dataclass
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import Any, Self

import numpy as np
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import code, sequential
from coraplex.robot_plans.actions.composite.tool_based import PouringAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from coraplex.testing import setup_world
from coraplex.view_manager import ViewManager
from krrood.entity_query_language.factories import a
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    PouringCup,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Pose,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import BoundingBox
from semantic_digital_twin.world_description.graph_of_convex_sets import (
    GraphOfConvexSets,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import Body

import coraplex


# %% Failures
class CounterPouringRouteError(RuntimeError):
    """Report that the compact kitchen route cannot be constructed."""


class CounterPouringExecutionError(RuntimeError):
    """Report that the compact pouring task ended in an invalid state."""


# %% Demo data
@dataclass(frozen=True)
class CounterPouringConfiguration:
    """Define the fixed scene and motion values for the compact demonstration."""

    initial_base_x: float = 1.45
    """Initial robot-base position on the world x-axis."""

    initial_base_y: float = 2.32
    """Initial robot-base position on the world y-axis."""

    pickup_base_x: float = 1.75
    """Robot-base x-coordinate used to reach the cereal box."""

    pickup_base_y: float = 2.32
    """Robot-base y-coordinate used to reach the cereal box."""

    pickup_base_yaw: float = -0.19
    """Robot-base heading used for pickup and pouring."""

    cereal_box_x: float = 2.37
    """Cereal-box position on the counter's x-axis."""

    cereal_box_y: float = 2.2
    """Cereal-box position on the counter's y-axis."""

    bowl_x: float = 2.4
    """Bowl position on the counter's x-axis."""

    bowl_y: float = 2.45
    """Bowl position on the counter's y-axis."""

    navigation_height: float = 0.2
    """Height of the planar GCS navigation layer."""

    navigation_step_distance: float = 0.08
    """Maximum distance represented by one visible navigation update."""

    seconds_per_navigation_step: float = 0.08
    """Display delay following each navigation update."""

    maximum_route_length: float = 0.75
    """Largest route accepted by this compact demonstration."""

    pour_hold_seconds: float = 1.75
    """How long the source container stays tilted before it is righted and placed
    back at its original position."""


@dataclass(frozen=True)
class CounterPouringScene:
    """Hold the world entities required by the pouring task."""

    world: World
    """Apartment world containing the fixed kitchen."""

    robot: PR2
    """Robot annotation bound to the apartment world."""

    cereal_box: Body
    """Cereal-box body placed on the kitchen counter."""

    cereal_box_original_pose: Pose
    """Cereal-box pose before pickup, where it is placed back after pouring."""

    bowl: Body
    """Bowl body placed beside the cereal box."""

    pouring_source: PouringCup
    """Pouring annotation attached to the cereal box."""

    grasp_description: GraspDescription
    """Right-arm front grasp used to pick up the cereal box."""

    context: Context
    """Execution context shared by every plan node."""


@dataclass(frozen=True)
class NavigationRoute:
    """Represent the short GCS route to the cereal box."""

    destination_pose: Pose
    """Final robot-base pose for pickup and pouring."""

    path: list[Point3]
    """Filtered points returned by the GCS planner."""

    length: float
    """Total planar length of the filtered route."""


@dataclass(frozen=True)
class NavigationPlayback:
    """Hold visible base poses and their executable plan nodes."""

    poses: list[Pose]
    """Interpolated robot-base poses shown during navigation."""

    steps: list[Any]
    """Navigation and timing nodes inserted into the top-level plan."""


@dataclass(frozen=True)
class PouringAlignment:
    """Describe the final relation between the cereal opening and bowl rim."""

    source_opening_position: tuple[float, float, float]
    """World position of the center of the cereal-box opening."""

    target_rim_center: tuple[float, float, float]
    """World position of the center of the bowl rim."""

    @classmethod
    def measure(cls, cereal_box: Body, bowl: Body) -> Self:
        """Measure the cereal-box opening relative to the bowl rim."""
        cereal_box_bounding_box = cereal_box.visual.as_bounding_box_collection_in_frame(
            cereal_box
        ).bounding_box()
        bowl_bounding_box = bowl.visual.as_bounding_box_collection_in_frame(
            bowl
        ).bounding_box()
        cereal_box_opening = np.array(
            [
                0.5 * (cereal_box_bounding_box.min_x + cereal_box_bounding_box.max_x),
                0.5 * (cereal_box_bounding_box.min_y + cereal_box_bounding_box.max_y),
                cereal_box_bounding_box.max_z,
                1.0,
            ]
        )
        bowl_rim_center = np.array(
            [
                0.5 * (bowl_bounding_box.min_x + bowl_bounding_box.max_x),
                0.5 * (bowl_bounding_box.min_y + bowl_bounding_box.max_y),
                bowl_bounding_box.max_z,
                1.0,
            ]
        )
        world_opening = cereal_box.global_transform.to_np() @ cereal_box_opening
        world_rim_center = bowl.global_transform.to_np() @ bowl_rim_center
        return cls(
            source_opening_position=tuple(float(value) for value in world_opening[:3]),
            target_rim_center=tuple(float(value) for value in world_rim_center[:3]),
        )

    @property
    def horizontal_error(self) -> float:
        """Return the horizontal opening distance from the bowl center."""
        return float(
            np.linalg.norm(
                np.asarray(self.source_opening_position[:2])
                - np.asarray(self.target_rim_center[:2])
            )
        )

    @property
    def vertical_clearance(self) -> float:
        """Return the cereal-box opening height above the bowl rim."""
        return self.source_opening_position[2] - self.target_rim_center[2]


@dataclass(frozen=True)
class CounterPouringResult:
    """Summarize the observable outcome of the live demonstration."""

    route_length: float
    """Length of the GCS approach route in metres."""

    navigation_updates: int
    """Number of visible robot-base navigation updates."""

    final_base_error: float
    """Planar distance from the intended final base position."""

    source_name: str
    """Mesh name of the object poured from."""

    target_name: str
    """Mesh name of the container poured into."""

    alignment: PouringAlignment
    """Final spatial relation between the cereal-box opening and bowl rim."""


# %% Scene construction
def create_counter_pouring_scene(
    configuration: CounterPouringConfiguration,
) -> CounterPouringScene:
    """Create the fixed kitchen with a nearby cereal box and bowl."""
    world = setup_world()
    robot = PR2.from_world(world)
    object_directory = (
        Path(coraplex.__file__).resolve().parents[2] / "resources" / "objects"
    )
    bowl_world = STLParser(str(object_directory / "bowl.stl")).parse()
    milk = world.get_body_by_name("milk.stl")
    cereal_box = world.get_body_by_name("breakfast_cereal.stl")
    counter_height = float(milk.global_pose.z + milk.combined_mesh.bounds[0][2])
    cereal_box_height = counter_height - float(cereal_box.combined_mesh.bounds[0][2])
    bowl_height = counter_height - float(bowl_world.root.combined_mesh.bounds[0][2])

    with world.modify_world():
        robot.root.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=configuration.initial_base_x,
                y=configuration.initial_base_y,
                reference_frame=world.root,
            )
        )
        cereal_box.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=configuration.cereal_box_x,
                y=configuration.cereal_box_y,
                z=cereal_box_height,
                reference_frame=world.root,
            )
        )
        world.merge_world_at_pose(
            bowl_world,
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=configuration.bowl_x,
                y=configuration.bowl_y,
                z=bowl_height,
                reference_frame=world.root,
            ),
        )

    cereal_box_original_pose = Pose.from_xyz_rpy(
        x=configuration.cereal_box_x,
        y=configuration.cereal_box_y,
        z=cereal_box_height,
        reference_frame=world.root,
    )

    bowl = world.get_body_by_name("bowl.stl")
    pouring_source = PouringCup(root=cereal_box)
    with world.modify_world():
        world.add_semantic_annotations([pouring_source, Bowl(root=bowl)])

    grasp_description = GraspDescription(
        approach_direction=ApproachDirection.FRONT,
        vertical_alignment=VerticalAlignment.NoAlignment,
        end_effector=ViewManager.get_end_effector_view(Arms.RIGHT, robot),
    )
    return CounterPouringScene(
        world=world,
        robot=robot,
        cereal_box=cereal_box,
        cereal_box_original_pose=cereal_box_original_pose,
        bowl=bowl,
        pouring_source=pouring_source,
        grasp_description=grasp_description,
        context=Context(world=world, robot=robot, evaluate_conditions=False),
    )


# %% GCS navigation
def filter_navigation_path(
    raw_path: list[Point3],
    minimum_waypoint_distance: float,
) -> list[Point3]:
    """Remove consecutive decomposition points below the requested distance."""
    filtered_path = [raw_path[0]]
    for candidate in raw_path[1:-1]:
        previous = filtered_path[-1]
        distance = math.hypot(
            float(candidate.x - previous.x),
            float(candidate.y - previous.y),
        )
        if distance >= minimum_waypoint_distance:
            filtered_path.append(candidate)
    filtered_path.append(raw_path[-1])
    return filtered_path


def create_short_route(
    scene: CounterPouringScene,
    configuration: CounterPouringConfiguration,
) -> NavigationRoute:
    """Plan a compact GCS route from the initial base pose to the cereal box."""
    search_space = BoundingBoxCollection(
        [
            BoundingBox(
                min_x=1.2,
                max_x=2.0,
                min_y=1.6,
                max_y=3.0,
                min_z=configuration.navigation_height - 0.05,
                max_z=configuration.navigation_height + 0.05,
                origin=HomogeneousTransformationMatrix(
                    reference_frame=scene.world.root
                ),
            )
        ],
        reference_frame=scene.world.root,
    )
    navigation_graph = GraphOfConvexSets.free_space_from_world(
        world=scene.world,
        search_space=search_space,
        bloat_obstacles=0.22,
    )
    navigation_graph.create_as_region(
        name=PrefixedName("kitchen_counter_pouring_navigation_free_space")
    )
    start_pose = scene.robot.root.global_pose
    start_point = Point3(
        x=start_pose.x,
        y=start_pose.y,
        z=configuration.navigation_height,
        reference_frame=scene.world.root,
    )
    destination_point = Point3(
        x=configuration.pickup_base_x,
        y=configuration.pickup_base_y,
        z=configuration.navigation_height,
        reference_frame=scene.world.root,
    )
    raw_path = navigation_graph.path_from_to(start_point, destination_point)
    if raw_path is None or len(raw_path) < 2:
        raise CounterPouringRouteError(
            "The cereal-box approach is not connected in the kitchen navigation graph"
        )
    path = filter_navigation_path(raw_path, configuration.navigation_step_distance)
    route_length = sum(
        math.hypot(
            float(route_end.x - route_start.x),
            float(route_end.y - route_start.y),
        )
        for route_start, route_end in pairwise(path)
    )
    if route_length > configuration.maximum_route_length:
        raise CounterPouringRouteError(
            f"The cereal-box approach is {route_length:.3f} m, which is not compact"
        )
    return NavigationRoute(
        destination_pose=Pose.from_xyz_rpy(
            x=configuration.pickup_base_x,
            y=configuration.pickup_base_y,
            yaw=configuration.pickup_base_yaw,
            reference_frame=scene.world.root,
        ),
        path=path,
        length=route_length,
    )


def interpolate_navigation_path(
    route: NavigationRoute,
    reference_frame: Body,
    step_distance: float,
) -> list[Pose]:
    """Convert a GCS path into closely spaced robot-base poses."""
    poses = []
    for segment_start, segment_end in pairwise(route.path):
        start_x = float(segment_start.x)
        start_y = float(segment_start.y)
        distance_x = float(segment_end.x) - start_x
        distance_y = float(segment_end.y) - start_y
        segment_distance = math.hypot(distance_x, distance_y)
        step_count = max(1, math.ceil(segment_distance / step_distance))
        segment_yaw = math.atan2(distance_y, distance_x)
        for step_index in range(1, step_count + 1):
            progress = step_index / step_count
            poses.append(
                Pose.from_xyz_rpy(
                    x=start_x + distance_x * progress,
                    y=start_y + distance_y * progress,
                    yaw=segment_yaw,
                    reference_frame=reference_frame,
                )
            )
    final_point = route.path[-1]
    final_yaw = float(route.destination_pose.to_quaternion().to_rpy()[2])
    poses[-1] = Pose.from_xyz_rpy(
        x=final_point.x,
        y=final_point.y,
        yaw=final_yaw,
        reference_frame=reference_frame,
    )
    return poses


def create_navigation_playback(
    route: NavigationRoute,
    scene: CounterPouringScene,
    configuration: CounterPouringConfiguration,
) -> NavigationPlayback:
    """Create visible navigation poses and their sequential plan nodes."""
    poses = interpolate_navigation_path(
        route,
        scene.world.root,
        configuration.navigation_step_distance,
    )
    steps = []
    for pose in poses:
        steps.extend(
            [
                NavigateAction(pose),
                code(partial(time.sleep, configuration.seconds_per_navigation_step)),
            ]
        )
    return NavigationPlayback(poses=poses, steps=steps)


# %% Task execution
def execute_counter_pouring(
    configuration: CounterPouringConfiguration | None = None,
) -> CounterPouringResult:
    """Approach the counter, pick up the cereal box, and pour into the bowl."""
    selected_configuration = configuration or CounterPouringConfiguration()
    scene = create_counter_pouring_scene(selected_configuration)
    route = create_short_route(scene, selected_configuration)
    navigation = create_navigation_playback(route, scene, selected_configuration)
    pour_task = sequential(
        [
            ParkArmsAction(Arms.BOTH),
            MoveTorsoAction(TorsoState.HIGH),
            *navigation.steps,
            PickUpAction(scene.cereal_box, Arms.RIGHT, scene.grasp_description),
            a(PouringAction)(
                target_container=scene.bowl,
                source_container=scene.pouring_source,
                arm=Arms.RIGHT,
            ),
        ],
        context=scene.context,
    )

    with simulated_robot:
        pour_task.plan.perform()

    # Read only after pour_task's own plan has fully returned: a check made from
    # *inside* a still-running plan (e.g. a code() node appended to the same
    # sequence) sees the world mid-update, before attachment/detachment settles,
    # and reports the cereal box as still on the counter even though it was just
    # picked up and poured from.
    tool_frame = ViewManager.get_end_effector_view(Arms.RIGHT, scene.robot).tool_frame
    if scene.cereal_box.parent_connection.parent != tool_frame:
        raise CounterPouringExecutionError(
            "The cereal box is not attached to the right gripper after pouring"
        )
    alignment = PouringAlignment.measure(scene.cereal_box, scene.bowl)

    revert_task = sequential(
        [
            code(partial(time.sleep, selected_configuration.pour_hold_seconds)),
            PouringAction(
                target_container=scene.bowl,
                source_container=scene.pouring_source,
                arm=Arms.RIGHT,
                tilt_angle=0.0,
            ),
            PlaceAction(scene.cereal_box, scene.cereal_box_original_pose, Arms.RIGHT),
        ],
        context=scene.context,
    )

    with simulated_robot:
        revert_task.plan.perform()

    final_position = scene.robot.root.global_pose.to_position()
    destination_position = route.destination_pose.to_position()
    final_base_error = math.dist(
        [float(final_position.x), float(final_position.y)],
        [float(destination_position.x), float(destination_position.y)],
    )
    return CounterPouringResult(
        route_length=route.length,
        navigation_updates=len(navigation.poses),
        final_base_error=final_base_error,
        source_name=scene.cereal_box.name.name,
        target_name=scene.bowl.name.name,
        alignment=alignment,
    )


def main() -> None:
    """Execute the live demonstration and print its verified outcome."""
    result = execute_counter_pouring()
    print(
        "Kitchen-counter cereal pouring completed with "
        f"{result.route_length:.3f} m of GCS navigation, "
        f"{result.navigation_updates} base updates, and "
        f"{result.alignment.horizontal_error:.3f} m horizontal mouth error."
    )


if __name__ == "__main__":
    main()
