#!/usr/bin/env python
# coding: utf-8

# # GCS full-kitchen pick, navigate, and pour
# 
# This notebook extends the full-kitchen transport example with a pouring task. The PR2 plans through the apartment with a Graph of Convex Sets (GCS), picks up a pitcher from the kitchen island, navigates to a bowl on the dining table, and executes `PouringAction`.
# 
# The world models the source and target vessels plus the robot motion. Pouring is represented by positioning the pitcher's mouth over the bowl and tilting the pitcher; liquid particles are not simulated. Run the notebook from top to bottom with the `new-cram-env` kernel.

# In[1]:


import math
import threading
import time
from functools import partial
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import rclpy
from scipy.spatial.transform import Rotation

import coraplex
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot
from coraplex.locations.factories import reachability_location
from coraplex.plans.factories import code, sequential
from coraplex.robot_plans.actions.composite.tool_based import PouringAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from coraplex.testing import setup_world
from coraplex.view_manager import ViewManager
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl, PouringCup
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3, Pose
from semantic_digital_twin.world_description.geometry import BoundingBox
from semantic_digital_twin.world_description.graph_of_convex_sets import GraphOfConvexSets
from semantic_digital_twin.world_description.shape_collection import BoundingBoxCollection
from semantic_digital_twin.world_description.world_entity import Body


# ## Spawn the apartment, PR2, pitcher, and bowl
# 
# The apartment and initial PR2 state come from the same `setup_world()` used by the transport demo. The pitcher is placed beside the existing groceries on the island and the bowl is placed on the main table. Their heights are derived from the supporting geometry and mesh bounds.

# In[2]:


world = setup_world()
robot = PR2.from_world(world)
initial_base_pose = robot.root.global_pose

coraplex_root = Path(coraplex.__file__).resolve().parents[2]
object_directory = coraplex_root / "resources" / "objects"
pitcher_world = STLParser(str(object_directory / "Static_MilkPitcher.stl")).parse()
bowl_world = STLParser(str(object_directory / "bowl.stl")).parse()

milk = world.get_body_by_name("milk.stl")
island_height = float(milk.global_pose.z + milk.combined_mesh.bounds[0][2])
pitcher_height = island_height - float(pitcher_world.root.combined_mesh.bounds[0][2])

main_table = world.get_body_by_name("table_area_main")
table_top_height = float(main_table.global_pose.z + main_table.combined_mesh.bounds[1][2])
bowl_height = table_top_height - float(bowl_world.root.combined_mesh.bounds[0][2])

with world.modify_world():
    world.merge_world_at_pose(
        pitcher_world,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            x=2.37, y=2.2, z=pitcher_height, reference_frame=world.root
        ),
    )
    world.merge_world_at_pose(
        bowl_world,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            x=4.8, y=3.35, z=bowl_height, reference_frame=world.root
        ),
    )

pitcher = world.get_body_by_name("Static_MilkPitcher.stl")
bowl = world.get_body_by_name("bowl.stl")
pouring_source = PouringCup(root=pitcher)
target_bowl = Bowl(root=bowl)
with world.modify_world():
    world.add_semantic_annotations([pouring_source, target_bowl])

{
    "pitcher_position": [round(float(value), 3) for value in pitcher.global_pose.to_position().to_list()[:3]],
    "bowl_position": [round(float(value), 3) for value in bowl.global_pose.to_position().to_list()[:3]],
}


# ## Publish the world
# 
# The publishers expose the same TF and marker streams as the transport notebook. In RViz, use the printed fixed frame and marker topic.

# In[3]:


if not rclpy.ok():
    rclpy.init()

ros_node = rclpy.create_node("coraplex_gcs_full_kitchen_pouring")
ros_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
ros_thread.start()

tf_publisher = TFPublisher(_world=world, node=ros_node)
visualization_publisher = VizMarkerPublisher(_world=world, node=ros_node)

print(f"RViz fixed frame: {world.root.name}")
print(f"MarkerArray topic: {visualization_publisher.topic_name}")


# In[4]:


context = Context(world=world, robot=robot, ros_node=ros_node)
assert context.evaluate_conditions

grasp_description = GraspDescription(
    approach_direction=ApproachDirection.FRONT,
    vertical_alignment=VerticalAlignment.NoAlignment,
    end_effector=ViewManager.get_end_effector_view(Arms.RIGHT, robot),
)
preparation = sequential(
    [ParkArmsAction(Arms.BOTH), MoveTorsoAction(TorsoState.HIGH)],
    context=context,
)
with simulated_robot:
    preparation.perform()


# ## Plan both navigation legs with GCS
# 
# The first path ends at a reachable pitcher grasp. The second starts there and ends at a reachable pose beside the bowl. Short decomposition segments are filtered before the path is interpolated for visible base playback.

# In[5]:


def navigation_point(base_pose: Pose, height: float, reference_frame: Body) -> Point3:
    """Project a robot base pose onto the GCS navigation layer."""
    return Point3(
        x=base_pose.x,
        y=base_pose.y,
        z=height,
        reference_frame=reference_frame,
    )


def filter_path(
    raw_path: list[Point3],
    minimum_waypoint_distance: float,
) -> list[Point3]:
    """Remove consecutive decomposition waypoints below a distance threshold."""
    filtered_path = [raw_path[0]]
    for candidate in raw_path[1:-1]:
        previous = filtered_path[-1]
        candidate_distance = math.hypot(
            float(candidate.x - previous.x),
            float(candidate.y - previous.y),
        )
        if candidate_distance >= minimum_waypoint_distance:
            filtered_path.append(candidate)
    filtered_path.append(raw_path[-1])
    return filtered_path


def interpolate_path(
    path: list[Point3],
    step_distance: float,
    final_yaw: float,
    reference_frame: Body,
) -> list[Pose]:
    """Convert a GCS path into closely spaced base poses for RViz playback."""
    poses = []
    for segment_start, segment_end in zip(path, path[1:]):
        start_x = float(segment_start.x)
        start_y = float(segment_start.y)
        distance_x = float(segment_end.x) - start_x
        distance_y = float(segment_end.y) - start_y
        segment_distance = math.hypot(distance_x, distance_y)
        segment_step_count = max(1, math.ceil(segment_distance / step_distance))
        segment_yaw = math.atan2(distance_y, distance_x)

        for segment_step_index in range(1, segment_step_count + 1):
            progress = segment_step_index / segment_step_count
            poses.append(
                Pose.from_xyz_rpy(
                    x=start_x + distance_x * progress,
                    y=start_y + distance_y * progress,
                    z=0.0,
                    yaw=segment_yaw,
                    reference_frame=reference_frame,
                )
            )

    final_point = path[-1]
    poses[-1] = Pose.from_xyz_rpy(
        x=final_point.x,
        y=final_point.y,
        z=0.0,
        yaw=final_yaw,
        reference_frame=reference_frame,
    )
    return poses


# In[6]:


navigation_height = 0.2
search_space = BoundingBoxCollection(
    [
        BoundingBox(
            min_x=0.2,
            max_x=6.5,
            min_y=0.2,
            max_y=5.8,
            min_z=0.15,
            max_z=0.25,
            origin=HomogeneousTransformationMatrix(reference_frame=world.root),
        )
    ],
    reference_frame=world.root,
)
graph_of_convex_sets = GraphOfConvexSets.free_space_from_world(
    world=world,
    search_space=search_space,
    bloat_obstacles=0.22,
)
graph_of_convex_sets.create_as_region(
    name=PrefixedName("full_kitchen_pouring_navigation_free_space")
)

start_point = navigation_point(initial_base_pose, navigation_height, world.root)
pickup_base_pose = None
pickup_point = None
raw_pickup_path = None
for candidate_base_pose in reachability_location(
    pitcher, context, Arms.RIGHT, grasp_description
):
    candidate_point = navigation_point(candidate_base_pose, navigation_height, world.root)
    candidate_path = graph_of_convex_sets.path_from_to(start_point, candidate_point)
    if candidate_path is None:
        continue
    pickup_base_pose = candidate_base_pose
    pickup_point = candidate_point
    raw_pickup_path = candidate_path
    break

assert pickup_base_pose is not None
assert pickup_point is not None
assert raw_pickup_path is not None

pouring_base_pose = None
pouring_point = None
raw_pouring_path = None
for candidate_base_pose in reachability_location(
    bowl, context, Arms.RIGHT, grasp_description
):
    candidate_point = navigation_point(candidate_base_pose, navigation_height, world.root)
    candidate_path = graph_of_convex_sets.path_from_to(pickup_point, candidate_point)
    if candidate_path is None:
        continue
    pouring_base_pose = candidate_base_pose
    pouring_point = candidate_point
    raw_pouring_path = candidate_path
    break

assert pouring_base_pose is not None
assert pouring_point is not None
assert raw_pouring_path is not None

pickup_path = filter_path(raw_pickup_path, minimum_waypoint_distance=0.08)
pouring_path = filter_path(raw_pouring_path, minimum_waypoint_distance=0.08)

{
    "graph_nodes": len(graph_of_convex_sets.graph.nodes()),
    "graph_edges": len(graph_of_convex_sets.graph.edges()),
    "initial_base": [round(float(value), 3) for value in initial_base_pose.to_position().to_list()[:3]],
    "pickup_base": [round(float(value), 3) for value in pickup_base_pose.to_position().to_list()[:3]],
    "pouring_base": [round(float(value), 3) for value in pouring_base_pose.to_position().to_list()[:3]],
}


# In[7]:


figure = go.Figure(graph_of_convex_sets.plot_free_space())
figure.add_trace(
    go.Scatter3d(
        x=[point.x for point in pickup_path],
        y=[point.y for point in pickup_path],
        z=[navigation_height for point in pickup_path],
        mode="lines+markers",
        name="Pitcher pickup route",
        line={"color": "black", "width": 8},
    )
)
figure.add_trace(
    go.Scatter3d(
        x=[point.x for point in pouring_path],
        y=[point.y for point in pouring_path],
        z=[navigation_height + 0.01 for point in pouring_path],
        mode="lines+markers",
        name="Bowl route",
        line={"color": "orange", "width": 8},
    )
)
figure.add_trace(
    go.Scatter3d(
        x=[bowl.global_pose.x],
        y=[bowl.global_pose.y],
        z=[bowl.global_pose.z],
        mode="markers",
        name="Pour target",
        marker={"color": "green", "size": 8},
    )
)
figure.update_layout(height=600, scene_aspectmode="data")
figure


# In[8]:


navigation_step_distance = 0.08
seconds_per_navigation_step = 0.08
pickup_yaw = float(pickup_base_pose.to_quaternion().to_rpy()[2])
pouring_yaw = float(pouring_base_pose.to_quaternion().to_rpy()[2])

pickup_navigation_poses = interpolate_path(
    pickup_path, navigation_step_distance, pickup_yaw, world.root
)
pouring_navigation_poses = interpolate_path(
    pouring_path, navigation_step_distance, pouring_yaw, world.root
)

pickup_navigation_steps = []
for navigation_pose in pickup_navigation_poses:
    pickup_navigation_steps.extend(
        [
            NavigateAction(navigation_pose),
            code(partial(time.sleep, seconds_per_navigation_step)),
        ]
    )

pouring_navigation_steps = []
for navigation_pose in pouring_navigation_poses:
    pouring_navigation_steps.extend(
        [
            NavigateAction(navigation_pose),
            code(partial(time.sleep, seconds_per_navigation_step)),
        ]
    )

len(pickup_navigation_poses), len(pouring_navigation_poses)


# ## Pick up the pitcher and pour
# 
# `PouringAction` computes an upright pre-pour pose and a tilted pose whose source opening is aligned over the bowl. The action finishes at the tilted pose so the result remains visible in RViz.

# In[9]:


pouring_action = PouringAction(
    target_container=bowl,
    source_container=pouring_source,
    arm=Arms.RIGHT,
)
task = sequential(
    [
        *pickup_navigation_steps,
        PickUpAction(pitcher, Arms.RIGHT, grasp_description),
        ParkArmsAction(Arms.BOTH),
        *pouring_navigation_steps,
        pouring_action,
    ],
    context=context,
)

with simulated_robot:
    task.perform()


# In[10]:


pre_pour_pose, planned_pour_pose = pouring_action._pour_poses()
pre_pour_rotation = Rotation.from_quat(
    [float(value) for value in pre_pour_pose.to_quaternion().to_np()]
)
planned_pour_rotation = Rotation.from_quat(
    [float(value) for value in planned_pour_pose.to_quaternion().to_np()]
)
pouring_tilt_radians = (pre_pour_rotation.inv() * planned_pour_rotation).magnitude()

tool_frame = ViewManager.get_end_effector_view(Arms.RIGHT, robot).tool_frame
pitcher_rotation = pitcher.global_pose.to_rotation_matrix().to_np()[:3, :3]
pitcher_up = pitcher_rotation @ np.array([0.0, 0.0, 1.0])
achieved_pitcher_tilt = math.acos(float(np.clip(pitcher_up[2], -1.0, 1.0)))

assert pitcher.parent_connection.parent == tool_frame
assert math.isclose(pouring_tilt_radians, pouring_action.tilt_angle, abs_tol=1e-6)
assert math.isclose(achieved_pitcher_tilt, pouring_tilt_radians, abs_tol=0.02)

{
    "graph_nodes": len(graph_of_convex_sets.graph.nodes()),
    "graph_edges": len(graph_of_convex_sets.graph.edges()),
    "pickup_navigation_updates": len(pickup_navigation_poses),
    "pouring_navigation_updates": len(pouring_navigation_poses),
    "pitcher_attached_to_right_gripper": pitcher.parent_connection.parent == tool_frame,
    "pouring_tilt_radians": round(pouring_tilt_radians, 3),
    "achieved_pitcher_tilt_radians": round(achieved_pitcher_tilt, 3),
    "target_container": str(bowl.name),
}

