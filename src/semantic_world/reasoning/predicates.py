import itertools

import numpy as np
import trimesh.boolean
from entity_query_language import let, an, entity, contains, and_, not_
from trimesh.ray.ray_triangle import RayMeshIntersector
from typing_extensions import List, Optional, Tuple

from ..spatial_computations.raytracer import RayTracer
from ..collision_checking.collision_detector import CollisionCheck
from ..collision_checking.trimesh_collision_detector import TrimeshCollisionDetector
from ..robots import (
    RobotView,
    Camera,
    Manipulator,
    Finger,
    AbstractRobot,
    ParallelGripper,
)
from ..spatial_types.spatial_types import Point3, TransformationMatrix
from ..world_description.geometry import BoundingBoxCollection
from ..world_description.world_entity import Body, Region, KinematicStructureEntity


def stable(obj: Body) -> bool:
    """
    Checks if an object is stable in the world. Stable meaning that its position will not change after simulating
    physics in the World. This will be done by simulating the world for 10 seconds and comparing
    the previous coordinates with the coordinates after the simulation.

    :param obj: The object which should be checked
    :return: True if the given object is stable in the world False else
    """
    raise NotImplementedError("Needs multiverse")


def contact(
    body1: Body,
    body2: Body,
    threshold: float = 0.001,
) -> bool:
    """
    Checks if two objects are in contact or not.

    :param body1: The first object
    :param body2: The second object
    :param threshold: The threshold for contact detection
    :return: True if the two objects are in contact False else
    """
    assert body1._world == body2._world, "Both bodies must be in the same world"
    tcd = TrimeshCollisionDetector(body1._world)
    result = tcd.check_collision_between_bodies(body1, body2)

    if result is None:
        return False
    return result.contact_distance < threshold


def robot_in_collision(
    robot: AbstractRobot,
    ignore_collision_with: Optional[List[Body]] = None,
    threshold: float = 0.001,
) -> bool:
    """
    Check if the robot collides with any object in the world at the given pose.

    :param robot: The robot object
    :param ignore_collision_with: A list of objects to ignore collision with
    :param threshold: The threshold for contact detection
    :return: True if the robot collides with any object, False otherwise
    """

    if ignore_collision_with is None:
        ignore_collision_with = []

    body = let("body", type_=Body, domain=robot._world.bodies_with_enabled_collision)
    possible_collisions_bodies = an(
        entity(
            body,
            and_(
                not_(contains(robot.bodies, body)),
                not_(contains(ignore_collision_with, body)),
            ),
        ),
    ).evaluate()

    tcd = TrimeshCollisionDetector(robot._world)

    collisions = tcd.check_collisions(
        {
            CollisionCheck(robot_body, collision_body, threshold, robot._world)
            for robot_body, collision_body in itertools.product(
                robot.bodies_with_collisions, possible_collisions_bodies
            )
        }
    )
    return len(collisions) > 0


def robot_holds_body(robot: AbstractRobot, body: Body) -> bool:
    """
    Check if a robot is holding an object.

    :param robot: The robot object
    :param body: The body to check if it is picked
    :return: True if the robot is holding the object, False otherwise
    """
    grippers = an(
        entity(
            g := let("gripper", ParallelGripper, robot._world.views), g._robot == robot
        )
    ).evaluate()

    return any([is_body_in_gripper(body, gripper) > 0.0 for gripper in grippers])


def get_visible_bodies(camera: Camera) -> List[KinematicStructureEntity]:
    """
    Get all bodies and regions that are visible from the given camera using a segmentation mask.

    :param camera: The camera for which the visible objects should be returned
    :return: A list of bodies/regions that are visible from the camera
    """
    rt = RayTracer(camera._world)
    rt.update_scene()

    # Build a camera pose at the camera's position, looking along +X in world coordinates
    # (RayTracer internally orients the camera to +X for an identity rotation).
    cam_pose = np.eye(4, dtype=float)
    cam_pose[:3, 3] = camera.root.global_pose.to_np()[:3, 3]

    seg = rt.create_segmentation_mask(cam_pose, resolution=256)
    indices = np.unique(seg)
    indices = indices[indices > -1]
    bodies = [camera._world.kinematic_structure[i] for i in indices]

    return bodies


def visible(camera: Camera, obj: KinematicStructureEntity) -> bool:
    """
    Checks if a body/region is visible by the given camera.
    """
    return obj in get_visible_bodies(camera)


def occluding_bodies(camera: Camera, body: Body) -> List[Body]:
    """
    Determines the bodies that occlude a given body in the scene as seen from a specified camera.

    This function uses a ray-tracing approach to check occlusion. Rays are cast from the camera's
    origin towards points sampled on the given body's bounding boxes. If the rays are obstructed by
    another body before reaching the target body, these obstructing bodies are identified as occluders.

    :param camera: The camera for which the occluding bodies should be returned
    :param body: The body for which the occluding bodies should be returned
    :return: A list of bodies that are occluding the given body.
    """
    # Initialize ray tracer and ensure scene is up-to-date
    rt = RayTracer(camera._world)
    rt.update_scene()

    # Camera origin (use same convention as get_visible_bodies: camera root position)
    cam_T_world = camera.root.global_pose.to_np()
    cam_origin = cam_T_world[:3, 3]

    # Sample points on the body's world-aligned bounding boxes
    # Use all collision shapes' bounding boxes transformed to world frame
    bb_collection = body.as_bounding_box_collection_in_frame(camera._world.root)
    target_points_list: List[np.ndarray] = []
    for bb in bb_collection.bounding_boxes:
        # 8 corners per bounding box
        for pt in bb.get_points():
            # Convert Point3 to numpy array in world frame
            target_points_list.append(pt.to_np()[:3])

    target_points = np.asarray(target_points_list, dtype=float)
    origin_points = np.repeat(cam_origin.reshape(1, 3), len(target_points), axis=0)

    # Perform ray tests
    hit_points, hit_indices, hit_bodies = rt.ray_test(origin_points, target_points)

    occluders: list[Body] = []
    seen = set()

    # Map from local index in hit results to original ray index
    # hit_indices are the indices into the input rays that had a hit
    eps = 1e-9
    for i_result, ray_idx in enumerate(hit_indices):
        hit_body = hit_bodies[i_result]
        if hit_body is None:
            continue
        # Ignore the target body itself
        if hit_body == body:
            continue

        # Distances: camera -> first hit, camera -> target sample point
        d_hit = float(np.linalg.norm(hit_points[i_result] - origin_points[ray_idx]))
        d_target = float(
            np.linalg.norm(target_points[ray_idx] - origin_points[ray_idx])
        )

        # If the first hit is before reaching the sample point on the target, it occludes that ray
        if d_hit + eps < d_target:
            if hit_body not in seen:
                seen.add(hit_body)
                occluders.append(hit_body)

    return occluders


def reachable(
    position: Point3, manipulator: Manipulator, threshold: float = 0.05
) -> bool:
    """
    Checks if a manipulator can reach a given position. To determine this the inverse kinematics are
    calculated and applied. Afterward the distance between the position and the given manipulator is calculated, if
    it is smaller than the threshold the reasoning query returns True, if not it returns False.

    :param position: The position to reach
    :param manipulator: The manipulator that should reach for the position
    :param threshold: The threshold between the end effector and the position.
    :return: True if the end effector is closer than the threshold to the target position, False in every other case
    """
    raise NotImplementedError


def blocking(position: Point3, manipulator: Manipulator) -> Optional[List[Body]]:
    """
    Checks if any objects are blocking another object when a robot tries to pick it. This works
    similar to the reachable predicate. First the inverse kinematics between the robot and the object will be
    calculated and applied. Then it will be checked if the robot is in contact with any object except the given one.
    If the given pose or Object is not reachable None will be returned

    :param position: The position to reach
    :param manipulator: The manipulator that should reach for the position
    :return: A list of bodies the robot is in collision with when reaching for the specified object or None if the pose or object is not reachable.
    """
    raise NotImplementedError


def is_supported_by(supported_body: Body, supporting_body: Body) -> bool:
    """
    Checks if one object is supporting another object.

    :param supported_body: Object that is supported
    :param supporting_body: Object that potentially supports the first object
    :return: True if the second object is supported by the first object, False otherwise
    """
    return contact(supported_body, supporting_body) and above(
        supported_body, supporting_body, TransformationMatrix()
    )


def is_body_in_gripper(
    body: Body, gripper: ParallelGripper, sample_size: int = 100
) -> float:
    """
    Check if the body in the gripper.

    This method samples random rays between the finger and the thumb and returns the marginal probability that the rays
    intersect.

    :param body: The body for which the check should be done.
    :param gripper: The gripper for which the check should be done.
    :param sample_size: The number of rays to sample.

    :return: The percentage of rays between the fingers that hit the body.
    """

    # Retrieve meshes in local frames
    thumb_mesh = gripper.thumb.tip.combined_collision_mesh.copy()
    finger_mesh = gripper.finger.tip.combined_collision_mesh.copy()
    body_mesh = body.combined_collision_mesh.copy()

    # Transform copies of the meshes into the world frame
    body_mesh.apply_transform(body.global_pose.to_np())
    thumb_mesh.apply_transform(gripper.thumb.tip.global_pose.to_np())
    finger_mesh.apply_transform(gripper.finger.tip.global_pose.to_np())

    # get random points from thumb mesh
    finger_points = trimesh.sample.sample_surface(finger_mesh, sample_size)[0]
    thumb_points = trimesh.sample.sample_surface(thumb_mesh, sample_size)[0]

    ray_intersector = RayMeshIntersector(body_mesh)
    hit_indices = ray_intersector.intersects_id(finger_points, thumb_points)[1]
    return len(hit_indices) / sample_size


def is_body_in_region(body: Body, region: Region) -> float:
    """
    Check if the body is in the region by computing the fraction of the body's
    collision volume that lies inside the region's area volume.

    Implementation detail: both the body and region meshes are defined in their
    respective local frames; we must transform them into a common (world) frame
    using their global poses before computing the boolean intersection.

    :param body: The body for which the check should be done.
    :param region: The region to check if the body is in.
    :return: The percentage (0.0..1.0) of the body's volume that lies in the region.
    """
    # Retrieve meshes in local frames
    body_mesh_local = body.combined_collision_mesh
    region_mesh_local = region.combined_area_mesh

    # Transform copies of the meshes into the world frame
    body_mesh = body_mesh_local.copy().apply_transform(body.global_pose.to_np())
    region_mesh = region_mesh_local.copy().apply_transform(region.global_pose.to_np())
    intersection = trimesh.boolean.intersection([body_mesh, region_mesh])

    # no body volume -> zero fraction
    body_volume = body_mesh.volume
    if body_volume <= 1e-12:
        return 0.0

    return intersection.volume / body_volume


def left_of(body: Body, other: Body, point_of_view: TransformationMatrix) -> bool:
    """
    Check if the body is left of the other body if you are looking from the reference point.

    The "left" direction is taken as the -Y axis of the given reference_point.
    The comparison is done using the centers of mass computed from the bodies' collision geometry.

    :param body: The body for which the check should be done.
    :param other: The other body.
    :param point_of_view: The reference spot from where to look at the bodies.
    :return: True if the body is left of the other body, False otherwise
    """
    return _signed_distance_along_direction(body, other, point_of_view, 1) > 0.0


def right_of(body: Body, other: Body, point_of_view: TransformationMatrix) -> bool:
    """
    Check if the body is right of the other body if you are looking from the reference point.

    The "right" direction is taken as the +Y axis of the given reference_point.
    The comparison is done using the centers of mass computed from the bodies' collision geometry.

    :param body: The body for which the check should be done.
    :param other: The other body.
    :param point_of_view: The reference pose that defines the up direction for the comparison.
    :return: True if the body is right of the other body, False otherwise
    """
    return _signed_distance_along_direction(body, other, point_of_view, 1) < 0.0


def above(body: Body, other: Body, point_of_view: TransformationMatrix) -> bool:
    """
    Check if the body is above the other body with respect to the point_of_view's up direction (+Z axis).

    The "up" direction is taken as the +Z axis of the given point_of_view.
    The comparison is done using the centers of mass computed from the bodies' collision geometry.

    :param body: The body for which the check should be done.
    :param other: The other body.
    :param point_of_view: The reference pose that defines the up direction for the comparison.
    :return: True if the center of mass of "body" is above that of "other" along the point_of_view's +Z axis.
    """
    return _signed_distance_along_direction(body, other, point_of_view, 2) > 0.0


def below(body: Body, other: Body, point_of_view: TransformationMatrix) -> bool:
    """
    Check if the body is below the other body with respect to the point_of_view's up direction (+Z axis).

    The "below" direction is taken as the -Z axis of the given point_of_view.
    The comparison is done using the centers of mass computed from the bodies' collision geometry.

    :param body: The body for which the check should be done.
    :param other: The other body.
    :param point_of_view: The reference pose that defines the up direction for the comparison.
    :return: True if the center of mass of "body" is below that of "other" along the point_of_view's +Z axis.
    """
    return _signed_distance_along_direction(body, other, point_of_view, 2) < 0.0


def behind(body: Body, other: Body, point_of_view: TransformationMatrix) -> bool:
    """
    Check if the body is behind the other body if you are looking from the reference point.

    The "behind" direction is defined as the -X axis of the given reference_point.
    The comparison is done using the centers of mass computed from the bodies' collision
    geometry.

    :param body: The body for which the check should be done.
    :param other: The other body.
    :param point_of_view: The reference spot from where to look at the bodies.
    :return: True if the body is behind the other body, False otherwise
    """
    return _signed_distance_along_direction(body, other, point_of_view, 0) < 0.0


def in_front_of(body: Body, other: Body, point_of_view: TransformationMatrix) -> bool:
    """
    Check if the body is in front of another body if you are looking from the reference point.

    The "front" direction is defined as the +X axis of the given reference_point.
    The comparison is done using the centers of mass computed from the bodies' collision
    geometry.

    :param body: The body for which the check should be done.
    :param other: The other body.
    :param point_of_view: The reference spot from where to look at the bodies.
    :return: True if the body is in front of the other body, False otherwise
    """
    return _signed_distance_along_direction(body, other, point_of_view, 0) > 0.0


def _center_of_mass_in_world(b: Body) -> np.ndarray:
    """
    Compute the center of mass of an object in the world coordinate frame.
    :param b: The body to compute the center of mass of.
    :return: The bodies center of mass as a 3D array.
    """
    # Center of mass in the body's local frame (collision geometry)
    com_local = b.combined_collision_mesh.center_mass  # (3,)
    # Transform to world frame using the body's global pose
    T_bw = b.global_pose.to_np()  # body -> world
    com_h = np.array([com_local[0], com_local[1], com_local[2], 1.0], dtype=float)
    return (T_bw @ com_h)[:3]


def _signed_distance_along_direction(
    body: Body, other: Body, point_of_view: TransformationMatrix, index: int
) -> float:
    """
    Calculate the spatial relation between two bodies with respect to a given
    reference point and a specified axis index. This function computes the
    signed distance along a specified direction derived from the reference point
    to compare the positions of the centers of mass of the two bodies.

    :param body: The first body for which the spatial relation is calculated.
    :param other: The second body to which the spatial relation is compared.
    :param point_of_view: Transformation matrix that provides the reference
        frame for the calculation.
    :param index: The index of the axis in the transformation matrix along which
        the spatial relation is computed.
    :return: The signed distance between the first and the second body's centers
        of mass along the given direction. A positive result indicates that the
        first body's center of mass is further along the direction of the
        specified axis than the second body's center of mass.
    """
    ref_np = point_of_view.to_np()
    front_world = ref_np[:3, index]
    front_norm = front_world / (np.linalg.norm(front_world) + 1e-12)

    s_body = float(np.dot(front_norm, _center_of_mass_in_world(body)))
    s_other = float(np.dot(front_norm, _center_of_mass_in_world(other)))
    return s_body - s_other
