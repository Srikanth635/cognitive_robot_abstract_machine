import itertools

import numpy as np
from entity_query_language import let, an, entity, contains, and_, not_
from typing_extensions import List, Optional, Tuple

from ..spatial_computations.raytracer import RayTracer
from ..collision_checking.collision_detector import CollisionCheck
from ..collision_checking.trimesh_collision_detector import TrimeshCollisionDetector
from ..robots import RobotView, Camera, Manipulator, Finger, AbstractRobot
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

    body = let("body", type_=Body, domain=robot._world.bodies)
    possible_collisions_bodies = an(
        entity(
            body,
            and_(
                body.has_collision(),
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


def robot_holds_body(robot: RobotView, body: Body) -> bool:
    """
    Check if a robot is holding an object.

    :param robot: The robot object
    :param body: The body to check if it is picked
    :return: True if the robot is holding the object, False otherwise
    """
    raise NotImplementedError


def get_visible_objects(camera: Camera) -> List[KinematicStructureEntity]:
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
    return obj in get_visible_objects(camera)


def occluding_bodies(camera: Camera, body: Body) -> List[Body]:
    """
    Get all bodies that are occluding the given body.
    :param camera: The camera for which the occluding bodies should be returned
    :param body: The body for which the occluding bodies should be returned
    :return: A list of bodies that are occluding the given body.
    """
    raise NotImplementedError


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


def supporting(supported_body: Body, supporting_body: Body) -> bool:
    """
    Checks if one object is supporting another object.

    :param supported_body: Object that is supported
    :param supporting_body: Object that potentially supports the first object
    :return: True if the second object is supported by the first object, False otherwise
    """
    raise NotImplementedError


def is_body_between_fingers(body: Body, fingers: List[Finger]) -> bool:
    """
    Check if the body is between the fingers.

    :param body: The body for which the check should be done.
    :param fingers: The fingers that should be checked.
    :return: True if the body is between the fingers, False otherwise
    """
    raise NotImplementedError


def is_body_in_region(body: Body, region: Region) -> bool:
    """
    Check if the body is in the region.

    :param body: The body for which the check should be done.
    :param region: The region to check if the body is in.
    """
    raise NotImplementedError


def left_of(body: Body, other: Body, reference_point: TransformationMatrix) -> bool:
    """
    Check if the body is left of the other body if you are looking from the reference point.

    :param body: The body for which the check should be done.
    :param other: The other body.
    :param reference_point: The reference spot from where to look at the bodies.
    :return: True if the body is left of the other body, False otherwise
    """
    ...


def right_of(body: Body, other: Body, reference_point: TransformationMatrix) -> bool:
    """
    Check if the body is right of the other body if you are looking from the reference point.

    :param body: The body for which the check should be done.
    :param other: The other body.
    :return: True if the body is right of the other body, False otherwise
    """
    ...


def above(body: Body, other: Body, point_of_view: TransformationMatrix) -> bool:
    """
    Check if the body is above the other body with respect to the point_of_view's up direction (+Z axis).

    The "up" direction is taken as the +Z axis of the given point_of_view.

    :param body: The body for which the check should be done.
    :param other: The other body.
    :param point_of_view: The reference pose that defines the up direction for the comparison.
    :return: True if the lowest point of `body` along the up direction is higher than the highest point of `other`.
    """
    assert body._world == other._world, "Both bodies must be in the same world"

    world = body._world
    reference_frame = world.root  # Use a consistent frame for coordinates

    # Get AABBs in the chosen reference frame
    body_bbs = body.as_bounding_box_collection_in_frame(reference_frame)
    other_bbs = other.as_bounding_box_collection_in_frame(reference_frame)

    if not body_bbs.bounding_boxes or not other_bbs.bounding_boxes:
        return False

    # Up vector from the point_of_view (+Z axis)
    R = point_of_view.to_np()[:3, :3]
    up = R[:, 2]
    up_norm = up / (np.linalg.norm(up) + 1e-12)

    body_min, _ = _proj_min_max(body_bbs, up_norm)
    _, other_max = _proj_min_max(other_bbs, up_norm)

    eps = 1e-6
    return (body_min - other_max) > eps


def below(body: Body, other: Body, point_of_view: TransformationMatrix) -> bool:
    """
    Check if the body is below the other body with respect to the point_of_view's up direction (+Z axis).

    The "below" direction is taken as the -Z axis of the given point_of_view.

    :param body: The body for which the check should be done.
    :param other: The other body.
    :param point_of_view: The reference pose that defines the up direction for the comparison.
    :return: True if the highest point of `body` along the up direction is lower than the lowest point of `other`.
    """
    assert body._world == other._world, "Both bodies must be in the same world"

    world = body._world
    reference_frame = world.root

    # Get AABBs in the chosen reference frame
    body_bbs = body.as_bounding_box_collection_in_frame(reference_frame)
    other_bbs = other.as_bounding_box_collection_in_frame(reference_frame)

    if not body_bbs.bounding_boxes or not other_bbs.bounding_boxes:
        return False

    # Up vector from the point_of_view (+Z axis)
    R = point_of_view.to_np()[:3, :3]
    up = R[:, 2]
    up_norm = up / (np.linalg.norm(up) + 1e-12)

    _, body_max = _proj_min_max(body_bbs, up_norm)
    other_min, _ = _proj_min_max(other_bbs, up_norm)

    eps = 1e-6
    return (other_min - body_max) > eps


def behind(body: Body, other: Body, reference_point: TransformationMatrix) -> bool:
    """
    Check if the body is behind the other body if you are looking from the reference point.

    :param body: The body for which the check should be done.
    :param other: The other body.
    :param reference_point: The reference spot from where to look at the bodies.
    :return: True if the body is behind the other body, False otherwise
    """


def in_front_of(body: Body, other: Body, reference_point: TransformationMatrix) -> bool:
    """
    Check if the body is in front of another body if you are looking from the reference point.

    :param body: The body for which the check should be done.
    :param other: The other body.
    :param reference_point: The reference spot from where to look at the bodies.
    :return: True if the body is in front of the other body, False otherwise
    """


def _proj_min_max(
    bounding_boxes: BoundingBoxCollection, up_norm: np.ndarray
) -> Tuple[float, float]:
    """
    Projects the bounding boxes onto a given normalized direction vector and calculates
    the minimum and maximum projections along that vector.

    This function iterates over all vertices of the given bounding boxes, projecting each
    vertex onto the specified direction vector. It computes the minimum and maximum
    projection values among all the vertices.

    :param bounding_boxes: The bounding boxes to project onto.
    :param up_norm: The normalized direction 3D vector.

    :return: The minimum and maximum projection values.
    """
    s_min = float("inf")
    s_max = float("-inf")
    ux, uy, uz = up_norm
    for bb in bounding_boxes:
        for x in (bb.min_x, bb.max_x):
            for y in (bb.min_y, bb.max_y):
                for z in (bb.min_z, bb.max_z):
                    s = x * ux + y * uy + z * uz
                    if s < s_min:
                        s_min = s
                    if s > s_max:
                        s_max = s
    return s_min, s_max
