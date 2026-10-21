from __future__ import annotations

from dataclasses import dataclass, field

from krrood.entity_query_language.query.match import Match
from llm_reasoner.resolver import CandidateRenderer, PromptContextProvider
from llm_reasoner.resolver import Slot, SlotKind, reflect_slots
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    AbstractRobotPart,
    EndEffector,
)
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageDescription,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    SemanticAnnotation,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose, Pose2D
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Any, List, Optional

from coraplex.datastructures.enums import Arms
from coraplex.datastructures.grasp import GraspDescription
from coraplex.querying.grasps import describe_grasp_options
from coraplex.querying.predicates import GripperIsFree
from coraplex.querying.spatial import describe_relative_position, relative_position
from coraplex.view_manager import ViewManager

# %% numeric rendering


def format_coordinates(symbolic_vector: Any, labels: List[str]) -> str:
    """
    Render a symbolic spatial value as plain numbers.

    Spatial types hold symbolic expressions rather than floats, so their representation
    is a matrix of solver placeholders. ``to_np`` resolves them.

    :param symbolic_vector: A spatial value.
    :param labels: One label per component, in order.
    :return: The components as ``label=value`` pairs.
    """
    values = symbolic_vector.to_np().flatten().tolist()
    return ", ".join(f"{label}={value:.2f}" for label, value in zip(labels, values))


def format_dimensions(body: Body) -> Optional[str]:
    """
    Measure a body's collision geometry.

    Size decides which approach directions and which end effectors are viable, and it is
    knowledge no schema carries.

    :param body: The body to measure.
    :return: Its extent in metres, or ``None`` when it has no collision geometry.
    """
    if not body.collision.shapes:
        return None
    dimensions = (
        body.collision.as_bounding_box_collection_in_frame(body)
        .bounding_box()
        .dimensions
    )
    depth, width, height = dimensions
    return f"{depth:.2f} x {width:.2f} x {height:.2f} m (depth x width x height)"


def widest_horizontal_extent(body: Body) -> Optional[float]:
    """
    :param body: The body to measure.
    :return: The larger of its depth and width, which is what a gripper must span, or
        ``None`` when it has no collision geometry.
    """
    if not body.collision.shapes:
        return None
    dimensions = (
        body.collision.as_bounding_box_collection_in_frame(body)
        .bounding_box()
        .dimensions
    )
    return max(dimensions[0], dimensions[1])


# %% robot lookups


def end_effectors_of(robot: AbstractRobot) -> List[EndEffector]:
    """
    Collect the end effectors of one robot.

    Scoped to a single robot rather than the whole world, because a world may hold
    several robots while only one is acting.

    :param robot: The acting robot.
    :return: Its end effectors, without duplicates, in left-then-right order.
    """
    found: List[EndEffector] = []
    for arm in (Arms.LEFT, Arms.RIGHT):
        end_effector = ViewManager.get_end_effector_view(arm, robot)
        if end_effector is not None and end_effector not in found:
            found.append(end_effector)
    return found


def bodies_of(robot: AbstractRobot) -> List[Body]:
    """
    Collect every body belonging to a robot.

    A robot's own links are annotated exactly like scene objects are, so without this
    the robot's fingers and torso appear in the prompt as things it might manipulate.

    :param robot: The robot whose kinematic subtree to walk.
    :return: Its bodies, root included.
    """
    world = robot.root._world
    collected: List[Body] = []
    frontier = [robot.root]
    while frontier:
        entity = frontier.pop()
        if entity in collected:
            continue
        if isinstance(entity, Body):
            collected.append(entity)
        frontier.extend(world.compute_child_kinematic_structure_entities(entity))
    return collected


def is_manipulable(body: Body) -> bool:
    """
    Decide whether a body is a thing rather than a frame.

    Coordinate frames such as ``map`` and ``odom_combined`` are bodies with no geometry.
    They are never what an action is about, and offering one as the target would be
    confidently wrong.

    :param body: The body to test.
    :return: Whether it has collision geometry.
    """
    return bool(body.collision.shapes)


def held_body(end_effector: EndEffector) -> Optional[Body]:
    """
    Find what an end effector is currently holding.

    :param end_effector: The end effector to inspect.
    :return: The held body, or ``None`` when the end effector is empty.
    """
    attached = end_effector.tool_frame.child_kinematic_structure_entities
    if not attached:
        return None
    return attached[0]


def occupancy_of(end_effector: EndEffector) -> str:
    """
    :param end_effector: The end effector to inspect.
    :return: Whether it is free, and what it holds when it is not.
    """
    if GripperIsFree(end_effector=end_effector)():
        return "free"
    holding = held_body(end_effector)
    if holding is None:
        return "occupied"
    return f"holding {holding.name.name}"


# %% world lookups


def natural_language_description_of(world: World, body: Body) -> Optional[str]:
    """
    Find the natural-language annotation attached to a body.

    :param world: The world holding the annotations.
    :param body: The body to describe.
    :return: The description, or ``None`` when the body carries no annotation.
    """
    for annotation in world.get_semantic_annotations_by_type(
        NaturalLanguageDescription
    ):
        if annotation.root is body:
            return annotation.description
    return None


def annotations_rooted_at(world: World, body: Body) -> List[HasRootBody]:
    """
    Find every semantic annotation whose root is a body.

    The annotation types tell the model what a body *is* -- a ``Milk`` rather than an
    anonymous mesh -- which is the sortal knowledge a language model reasons with.

    :param world: The world holding the annotations.
    :param body: The body to look up.
    :return: The annotations rooted at that body.
    """
    return [
        annotation
        for annotation in world.get_semantic_annotations_by_type(HasRootBody)
        if annotation.root is body
    ]


def supporting_body_of(body: Body) -> Optional[Body]:
    """
    Find what a body rests on or is attached to.

    :param body: The body to look up.
    :return: Its parent in the kinematic structure, or ``None`` when it has none.
    """
    parent_connection = body.parent_connection
    if parent_connection is None:
        return None
    return parent_connection.parent


def body_nearest_to(
    world: World,
    pose: Pose,
    excluded: Optional[List[Body]] = None,
    tolerance: float = 0.05,
) -> Optional[Body]:
    """
    Identify the body a pose refers to.

    A designator states a target as coordinates, but what matters for choosing an
    approach direction or a gripper is *which object* stands there. This recovers the
    identity that the pose alone omits.

    Only bodies with geometry are considered, and the caller may exclude the robot's
    own links. A wrong answer here is worse than none, because the model has no way to
    tell that the object it was told about is not the object being reached for.

    :param world: The world to search.
    :param pose: The pose to resolve.
    :param excluded: Bodies that cannot be the target, typically the robot's own.
    :param tolerance: Largest distance in metres still counted as the same place.
    :return: The nearest body within the tolerance, or ``None`` when there is none.
    """
    excluded = excluded or []
    target = pose.to_position()[:3].to_np().flatten()
    nearest: Optional[Body] = None
    nearest_distance = tolerance
    for body in world.bodies:
        if body in excluded or not is_manipulable(body):
            continue
        position = body.global_pose.to_position()[:3].to_np().flatten()
        distance = float(sum((target - position) ** 2) ** 0.5)
        if distance <= nearest_distance:
            nearest, nearest_distance = body, distance
    return nearest


def describe_body(
    world: World, body: Body, observer: Optional[AbstractRobot] = None
) -> str:
    """
    Describe a body with everything that bears on manipulating it.

    When an observer is given, the body's position is stated relative to it rather than
    in world coordinates, because which arm to use and how high to raise the torso
    follow from the relation, not from the absolute numbers.

    :param world: The world the body belongs to.
    :param body: The body to describe.
    :param observer: The robot the position is expressed relative to.
    :return: A single-line description.
    """
    parts = [body.name.name]

    annotation_names = [
        type(annotation).__name__
        for annotation in annotations_rooted_at(world, body)
        if not isinstance(annotation, NaturalLanguageDescription)
    ]
    if annotation_names:
        parts.append("a " + ", a ".join(annotation_names))

    description = natural_language_description_of(world, body)
    if description:
        parts.append(f'"{description}"')

    dimensions = format_dimensions(body)
    if dimensions:
        parts.append(dimensions)

    supporter = supporting_body_of(body)
    if supporter is not None:
        parts.append(f"on {supporter.name.name}")

    if observer is None:
        position = format_coordinates(
            body.global_pose.to_position()[:3], ["x", "y", "z"]
        )
        parts.append(f"at {position}")
    else:
        parts.append(describe_relative_position(world, observer.root, body))
    return "  ".join(parts)


# %% task-relative candidate description


@dataclass
class SituationCandidateRenderer(CandidateRenderer):
    """
    Describes world individuals offered as candidates for a referent slot.

    Renders what distinguishes one candidate from another, including its fitness for
    the query at hand: an end effector is described against the object being reached
    for, because that comparison is what decides between two otherwise identical
    grippers.

    .. note::
        This exists because krrood's verbalization renders *expressions*, not
        individuals: there is no path from a live world entity to English. Once krrood
        offers one, this renderer delegates to it and the type-by-type handling here
        goes away.
    """

    world: World
    """The world the candidates belong to."""

    robot: Optional[AbstractRobot] = None
    """The acting robot, whose own links are never the target of a query."""

    def render(self, candidate: Any, expression: Match) -> str:
        if isinstance(candidate, EndEffector):
            return self._render_end_effector(candidate, expression)
        if isinstance(candidate, SemanticAnnotation):
            return self._render_annotation(candidate)
        if isinstance(candidate, Body):
            return describe_body(self.world, candidate, observer=self.robot)
        if isinstance(candidate, Pose):
            return self._render_pose(candidate)
        if isinstance(candidate, Pose2D):
            return self._render_pose_2d(candidate)
        if candidate is None:
            return "none"
        return f"{type(candidate).__name__}  {candidate!r}"

    def _render_end_effector(
        self, candidate: EndEffector, expression: Match
    ) -> str:
        """
        :param candidate: An end effector offered as a candidate.
        :param expression: The match being resolved.
        :return: Its name, occupancy and, when a target is known, how its reach relates
            to the object.
        """
        parts = [candidate.name.name, occupancy_of(candidate)]
        if self.robot is None:
            return "  ".join(parts)

        mounting = relative_position(
            self.world, self.robot.root, candidate.tool_frame
        )
        if mounting.side is not None:
            parts.append(f"mounted on the robot's {mounting.side}")

        target = target_body_of(
            self.world, expression, excluded=bodies_of(self.robot)
        )
        if target is None:
            return "  ".join(parts)

        extent = widest_horizontal_extent(target)
        if extent is not None:
            parts.append(f"must span {extent:.2f} m to grasp {target.name.name}")
        return "  ".join(parts)

    def _render_annotation(self, candidate: SemanticAnnotation) -> str:
        """
        :param candidate: A semantic annotation offered as a candidate.
        :return: Its name, type and the description of its root body.
        """
        parts = [candidate.name.name, f"a {type(candidate).__name__}"]
        description = natural_language_description_of(self.world, candidate.root)
        if description:
            parts.append(f'"{description}"')
        dimensions = format_dimensions(candidate.root)
        if dimensions:
            parts.append(dimensions)
        if self.robot is not None:
            parts.append(
                describe_relative_position(self.world, self.robot.root, candidate.root)
            )
        return "  ".join(parts)

    @staticmethod
    def _render_pose(candidate: Pose) -> str:
        """
        :param candidate: A pose.
        :return: Its position and the frame it is expressed in.
        """
        position = format_coordinates(candidate.to_position()[:3], ["x", "y", "z"])
        frame = candidate.reference_frame
        if frame is None:
            return f"Pose  {position}"
        return f"Pose  {position}  relative to {frame.name.name}"

    @staticmethod
    def _render_pose_2d(candidate: Pose2D) -> str:
        """
        :param candidate: A planar pose.
        :return: Its coordinates and the frame it is expressed in.
        """
        coordinates = format_coordinates(candidate, ["x", "y", "yaw"])
        frame = candidate.reference_frame
        if frame is None:
            return f"Pose2D  {coordinates}"
        return f"Pose2D  {coordinates}  relative to {frame.name.name}"


# %% recovering the target from the query


def target_body_of(
    world: World, expression: Match, excluded: Optional[List[Body]] = None
) -> Optional[Body]:
    """
    Work out which body a query is about.

    A designator names its target either directly, as a bound body or annotation, or
    indirectly, as a bound pose. Both are recovered here, because almost every choice
    the model makes -- approach direction, alignment, which gripper -- depends on the
    object rather than on the schema.

    :param world: The world to search.
    :param expression: The match being resolved.
    :param excluded: Bodies that cannot be the target, typically the robot's own.
    :return: The body the query targets, or ``None`` when none can be identified.
    """
    poses: List[Pose] = []
    for slot in reflect_slots(expression):
        if slot.kind is not SlotKind.BOUND:
            continue
        value = slot.bound_value
        if isinstance(value, Body):
            return value
        if isinstance(value, SemanticAnnotation):
            return value.root
        if isinstance(value, Pose):
            poses.append(value)
    for pose in poses:
        body = body_nearest_to(world, pose, excluded=excluded)
        if body is not None:
            return body
    return None


def constructs_a_grasp(expression: Match) -> bool:
    """
    :param expression: The match being resolved.
    :return: Whether any slot of the match belongs to a grasp description, which is
        what makes the permitted grasps worth computing.
    """
    return any(
        slot.owner is GraspDescription for slot in reflect_slots(expression)
    )


# %% situational knowledge


@dataclass
class SituationContextProvider(PromptContextProvider):
    """
    Describes the current situation so a model can choose between valid alternatives.

    Reports the object the query is about, the acting robot's end effectors and what
    they hold, and the annotated bodies in the scene. None of this is derivable from a
    query's schema, and all of it bears on which parameter values make sense.
    """

    world: World
    """The world being described."""

    robot: AbstractRobot
    """The robot the plan is executed by."""

    max_bodies: int = 20
    """Most scene bodies listed before the description is truncated."""

    describe_target: bool = True
    """Whether to identify and describe the body the query targets."""

    describe_scene: bool = True
    """Whether to list the annotated bodies in the scene."""

    describe_grasps: bool = True
    """Whether to state which grasps the object's orientation permits."""

    def context_for(self, expression: Match) -> str:
        robot_bodies = bodies_of(self.robot)
        target = target_body_of(self.world, expression, excluded=robot_bodies)
        sections = [
            self._target_section(target),
            self._grasp_section(expression, target),
            self._robot_section(),
            self._scene_section(target, robot_bodies),
        ]
        return "\n\n".join(section for section in sections if section)

    def _grasp_section(self, expression: Match, target: Optional[Body]) -> str:
        """
        State which grasps the object's orientation permits.

        Approach directions and vertical alignments are expressed in the object's own
        frame, so without this the model would be choosing a face of a frame it cannot
        see. Reporting them as context rather than narrowing the slots keeps the choice
        between permitted options -- which of them is convenient, which suits a handle
        -- with the model.

        :param expression: The match being resolved.
        :param target: The body the query is about, when one was identified.
        :return: The permitted grasps, or nothing when they do not apply.
        """
        if not self.describe_grasps or target is None:
            return ""
        if not constructs_a_grasp(expression):
            return ""
        return describe_grasp_options(end_effectors_of(self.robot), target)

    def _target_section(self, target: Optional[Body]) -> str:
        """
        :param target: The body the query is about, when one was identified.
        :return: A description of it.
        """
        if not self.describe_target or target is None:
            return ""
        return "The object being acted on:\n- " + describe_body(
            self.world, target, observer=self.robot
        )

    def _robot_section(self) -> str:
        """:return: The acting robot's end effectors and what they hold."""
        end_effectors = end_effectors_of(self.robot)
        if not end_effectors:
            return ""
        lines = [
            f"- {end_effector.name.name}: {occupancy_of(end_effector)}"
            for end_effector in end_effectors
        ]
        return (
            f"Positions below are given relative to {self.robot.name.name}, "
            f"which faces along its own forward axis.\n"
            f"Its end effectors:\n" + "\n".join(lines)
        )

    def _scene_section(self, target: Optional[Body], robot_bodies: List[Body]) -> str:
        """
        Describe the objects the robot could act on.

        Robot parts are annotated exactly like scene objects, so they are excluded
        explicitly: a gripper finger listed among the objects invites the model to
        treat it as one.

        :param target: The body already described, so it is not repeated.
        :param robot_bodies: The acting robot's own links.
        :return: The other manipulable bodies in the world.
        """
        if not self.describe_scene:
            return ""
        described: List[Body] = []
        for annotation in self.world.get_semantic_annotations_by_type(HasRootBody):
            if isinstance(annotation, (AbstractRobot, AbstractRobotPart)):
                continue
            root = annotation.root
            if not isinstance(root, Body) or not is_manipulable(root):
                continue
            if root is target or root in robot_bodies or root in described:
                continue
            described.append(root)
        if not described:
            return ""
        listed = described[: self.max_bodies]
        lines = [
            f"- {describe_body(self.world, body, observer=self.robot)}"
            for body in listed
        ]
        section = "Other objects in the scene:\n" + "\n".join(lines)
        if len(described) > len(listed):
            section += f"\n- ... and {len(described) - len(listed)} more"
        return section