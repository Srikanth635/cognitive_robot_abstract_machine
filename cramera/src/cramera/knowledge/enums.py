"""
Enums shared across the knowledge package.
"""

from __future__ import annotations

from enum import StrEnum

from coraplex.robot_plans.actions.base import Action
from cramph.node import StatechartNode
from cramph.world_modification_nodes import MoveBranch
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from typing_extensions import Any, Dict, Optional, Tuple, Union


# %% viewer entity identifiers
class SceneEntityPrefix(StrEnum):
    """
    Prefixes distinguishing viewer entity identifiers from plain object names.
    """

    URDF_LINK = "urdf:"
    """
    A robot-model link addressed by the name stored in its URDF.
    """


class JointRegion(StrEnum):
    """
    Which region of the robot/scene a joint belongs to, as inferred from its name.

    Arms and grippers use :class:`cramera.robot_parts.ArmSide` instead, since they
    always name a specific arm; a joint can also belong to the robot's body or to the
    environment, which :class:`~cramera.robot_parts.ArmSide` has no member for.
    """

    LEFT = "left"
    RIGHT = "right"
    BODY = "body"
    ENVIRONMENT = "environment"


class NodeGroup(StrEnum):
    """
    Colour group of a graph-panel node.
    """

    ROBOT = "robot"
    OBJECT = "object"
    EVENT = "event"
    ROOT = "root"
    PACKAGE = "package"
    SUBPACKAGE = "subpackage"
    PLAN = "plan"
    PYTHON_CLASS = "python_class"
    EXTERNAL_CLASS = "external_class"
    OTHER = "other"


class EdgeKind(StrEnum):
    """
    Rendering kind of a graph-panel edge.
    """

    PROPERTY = "property"
    TYPE = "type"


class LabelledGroup(StrEnum):
    """
    A colour group that carries the text its legend row shows.

    Keeping both on the member is what stops a view from listing its groups twice: once
    to classify nodes and once to build the legend.
    """

    def __new__(cls, value: str, label: str) -> "LabelledGroup":
        """
        Bind a serialized group value to its legend text.

        :param value: Group identifier stored in the graph payload.
        :param label: Human-readable legend text.
        """
        member = str.__new__(cls, value)
        member._value_ = value
        member.label = label
        return member

    @classmethod
    def legend(cls) -> Tuple["LabelledGroup", ...]:
        """
        Every group of this kind, in declaration order, for a view's legend.
        """
        return tuple(cls)


class KinematicChainGroup(LabelledGroup):
    """
    Colour group of a link in the robot's kinematic tree.

    Separate from :class:`NodeGroup`, whose members name ontological categories of the
    knowledge graph: a right arm is not an "event", it just needs a colour of its own.
    """

    BASE = ("base", "Base / torso")
    LEFT_ARM = ("left_arm", "Left arm")
    RIGHT_ARM = ("right_arm", "Right arm")
    GRIPPER = ("gripper", "Grippers")
    SENSOR = ("sensor", "Head / sensors")


class PlanNodeGroup(LabelledGroup):
    """
    Colour group of a node in the executed plan tree.

    Separate from :class:`NodeGroup` for the same reason as
    :class:`KinematicChainGroup`: a motion is not a robot and a condition is not a goal,
    they are kinds of plan node that each need a colour of their own.
    """

    ACTION = ("action", "Action")
    MOTION = ("motion", "Motion")
    CONDITION = ("condition", "Condition")
    ATTACHMENT = ("attachment", "Attach / detach")
    OTHER = ("other_plan_node", "Other plan node")

    @classmethod
    def of_plan_node_kind(cls, kind: Optional[str]) -> "PlanNodeGroup":
        """
        The group a plan-node class of the plan trees coraplex had before its plans
        became statecharts belongs to.

        :param kind: The plan node's own class name, as recorded.
        """
        return {
            "ActionNode": cls.ACTION,
            "MotionNode": cls.MOTION,
            "ConditionNode": cls.CONDITION,
            "AttachNode": cls.ATTACHMENT,
            "DetachNode": cls.ATTACHMENT,
        }.get(kind or "", cls.OTHER)

    @classmethod
    def from_recorded_plan_node(cls, recorded: Dict[str, Any]) -> "PlanNodeGroup":
        """
        :param recorded: A plan node as a recording keeps it.
        :return: The group the node was recorded in, or, for a recording that kept no
            group, the group of its kind.
        """
        if "group" in recorded:
            return cls(recorded["group"])
        return cls.of_plan_node_kind(recorded.get("kind"))

    @classmethod
    def from_statechart_node(cls, node: StatechartNode) -> "PlanNodeGroup":
        """
        :param node: A node of the statechart a plan is performed as.
        :return: The group of the node: an action, a motion (a node of a motion or a
            composite of motions only), a change of the world's structure, or another.
        """
        if isinstance(node, Action):
            return cls.ACTION
        if isinstance(node, MoveBranch):
            return cls.ATTACHMENT
        if isinstance(node, MotionStatechartNode):
            return cls.MOTION
        if node.children and all(
            cls.from_statechart_node(child) is cls.MOTION for child in node.children
        ):
            return cls.MOTION
        return cls.OTHER


ColourGroup = Union[NodeGroup, KinematicChainGroup, PlanNodeGroup]
"""
Any colour group a graph-panel node can carry: an ontological one from the knowledge
graph, or a kinematic-chain one from the URDF tree.
"""
