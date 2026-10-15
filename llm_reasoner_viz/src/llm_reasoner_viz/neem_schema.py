"""
The names the viewer reads in the database the NEEMs are kept in.

The ORM of coraplex generates the tables, so the names are those of its classes.
"""

from __future__ import annotations

from enum import StrEnum

DAO_SUFFIX = "DAO"
"""
What ends the name of every table, after the name of the class it keeps.
"""


class Table(StrEnum):
    """
    The tables of the database the viewer reads.
    """

    PLAN = "PlanMappingDAO"
    """
    One row for each plan; its root is the first step of it.
    """

    NODE = "PlanNodeDAO"
    """
    One row for each node of a plan, whatever it is, with how it ended.
    """

    EDGE = "PlanEdgeDAO"
    """
    One row for each node that is below another.
    """

    ACTION_NODE = "ActionNodeDAO"
    """
    The nodes that perform an action.
    """

    EXECUTION_DATA = "ExecutionDataDAO"
    """
    What the execution of an action recorded, such as the state of the world before it.
    """

    ENTITY = "PlanEntityDAO"
    """
    The table every kind of node extends, saying which kind each is.
    """

    DESIGNATOR = "DesignatorDAO"
    """
    The table every action and motion extends, holding what the node was asked to do.
    """

    WORLD_ENTITY = "WorldEntityDAO"
    """
    The table every body and annotation of a world extends, naming it.
    """

    NAME = "PrefixedNameDAO"
    """
    The names of the things of a world.
    """

    SPATIAL = "SpatialTypeDAO"
    """
    The table points, orientations and poses extend, saying which frame they are in.
    """

    POSE = "PoseMappingDAO"
    """
    Where something is: a position and an orientation.
    """

    POINT = "Point3MappingDAO"
    """
    A position.
    """

    QUATERNION = "QuaternionMappingDAO"
    """
    An orientation.
    """


class Column(StrEnum):
    """
    The columns of those tables the viewer reads.
    """

    IDENTIFIER = "database_id"
    """
    What identifies a row, in every table.
    """

    ROOT = "_root_id"
    """
    The node a plan begins with.
    """

    PARENT = "_parent_id"
    """
    The node above, in an edge.
    """

    CHILD = "_child_id"
    """
    The node below, in an edge.
    """

    POLYMORPHIC_TYPE = "polymorphic_type"
    """
    In the table at the root of a family, the table of the class a row really is.
    """

    NAME = "name"
    """
    What a thing of a world is called.
    """

    NAME_REFERENCE = "_name_id"
    """
    The name of a thing of a world.
    """

    FRAME = "_reference_frame_id"
    """
    The body a position, an orientation or a pose is measured from.
    """

    POSITION = "_position_id"
    """
    The position of a pose.
    """

    ORIENTATION = "_orientation_id"
    """
    The orientation of a pose.
    """

    X = "x"
    """
    The first coordinate of a position, or part of an orientation.
    """

    Y = "y"
    """
    The second coordinate of a position, or part of an orientation.
    """

    Z = "z"
    """
    The third coordinate of a position, or part of an orientation.
    """

    W = "w"
    """
    The fourth part of an orientation.
    """

    STATUS = "status"
    """
    How a node ended.
    """

    STARTED = "start_time"
    """
    When a node began.
    """

    ENDED = "end_time"
    """
    When a node ended.
    """
