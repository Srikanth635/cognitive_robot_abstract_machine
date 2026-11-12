"""
The names the viewer reads in the database the NEEMs are kept in, and in the statecharts
kept there.

The ORM of cramph generates the tables, so the names are those of its classes; a
statechart is kept as the JSON its classes write of themselves.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum

# %% the tables


class Table(StrEnum):
    """
    The tables a NEEM database has, which the viewer reads.
    """

    STATECHART = "StatechartMappingDAO"
    """
    One row for each NEEM: the statechart the plan was performed as.
    """

    CONTEXT = "StatechartContextDAO"
    """
    What a statechart was performed in, such as how long a tick of it lasted.
    """

    TICK = "NodeStatesAtTickDAO"
    """
    The life cycle of every node of a statechart at a tick on which one of them changed.
    """


class WorldTable(StrEnum):
    """
    The tables of the things of a world, which say what kind of thing a step refers to.
    """

    ENTITY = "WorldEntityDAO"
    """
    The table every thing of a world extends, saying which kind each is.
    """

    ENTITY_WITH_ID = "WorldEntityWithIDDAO"
    """
    The identifier each thing of a world is known by.
    """


DAO_SUFFIX = "DAO"
"""
What ends the name of every table, after the name of the class it keeps.
"""


class Column(StrEnum):
    """
    The columns of those tables the viewer reads.
    """

    IDENTIFIER = "database_id"
    """
    What identifies a row, in every table.
    """

    STRUCTURE = "structure"
    """
    The nodes of a statechart, as JSON.
    """

    CONTEXT = "_context_id"
    """
    What a statechart was performed in.
    """

    TICK_DURATION = "tick_duration"
    """
    How many seconds a tick lasted.
    """

    TICK_COUNT = "tick_count"
    """
    The tick the life cycles were taken at.
    """

    LIFE_CYCLES = "life_cycle_values"
    """
    The life cycle of each node, in the order of the nodes.
    """

    POLYMORPHIC_TYPE = "polymorphic_type"
    """
    In the table at the root of a family, the table of the class a row really is.
    """

    ENTITY_ID = "id"
    """
    The identifier a thing of a world is known by.
    """


# %% the statechart as JSON


class JsonKey(StrEnum):
    """
    The keys of the JSON a statechart is kept as.
    """

    TYPE = "__json_type__"
    """
    The class of an object, as its module and its name.
    """

    COLLECTION = "__collection_type__"
    """
    Marks a list of values.
    """

    ITEMS = "__items__"
    """
    The values of a list.
    """

    NODES = "nodes"
    """
    The nodes of the statechart, in the order of their indices.
    """

    CHILDREN = "children"
    """
    The indices of the nodes directly below each node, by the index of that node.
    """

    PARENT = "parent_node_index"
    """
    The index of the node a node is below; none for a node at the top.
    """

    NODE_ID = "node_id"
    """
    The identifier of a node.
    """

    NAME = "name"
    """
    The name of a node, of a member of an enumeration, or what a world calls a thing.
    """

    PLOT = "plot_specifications"
    """
    How a node is drawn.
    """

    ID = "id"
    """
    The identifier of a thing of a world.
    """

    VALUE = "value"
    """
    The text of an identifier.
    """

    FRAME_NAME = "reference_frame_name"
    """
    The name of the body a pose is measured from.
    """

    POSITION = "position"
    """
    The coordinates of a pose.
    """

    ROTATION = "rotation"
    """
    The orientation of a pose, as a quaternion.
    """

    @property
    def places_a_node(self) -> bool:
        """
        Whether the key of a node places it in the statechart rather than says what it
        was asked to do.
        """
        return self in (
            JsonKey.TYPE,
            JsonKey.NAME,
            JsonKey.PARENT,
            JsonKey.NODE_ID,
            JsonKey.PLOT,
        )


class StoredClass(StrEnum):
    """
    The classes whose objects the viewer reads as more than values.
    """

    WORLD_THING = (
        "semantic_digital_twin.world_description.world_entity.ReferencedWorldEntity"
    )
    """
    A thing of a world, by its identifier and its name.
    """

    POSE = "krrood_spatial_math.spatial_types.Pose"
    """
    Where something is and which way it faces.
    """

    IDENTIFIER = "uuid.UUID"
    """
    An identifier.
    """


class ModulePrefix(StrEnum):
    """
    What begins the modules of a kind of node.
    """

    ACTION = "coraplex.robot_plans.actions."
    """
    The modules of the actions, which tell an action from the other nodes.
    """


class LifeCycle(IntEnum):
    """
    Where a node is in its life cycle, as the statechart numbers it.
    """

    NOT_STARTED = 0
    RUNNING = 1
    PAUSED = 2
    SUCCEEDED = 3
    FAILED = 4
    INTERRUPTED = 5

    @property
    def is_terminal(self) -> bool:
        """
        Whether a node in this state has ended.
        """
        return self >= LifeCycle.SUCCEEDED
