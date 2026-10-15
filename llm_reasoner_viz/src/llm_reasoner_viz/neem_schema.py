"""
The names the viewer reads in the database the NEEMs are kept in.

The ORM of coraplex generates the tables, so the names are those of its classes.
"""

from __future__ import annotations

from enum import StrEnum


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
