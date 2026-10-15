"""
The steps of a plan as the NEEM database keeps them: the tree they form, what each was
asked to do, and the things and places it refers to.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from typing_extensions import Any, Dict, List, Optional, Tuple

from llm_reasoner_viz.neem_schema import Column, Table
from llm_reasoner_viz.neem_tables import TableColumn, TableSchema

BOOKKEEPING_TABLES = (Table.DESIGNATOR, Table.EXECUTION_DATA)
"""
The tables whose rows a step refers to for the plan's own sake: what it was asked to do,
which is read as the step's values, and what its execution recorded.
"""

# %% what can go wrong


@dataclass
class MissingRow(LookupError):
    """
    A row that another row refers to is not in the database.
    """

    table: str
    """
    The table the row should be in.
    """

    identifier: int
    """
    The identifier the row should have.
    """

    def __post_init__(self):
        super().__init__(
            f"the NEEM database has no row {self.identifier} in {self.table}"
        )


# %% what a step holds


@dataclass(frozen=True)
class Parameter:
    """
    A value a step was asked for.
    """

    name: str
    """
    What the value is, as the class of the step calls it.
    """

    value: Any
    """
    The value as stored: a number, text, true or false, or none when it was not set.

    A choice from an enumeration is stored as the path of the member.
    """

    def to_json(self) -> Dict[str, Any]:
        """
        :return: The parameter as plain JSON.
        """
        return {"name": self.name, "value": self.value}


@dataclass(frozen=True)
class Position:
    """
    A place in space.
    """

    x: float
    """
    The first coordinate.
    """

    y: float
    """
    The second coordinate.
    """

    z: float
    """
    The third coordinate.
    """

    def to_json(self) -> Dict[str, float]:
        """
        :return: The position as plain JSON.
        """
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass(frozen=True)
class Orientation:
    """
    Which way something faces, as a quaternion.
    """

    x: float
    """
    The first part.
    """

    y: float
    """
    The second part.
    """

    z: float
    """
    The third part.
    """

    w: float
    """
    The fourth part.
    """

    def to_json(self) -> Dict[str, float]:
        """
        :return: The orientation as plain JSON.
        """
        return {"x": self.x, "y": self.y, "z": self.z, "w": self.w}


@dataclass(frozen=True)
class Pose:
    """
    Where something is and which way it faces, measured from a frame.
    """

    frame: Optional[str]
    """
    The name of the body it is measured from; none when the pose names none.
    """

    position: Optional[Position]
    """
    Where it is; none when the pose has no position.
    """

    orientation: Optional[Orientation]
    """
    Which way it faces; none when the pose has no orientation.
    """

    def to_json(self) -> Dict[str, Any]:
        """
        :return: The pose as plain JSON.
        """
        return {
            "frame": self.frame,
            "position": self.position and self.position.to_json(),
            "orientation": self.orientation and self.orientation.to_json(),
        }


@dataclass(frozen=True)
class Reference:
    """
    A thing or a place a step refers to.
    """

    name: str
    """
    What the step uses it as, such as ``target_container``.
    """

    type: str
    """
    What kind of thing it is, such as ``Body``.
    """

    identifier: int
    """
    The identifier the database gave it.
    """

    label: Optional[str]
    """
    What the world calls it; none for what a world does not name.
    """

    pose: Optional[Pose]
    """
    Where it is, for a place; none for anything else.
    """

    parameters: List[Parameter]
    """
    The values it holds, for an object that is only values, such as how an object is to
    be grasped; none for a thing of a world and for a place, whose values are not its
    own.
    """

    def to_json(self) -> Dict[str, Any]:
        """
        :return: The reference as plain JSON.
        """
        return {
            "name": self.name,
            "type": self.type,
            "identifier": self.identifier,
            "label": self.label,
            "pose": self.pose and self.pose.to_json(),
            "parameters": [parameter.to_json() for parameter in self.parameters],
        }


@dataclass(frozen=True)
class TableRow:
    """
    The row of one table a step is kept in, with its values as stored.
    """

    table: str
    """
    The table.
    """

    values: Dict[str, Any]
    """
    The value of each column, except the columns that hold bytes.
    """

    def to_json(self) -> Dict[str, Any]:
        """
        :return: The row as plain JSON.
        """
        return {"table": self.table, "values": self.values}


@dataclass(frozen=True)
class PlanNode:
    """
    A step of a plan: an action, a motion, a condition, or a sequence of steps.
    """

    identifier: int
    """
    The identifier the database gave the step.
    """

    parent: Optional[int]
    """
    The step above; none for the root of the plan.
    """

    depth: int
    """
    How many steps are above it.
    """

    type: str
    """
    What kind of step it is, such as ``ActionNode``.
    """

    designator: Optional[str]
    """
    What it was asked to do, such as ``PouringAction``; none for a step that was not
    asked for an action or a motion.
    """

    parameters: List[Parameter]
    """
    The values the step holds and was asked for.
    """

    references: List[Reference]
    """
    The things and places it was asked to act on.
    """

    rows: List[TableRow]
    """
    The rows it is kept in.
    """

    def to_json(self) -> Dict[str, Any]:
        """
        :return: The step as plain JSON.
        """
        return {
            "id": self.identifier,
            "parent": self.parent,
            "depth": self.depth,
            "type": self.type,
            "designator": self.designator,
            "parameters": [parameter.to_json() for parameter in self.parameters],
            "references": [reference.to_json() for reference in self.references],
            "rows": [row.to_json() for row in self.rows],
        }


# %% reading


@dataclass
class PlanReader:
    """
    Reads the steps of a plan from an open database.
    """

    connection: sqlite3.Connection
    """
    The connection to read from.
    """

    schema: TableSchema = field(init=False)
    """
    How the tables of the database extend one another.
    """

    def __post_init__(self) -> None:
        self.schema = TableSchema(connection=self.connection)

    def nodes(self, root: int) -> List[PlanNode]:
        """
        :param root: The step a plan begins with.
        :return: The steps of the plan, each followed by those below it.
        """
        nodes: List[PlanNode] = []
        pending: List[Tuple[int, Optional[int], int]] = [(root, None, 0)]
        while pending:
            identifier, parent, depth = pending.pop()
            nodes.append(self.node(identifier, parent, depth))
            pending.extend(
                (child, identifier, depth + 1)
                for child in reversed(self.children_of(identifier))
            )
        return nodes

    def children_of(self, identifier: int) -> List[int]:
        """
        :return: The steps directly below a step, in the order they were added.
        """
        return [
            child
            for (child,) in self.connection.execute(
                f"""
                select {Column.CHILD} from "{Table.EDGE}"
                where {Column.PARENT} = ? order by {Column.IDENTIFIER}
                """,
                (identifier,),
            )
        ]

    def node(self, identifier: int, parent: Optional[int], depth: int) -> PlanNode:
        """
        :return: One step, with what it was asked to do.
        """
        node_rows = self.rows_of(Table.ENTITY, identifier)
        designator = self.designator_of(node_rows)
        designator_rows = (
            [] if designator is None else self.rows_of(Table.DESIGNATOR, designator)
        )
        own_tables = [row.table for row in node_rows]
        own_rows = node_rows[own_tables.index(Table.NODE) + 1 :]
        return PlanNode(
            identifier=identifier,
            parent=parent,
            depth=depth,
            type=self.schema.class_name(node_rows[-1].table),
            designator=(
                self.schema.class_name(designator_rows[-1].table)
                if designator_rows
                else None
            ),
            parameters=self.parameters_of(own_rows)
            + self.parameters_of(designator_rows),
            references=self.references_of(own_rows + designator_rows),
            rows=node_rows + designator_rows,
        )

    def designator_of(self, rows: List[TableRow]) -> Optional[int]:
        """
        :return: The identifier of what a step was asked to do; none for a step that was
            not asked for an action or a motion.
        """
        identifiers = [
            row.values[column.name]
            for row in rows
            for column in self.schema.reference_columns(row.table)
            if column.references == Table.DESIGNATOR
        ]
        return next((found for found in identifiers if found is not None), None)

    def parameters_of(self, rows: List[TableRow]) -> List[Parameter]:
        """
        :return: The values the rows hold, table by table.
        """
        return [
            Parameter(name=column.name, value=row.values[column.name])
            for row in rows
            for column in self.schema.parameter_columns(row.table)
        ]

    def references_of(self, rows: List[TableRow]) -> List[Reference]:
        """
        :return: The things and places the rows refer to, leaving out those not set and
            those that tie a step to the rest of the plan.
        """
        return [
            self.describe(column, row.values[column.name])
            for row in rows
            for column in self.schema.reference_columns(row.table)
            if row.values[column.name] is not None and not self.is_bookkeeping(column)
        ]

    def is_bookkeeping(self, column: TableColumn) -> bool:
        """
        :return: Whether a column ties a step to another step, to what it was asked to do
            or to what its execution recorded, rather than to something the step acts on.
        """
        return column.references in BOOKKEEPING_TABLES or (
            Table.NODE in self.schema.chain(column.references)
        )

    def describe(self, column: TableColumn, identifier: int) -> Reference:
        """
        :param column: The column that refers to the row.
        :param identifier: The identifier of the row it refers to.
        :return: What the column refers to.
        """
        table = self.leaf_of(column.references, identifier)
        chain = self.schema.chain(table)
        is_world_thing = Table.WORLD_ENTITY in chain
        is_place = Table.SPATIAL in chain
        return Reference(
            name=column.attribute,
            type=self.schema.class_name(table),
            identifier=identifier,
            label=self.label_of(identifier) if is_world_thing else None,
            pose=self.pose_of(identifier) if Table.POSE in chain else None,
            parameters=(
                []
                if is_world_thing or is_place
                else self.parameters_of(self.rows_along(chain, identifier))
            ),
        )

    def label_of(self, identifier: Optional[int]) -> Optional[str]:
        """
        :return: What the world calls a thing; none for a thing it does not name.
        """
        found = self.connection.execute(
            f"""
            select named.{Column.NAME} from "{Table.WORLD_ENTITY}" entity
            join "{Table.NAME}" named
              on named.{Column.IDENTIFIER} = entity.{Column.NAME_REFERENCE}
            where entity.{Column.IDENTIFIER} = ?
            """,
            (identifier,),
        ).fetchone()
        return found and found[0]

    def pose_of(self, identifier: int) -> Pose:
        """
        :return: A pose, with the name of the body it is measured from.
        """
        pose = self.fetch(Table.POSE, identifier)
        spatial = self.fetch(Table.SPATIAL, identifier)
        return Pose(
            frame=self.label_of(spatial[Column.FRAME]),
            position=self.position_of(pose[Column.POSITION]),
            orientation=self.orientation_of(pose[Column.ORIENTATION]),
        )

    def position_of(self, identifier: Optional[int]) -> Optional[Position]:
        """
        :return: A position; none when the pose has none.
        """
        if identifier is None:
            return None
        row = self.fetch(Table.POINT, identifier)
        return Position(x=row[Column.X], y=row[Column.Y], z=row[Column.Z])

    def orientation_of(self, identifier: Optional[int]) -> Optional[Orientation]:
        """
        :return: An orientation; none when the pose has none.
        """
        if identifier is None:
            return None
        row = self.fetch(Table.QUATERNION, identifier)
        return Orientation(
            x=row[Column.X], y=row[Column.Y], z=row[Column.Z], w=row[Column.W]
        )

    def rows_of(self, table: str, identifier: int) -> List[TableRow]:
        """
        :param table: The table of a class, or of one it extends.
        :param identifier: The identifier of the row.
        :return: The rows the thing is spread over, from the root of its family to the
            class it really is.
        """
        return self.rows_along(
            self.schema.chain(self.leaf_of(table, identifier)), identifier
        )

    def rows_along(self, chain: List[str], identifier: int) -> List[TableRow]:
        """
        :return: The rows with the identifier in each table of a chain.
        """
        return [
            TableRow(table=found, values=self.fetch(found, identifier))
            for found in chain
        ]

    def leaf_of(self, table: str, identifier: int) -> str:
        """
        :return: The table of the class a row really is, which the root of its family
            says; the table given when the family does not say.
        """
        root = self.schema.chain(table)[0]
        if all(
            column.name != Column.POLYMORPHIC_TYPE
            for column in self.schema.columns(root)
        ):
            return table
        found = self.connection.execute(
            f'select {Column.POLYMORPHIC_TYPE} from "{root}" where {Column.IDENTIFIER} = ?',
            (identifier,),
        ).fetchone()
        if found is None or not self.schema.has_table(found[0]):
            return table
        return found[0]

    def fetch(self, table: str, identifier: int) -> Dict[str, Any]:
        """
        :return: The values of a row, by column, except those of bytes; a flag as true or
            false.
        :raises MissingRow: When the table has no such row.
        """
        columns = [
            column for column in self.schema.columns(table) if not column.is_binary
        ]
        row = self.connection.execute(
            f"""
            select {", ".join(f'"{column.name}"' for column in columns)}
            from "{table}" where {Column.IDENTIFIER} = ?
            """,
            (identifier,),
        ).fetchone()
        if row is None:
            raise MissingRow(table=table, identifier=identifier)
        return {
            column.name: (
                bool(value) if column.is_boolean and value is not None else value
            )
            for column, value in zip(columns, row)
        }
