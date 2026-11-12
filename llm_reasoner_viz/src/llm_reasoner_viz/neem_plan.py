"""
The steps of a plan as the NEEM database keeps them: the statechart the plan was
performed as, the tree its nodes form, how each ended, what each was asked to do, and the
things and places it refers to.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from functools import cached_property

from typing_extensions import Any, Callable, Dict, List, Optional, Tuple

from llm_reasoner_viz.neem_schema import (
    DAO_SUFFIX,
    Column,
    JsonKey,
    LifeCycle,
    ModulePrefix,
    StoredClass,
    Table,
    WorldTable,
)

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

    identifier: Optional[int]
    """
    The identifier the row should have; none when the row referring to it names none.
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

    identifier: Optional[str]
    """
    The identifier of a thing of a world; none for anything else.
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


@dataclass(kw_only=True)
class StepValues:
    """
    The values a step was asked for and the things and places it refers to, in the order
    of its fields.
    """

    parameters: List[Parameter] = field(default_factory=list)
    """
    The values.
    """

    references: List[Reference] = field(default_factory=list)
    """
    The things and places.
    """


@dataclass(frozen=True)
class PlanNode:
    """
    A step of a plan: an action, a motion, a monitor, or a node holding steps.
    """

    identifier: int
    """
    The index of the step in its statechart.
    """

    parent: Optional[int]
    """
    The step above; none for a step at the top of the statechart.
    """

    depth: int
    """
    How many steps are above it.
    """

    type: str
    """
    The class of the step, such as ``PouringAction`` or ``Sequence``.
    """

    designator: Optional[str]
    """
    The action it was asked to perform, such as ``PouringAction``; none for a step that
    is no action.
    """

    status: str
    """
    How the step ended, such as ``SUCCEEDED``, or where it was when the NEEM was kept.
    """

    started: Optional[float]
    """
    When the step began, in seconds after the plan did; none for a step never begun.
    """

    ended: Optional[float]
    """
    When the step ended, in seconds after the plan began; none for a step that has not.
    """

    parameters: List[Parameter]
    """
    The values the step holds and was asked for.
    """

    references: List[Reference]
    """
    The things and places it was asked to act on.
    """

    stored: Dict[str, Any]
    """
    The fields of the step as the statechart keeps them, without the steps below it.
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
            "status": self.status,
            "started": self.started,
            "ended": self.ended,
            "parameters": [parameter.to_json() for parameter in self.parameters],
            "references": [reference.to_json() for reference in self.references],
            "stored": self.stored,
        }


# %% the statechart kept


@dataclass(frozen=True)
class TickStates:
    """
    The life cycle of every node of a statechart at one tick.
    """

    tick: int
    """
    The tick, counted from the start of the plan.
    """

    life_cycles: List[LifeCycle]
    """
    The life cycle of each node, by its index.
    """


@dataclass(kw_only=True)
class StoredStatechart:
    """
    A statechart as a NEEM keeps it: its nodes and how their life cycles changed.
    """

    structure: Dict[str, Any]
    """
    The statechart as the JSON its classes write of it.
    """

    ticks: List[TickStates]
    """
    The life cycles on every tick on which one of them changed, in order.
    """

    tick_duration: float
    """
    How many seconds a tick lasted.
    """

    @property
    def nodes(self) -> List[Dict[str, Any]]:
        """
        The nodes, by their indices.
        """
        return self.structure[JsonKey.NODES]

    @cached_property
    def children(self) -> Dict[int, List[int]]:
        """
        The indices of the nodes directly below each node, in their order, by its index.
        """
        return {
            int(index): below
            for index, below in self.structure[JsonKey.CHILDREN].items()
        }

    @property
    def top_level(self) -> List[int]:
        """
        The indices of the nodes no node is above: the plan first, then what its executor
        added to watch it.
        """
        return [
            index
            for index, node in enumerate(self.nodes)
            if node[JsonKey.PARENT] is None
        ]

    @property
    def plan_root(self) -> int:
        """
        The index of the node the plan was performed from, which comes before the nodes
        its executor adds.
        """
        return self.top_level[0]

    def tree(self) -> List[Tuple[int, Optional[int], int]]:
        """
        :return: Every node with the one above it and how deep it is, each followed by
            those below it, the plan first.
        """
        order: List[Tuple[int, Optional[int], int]] = []
        pending = [(index, None, 0) for index in reversed(self.top_level)]
        while pending:
            index, parent, depth = pending.pop()
            order.append((index, parent, depth))
            pending.extend(
                (child, index, depth + 1)
                for child in reversed(self.children.get(index, []))
            )
        return order

    def class_path(self, *, index: int) -> str:
        """
        :return: The module and the name of the class of a node.
        """
        return self.nodes[index][JsonKey.TYPE]

    def is_action(self, *, index: int) -> bool:
        """
        :return: Whether a node is an action.
        """
        return self.class_path(index=index).startswith(ModulePrefix.ACTION)

    def final_state(self, *, index: int) -> LifeCycle:
        """
        :return: Where a node was in its life cycle when the NEEM was kept.
        """
        if not self.ticks:
            return LifeCycle.NOT_STARTED
        return self.ticks[-1].life_cycles[index]

    def started(self, *, index: int) -> Optional[float]:
        """
        :return: When a node began, in seconds after the plan did; none if it never did.
        """
        return self.first_second(
            index=index, reached=lambda state: state != LifeCycle.NOT_STARTED
        )

    def ended(self, *, index: int) -> Optional[float]:
        """
        :return: When a node ended, in seconds after the plan began; none if it has not.
        """
        return self.first_second(index=index, reached=lambda state: state.is_terminal)

    def first_second(
        self, *, index: int, reached: Callable[[LifeCycle], bool]
    ) -> Optional[float]:
        """
        :param index: The node.
        :param reached: Whether a life cycle state is the one looked for.
        :return: The second of the first tick the node was in such a state on; none if
            it never was.
        """
        return next(
            (
                states.tick * self.tick_duration
                for states in self.ticks
                if reached(states.life_cycles[index])
            ),
            None,
        )


# %% the values of a step


@dataclass
class ValueReader:
    """
    Reads the fields of a stored node as values, and as the things and places it refers
    to.
    """

    connection: sqlite3.Connection
    """
    The connection to look up the kinds of the things of a world in.
    """

    def step_values(self, *, node: Dict[str, Any]) -> StepValues:
        """
        :return: What a node was asked for, field by field, leaving out the fields that
            place it in the statechart and the steps below it.
        """
        values = StepValues()
        for name, value in node.items():
            if name in JsonKey and JsonKey(name).places_a_node:
                continue
            self.read(name=name, value=value, values=values)
        return values

    def read(self, *, name: str, value: Any, values: StepValues) -> None:
        """
        Add a field's value to the values of a step.

        :param name: The name of the field.
        :param value: The value as stored.
        :param values: The values of the step so far.
        """
        if is_node(value):
            return
        if is_collection(value):
            for item in value[JsonKey.ITEMS]:
                self.read(name=name, value=item, values=values)
            return
        if is_value(value):
            values.parameters.append(Parameter(name=name, value=value_of(value)))
            return
        values.references.append(self.reference(name=name, value=value))

    def reference(self, *, name: str, value: Dict[str, Any]) -> Reference:
        """
        :return: The thing, the place or the object of values a field refers to.
        """
        class_path = value[JsonKey.TYPE]
        if class_path == StoredClass.WORLD_THING:
            identifier = value[JsonKey.ID][JsonKey.VALUE]
            return Reference(
                name=name,
                type=self.kind_of_world_thing(identifier=identifier)
                or class_name(class_path),
                identifier=identifier,
                label=value[JsonKey.NAME][JsonKey.NAME],
                pose=None,
                parameters=[],
            )
        if class_path == StoredClass.POSE:
            return Reference(
                name=name,
                type=class_name(class_path),
                identifier=None,
                label=None,
                pose=pose_of(value),
                parameters=[],
            )
        return Reference(
            name=name,
            type=class_name(class_path),
            identifier=None,
            label=None,
            pose=None,
            parameters=[
                Parameter(name=key, value=value_of(held))
                for key, held in value.items()
                if key != JsonKey.TYPE and is_value(held)
            ],
        )

    def kind_of_world_thing(self, *, identifier: str) -> Optional[str]:
        """
        :return: The class of the thing of a world the identifier names; none when the
            database does not keep it.
        """
        if not self.keeps_world_things:
            return None
        found = self.connection.execute(
            f"""
            select entity.{Column.POLYMORPHIC_TYPE}
            from "{WorldTable.ENTITY_WITH_ID}" known
            join "{WorldTable.ENTITY}" entity
              on entity.{Column.IDENTIFIER} = known.{Column.IDENTIFIER}
            where known.{Column.ENTITY_ID} = ?
            limit 1
            """,
            (identifier.replace("-", ""),),
        ).fetchone()
        return found and found[0].removesuffix(DAO_SUFFIX)

    @cached_property
    def keeps_world_things(self) -> bool:
        """
        Whether the database has the tables of the things of a world.
        """
        present = {
            name
            for (name,) in self.connection.execute(
                "select name from sqlite_master where type = 'table'"
            )
        }
        return all(table in present for table in WorldTable)


def is_node(value: Any) -> bool:
    """
    :return: Whether a stored value is a node of the statechart, or a list of them.
    """
    if is_collection(value):
        return any(is_node(item) for item in value[JsonKey.ITEMS])
    return isinstance(value, dict) and JsonKey.NODE_ID in value


def is_collection(value: Any) -> bool:
    """
    :return: Whether a stored value is a list of values.
    """
    return isinstance(value, dict) and JsonKey.COLLECTION in value


def is_member(value: Any) -> bool:
    """
    :return: Whether a stored value is a choice from an enumeration.
    """
    return isinstance(value, dict) and set(value) == {JsonKey.TYPE, JsonKey.NAME}


def is_identifier(value: Any) -> bool:
    """
    :return: Whether a stored value is an identifier.
    """
    return isinstance(value, dict) and value.get(JsonKey.TYPE) == StoredClass.IDENTIFIER


def is_value(value: Any) -> bool:
    """
    :return: Whether a stored value is a value rather than an object holding values: a
        number, text, a flag, none, a choice from an enumeration or an identifier.
    """
    return not isinstance(value, dict) or is_member(value) or is_identifier(value)


def value_of(value: Any) -> Any:
    """
    :param value: A stored value that :func:`is_value`.
    :return: The value: itself for a number, text, a flag or none; the path of its
        member for a choice from an enumeration; the text of an identifier.
    """
    if is_member(value):
        return f"{value[JsonKey.TYPE]}.{value[JsonKey.NAME]}"
    if is_identifier(value):
        return value[JsonKey.VALUE]
    return value


def class_name(class_path: str) -> str:
    """
    :return: The name of a class, without its module.
    """
    return class_path.rsplit(".", 1)[-1]


def pose_of(value: Dict[str, Any]) -> Pose:
    """
    :return: A stored pose, with the name of the body it is measured from.
    """
    x, y, z, _ = value[JsonKey.POSITION]
    qx, qy, qz, qw = value[JsonKey.ROTATION]
    frame = value.get(JsonKey.FRAME_NAME)
    return Pose(
        frame=frame and frame[JsonKey.NAME],
        position=Position(x=x, y=y, z=z),
        orientation=Orientation(x=qx, y=qy, z=qz, w=qw),
    )


# %% reading


@dataclass
class PlanReader:
    """
    Reads the statecharts of the plans from an open database.
    """

    connection: sqlite3.Connection
    """
    The connection to read from.
    """

    def statechart(self, *, neem: int) -> Optional[StoredStatechart]:
        """
        :return: The statechart of a NEEM; none when the database holds no such NEEM.
        :raises MissingRow: When what the statechart was performed in is not there.
        """
        found = self.connection.execute(
            f"""
            select {Column.STRUCTURE}, {Column.CONTEXT} from "{Table.STATECHART}"
            where {Column.IDENTIFIER} = ?
            """,
            (neem,),
        ).fetchone()
        if found is None:
            return None
        structure, context = found
        duration = self.connection.execute(
            f"""
            select {Column.TICK_DURATION} from "{Table.CONTEXT}"
            where {Column.IDENTIFIER} = ?
            """,
            (context,),
        ).fetchone()
        if duration is None:
            raise MissingRow(table=Table.CONTEXT, identifier=context)
        return StoredStatechart(
            structure=json.loads(structure),
            ticks=self.ticks(neem=neem),
            tick_duration=duration[0],
        )

    def neems(self) -> List[int]:
        """
        :return: The identifiers of every NEEM, in order.
        """
        return [
            neem
            for (neem,) in self.connection.execute(
                f'select {Column.IDENTIFIER} from "{Table.STATECHART}" order by 1'
            )
        ]

    def ticks(self, *, neem: int) -> List[TickStates]:
        """
        :return: The life cycles of a NEEM's nodes on every tick kept of it, in order.
        """
        link, statechart_column, tick_column = self.tick_link
        return [
            TickStates(
                tick=tick,
                life_cycles=[LifeCycle(int(state)) for state in json.loads(states)],
            )
            for tick, states in self.connection.execute(
                f"""
                select tick.{Column.TICK_COUNT}, tick.{Column.LIFE_CYCLES}
                from "{Table.TICK}" tick
                join "{link}" link on link.{tick_column} = tick.{Column.IDENTIFIER}
                where link.{statechart_column} = ?
                order by tick.{Column.IDENTIFIER}
                """,
                (neem,),
            )
        ]

    @cached_property
    def tick_link(self) -> Tuple[str, str, str]:
        """
        The table the ORM ties a statechart to its ticks with, which it names by a hash,
        with its column of the statechart and its column of the tick.

        :raises MissingRow: When the database has no such table.
        """
        for (table,) in self.connection.execute(
            "select name from sqlite_master where type = 'table'"
        ):
            columns = {
                referenced: column
                for _, _, referenced, column, *_ in self.connection.execute(
                    f'pragma foreign_key_list("{table}")'
                )
            }
            if Table.STATECHART in columns and Table.TICK in columns:
                return table, columns[Table.STATECHART], columns[Table.TICK]
        raise MissingRow(table=f"{Table.STATECHART} to {Table.TICK}", identifier=None)

    def nodes(self, *, statechart: StoredStatechart) -> List[PlanNode]:
        """
        :return: The steps of the plan, each followed by those below it, the plan first
            and what its executor added to watch it after.
        """
        values = ValueReader(connection=self.connection)
        nodes = []
        for index, parent, depth in statechart.tree():
            stored = statechart.nodes[index]
            step = values.step_values(node=stored)
            nodes.append(
                PlanNode(
                    identifier=index,
                    parent=parent,
                    depth=depth,
                    type=class_name(statechart.class_path(index=index)),
                    designator=(
                        class_name(statechart.class_path(index=index))
                        if statechart.is_action(index=index)
                        else None
                    ),
                    status=statechart.final_state(index=index).name,
                    started=statechart.started(index=index),
                    ended=statechart.ended(index=index),
                    parameters=step.parameters,
                    references=step.references,
                    stored={
                        key: held
                        for key, held in stored.items()
                        if key != JsonKey.PLOT and not is_node(held)
                    },
                )
            )
        return nodes
