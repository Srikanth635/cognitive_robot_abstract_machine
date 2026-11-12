"""
The live-viz bridge state: what a running demo publishes to the viewer.

This module is free of HTTP and of hook installation — it holds the :class:`Bridge`
singleton whose snapshot methods run on the *simulation* thread (see
:mod:`cram_viz.live.hooks` for why that matters) and whose ``get_*`` accessors hand
finished, plain-dict snapshots to the HTTP layer.

A plan is the statechart its executor runs: every action and every motion is a node of
it, and a node's status is its own life cycle in that statechart.
"""

from __future__ import annotations

import logging
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from coraplex.robot_plans.actions.base import Action
from cramph.data_types import LifeCycleValues
from typing_extensions import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from semantic_digital_twin.robots.robot_parts import AbstractRobot
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    Connection6DoF,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

if TYPE_CHECKING:
    from cramph.node import StatechartNode
    from cramph.statechart import Statechart
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)

#: same palette the onboarder / viewer use, so a live object keeps its colour
PALETTE = [
    "#f3f0ea",
    "#cf5b3a",
    "#b8bcc4",
    "#e7c26a",
    "#7fb069",
    "#5b8cff",
    "#c98bdb",
    "#ff9d6b",
    "#6bd0c0",
    "#d0c86b",
]


class PlanNodeKey(StrEnum):
    """
    The keys of a plan node in the bridge's snapshot of the plan.
    """

    ID = "id"
    PARENT = "parent"
    KIND = "kind"
    LABEL = "label"
    STATUS = "status"
    DERIVED = "derived"


#: cramph ``LifeCycleValues`` ordinal → its name, for the statechart view
LIFE_CYCLE_NAME = {
    0: "NOT_STARTED",
    1: "RUNNING",
    2: "PAUSED",
    3: "DONE",
    4: "FAILED",
    5: "INTERRUPTED",
}

#: cramph ``LifeCycleValues`` ordinal → the plan-node status vocabulary,
#: so the viewer styles plan nodes and statechart nodes with one status palette
LIFE_CYCLE_TO_STATUS = {
    0: "CREATED",
    1: "RUNNING",
    2: "PAUSE",
    3: "SUCCEEDED",
    4: "FAILED",
    5: "INTERRUPTED",
}

#: how long a world binding stays fresh before the bridge re-discovers bodies
REBIND_INTERVAL_SECONDS = 3.0


def _pose_as_position_quaternion(body: Body) -> List[float]:
    """
    Return a body's world pose as ``[x, y, z, qx, qy, qz, qw]``.
    """
    pose = body.global_pose
    translation = pose.position.to_np().flatten()
    quaternion = pose.quaternion.to_np().flatten()
    return [round(float(value), 5) for value in (*translation[:3], *quaternion[:4])]


@dataclass
class Bridge:
    """
    Shared state between the running demo and the viewer.

    All world reads and writes happen on the simulation thread (the tick hook); the HTTP
    handlers only ever read the finished snapshot dicts under :attr:`_lock`.
    """

    world: Optional[World] = None
    """
    The executing world, captured by the tick hook on its first call.
    """

    robot: Optional[AbstractRobot] = None
    """
    The robot annotation of :attr:`world`, re-discovered on every bind.
    """

    seq: int = 0
    """
    Monotonic snapshot counter so the viewer can skip unchanged states.
    """

    state: Dict[str, Any] = field(
        default_factory=lambda: {"seq": 0, "frames": {}, "base": None, "objects": {}}
    )
    """
    The newest world snapshot in the trajectory-frame format.
    """

    object_meta: List[Dict[str, Any]] = field(default_factory=list)
    """
    Geometry catalog for the viewer: one entry per loose object.
    """

    plan_state: Dict[str, Any] = field(default_factory=lambda: {"sig": "", "nodes": []})
    """
    The newest plan-tree snapshot (see :meth:`snapshot_plan`).
    """

    plan_observers: List[Callable[[List[Dict[str, Any]]], None]] = field(
        default_factory=list
    )
    """
    Told the nodes of every plan snapshot, on the simulation thread; they must not touch
    the world.
    """

    chart_state: Dict[str, Any] = field(
        default_factory=lambda: {"sig": "", "title": "", "nodes": [], "edges": []}
    )
    """
    The newest motion-statechart snapshot (see :meth:`observe_chart`).
    """

    _connections: List[Any] = field(default_factory=list)
    """
    Actuated world connections whose positions are published as frames.
    """

    _bodies: Dict[str, Body] = field(default_factory=dict)
    """
    Published bodies by mesh key; ``__base__`` is the robot root.
    """

    _last_bind: float = 0.0
    """
    Timestamp of the last world discovery (see ``REBIND_INTERVAL_SECONDS``).
    """

    _lock: threading.Lock = field(default_factory=threading.Lock)
    """
    Guards every snapshot dict that the HTTP layer reads.
    """

    _moves: List[Dict[str, Any]] = field(default_factory=list)
    """
    Object moves queued by the viewer, applied on the simulation thread.
    """

    _moves_lock: threading.Lock = field(default_factory=threading.Lock)
    """
    Guards :attr:`_moves` (written by HTTP threads).
    """

    _mesh_files: Dict[str, str] = field(default_factory=dict)
    """
    Mesh basename (lowercase) → absolute path, filled by the mesh hook.
    """

    _mesh_serve: Dict[str, str] = field(default_factory=dict)
    """
    Object key → absolute mesh path served via the ``/mesh`` endpoint.
    """

    _plan: Optional[Statechart] = None
    """
    The statechart of the plan the executor performs, captured by the execute hook.
    """

    _chart: Optional[Statechart] = None
    """
    The statechart the executor is currently ticking.
    """

    _chart_structure: Optional[Dict[str, Any]] = None
    """
    Serialized structure of :attr:`_chart`, rebuilt when it changes.
    """

    _ticks: int = 0
    """
    Tick counter used to throttle the plan snapshot.
    """

    _last_life_cycle: Optional[List[int]] = None
    """
    Life-cycle vector of the last published chart snapshot.
    """

    # %% viewer -> world -------------------------------------------------------
    def queue_move(self, request: Dict[str, Any]) -> None:
        """
        Queue an object move from the viewer (called on an HTTP thread).
        """
        with self._moves_lock:
            self._moves.append(request)

    def apply_moves(self) -> None:
        """
        Apply queued object moves to the world.

        Called from the tick hook — the simulation thread is the only place that may
        write to the world.
        """
        with self._moves_lock:
            moves, self._moves = self._moves, []
        if not moves or self.world is None:
            return
        for move in moves:
            body = self._bodies.get(move.get("object"))
            if body is None:
                continue
            self._apply_move(move, body)

    def _apply_move(self, move: Dict[str, Any], body: Body) -> None:
        """
        Write one viewer move into the world.

        Only free-floating (:class:`Connection6DoF`) objects are draggable.
        Objects rigidly fixed to furniture — e.g. a spoon on a drawer that
        must ride along when the drawer opens — keep their ``FixedConnection``
        and are left untouched (a fixed connection has no settable origin).

        .. note:: The object is deliberately *not* re-parented here. That is a
           structural change of the kinematic tree (``modify_world`` + forward
           kinematics recompile) and running it inside the tick hook while a
           giskard goal is live hangs the executor. Re-parenting must happen
           as its own plan step between motions; the plain pose write already
           makes ``body.global_pose`` correct, which is all the plan's
           navigate/pick reachability needs.
        """
        connection = body.parent_connection
        if not isinstance(connection, Connection6DoF):
            logger.info(
                "%s is fixed (%s) — not draggable, skipping",
                move["object"],
                type(connection).__name__,
            )
            return
        position = move["pos"]
        quaternion = move.get("quat")
        if not quaternion:
            current = _pose_as_position_quaternion(body)
            quaternion = current[3:7]
        world_T_object = HomogeneousTransformationMatrix.from_xyz_quaternion(
            position[0],
            position[1],
            position[2],
            quaternion[0],
            quaternion[1],
            quaternion[2],
            quaternion[3],
            reference_frame=self.world.root,
        )
        # ``origin`` is parent-relative; express the target in the parent
        # frame (a no-op while the parent is the world root)
        parent_T_object = connection.parent.global_pose.inverse() @ world_T_object
        connection.origin = parent_T_object
        logger.info(
            "moved %s -> world (%.3f, %.3f, %.3f) [final=%s]",
            move["object"],
            position[0],
            position[1],
            position[2],
            bool(move.get("final")),
        )

    # %% world discovery -------------------------------------------------------
    def _bind(self) -> None:
        """
        Discover the robot, joints and loose objects of the current world.

        Re-run periodically because demos modify their world (objects get spawned and
        removed mid-run).
        """
        world = self.world
        if world is None:
            return
        self._last_bind = time.time()
        robots = world.get_semantic_annotations_by_type(AbstractRobot)
        self.robot = robots[0] if robots else None
        self._connections = self._actuated_connections(world)
        bodies: Dict[str, Body] = {}
        if self.robot is not None:
            bodies["__base__"] = self.robot.root
        try:
            # loose objects by convention: bodies named like mesh files
            for body in world.bodies:
                basename = str(body.name).split("/")[-1]
                if basename.lower().endswith((".stl", ".obj", ".dae")):
                    bodies[basename] = body
        except Exception as error:
            # the world is mid-modification (a body is being spawned or
            # removed): keep the previous catalog instead of publishing an
            # empty one — the viewer would otherwise hide every object it
            # already shows
            logger.debug("body scan skipped this bind: %s", error)
            for key, body in self._bodies.items():
                bodies.setdefault(key, body)
        self._bodies = bodies
        self._build_object_meta(bodies)

    @staticmethod
    def _actuated_connections(world: World) -> List[ActiveConnection1DOF]:
        """
        All 1-DOF connections — the joints published as trajectory frames.
        """
        return [
            connection
            for connection in world.connections or []
            if isinstance(connection, ActiveConnection1DOF)
        ]

    def _build_object_meta(self, bodies: Dict[str, Body]) -> None:
        """
        Rebuild the geometry catalog the viewer spawns live objects from.

        Each object gets either a mesh URL (served by the bridge) or a box size, so
        objects the viewer does not know yet can appear mid-run.
        """
        catalog: List[Dict[str, Any]] = []
        serve: Dict[str, str] = {}
        for index, (key, body) in enumerate(
            item for item in bodies.items() if item[0] != "__base__"
        ):
            color = PALETTE[index % len(PALETTE)]
            object_id = Path(key).stem
            mesh_path = self._mesh_files.get(key.lower())
            if mesh_path and Path(mesh_path).is_file():
                serve[key] = mesh_path
                catalog.append(
                    {
                        "key": key,
                        "id": object_id,
                        "kind": "mesh",
                        "mesh": "/mesh?key=" + urllib.parse.quote(key),
                        "format": Path(key).suffix.lstrip(".").lower(),
                        "color": color,
                    }
                )
            else:
                catalog.append(
                    {
                        "key": key,
                        "id": object_id,
                        "kind": "box",
                        "size": self._box_size(body) or [0.06, 0.06, 0.12],
                        "color": color,
                    }
                )
        self._mesh_serve = serve
        with self._lock:
            self.object_meta = catalog

    @staticmethod
    def _box_size(body: Body) -> Optional[List[float]]:
        """
        Best-effort bounding-box size of a body's visual shape in metres.
        """
        for attribute in ("visual", "collision"):
            shape_container = getattr(body, attribute, None)
            shapes = getattr(shape_container, "shapes", None)
            if shapes is None and isinstance(shape_container, (list, tuple)):
                shapes = shape_container
            for shape in shapes or []:
                scale = getattr(shape, "scale", None)
                if scale is not None:
                    return [
                        round(float(scale.x), 4),
                        round(float(scale.y), 4),
                        round(float(scale.z), 4),
                    ]
        return None

    # %% world snapshot ---------------------------------------------------------
    def snapshot(self) -> None:
        """
        Publish the world's joints, base pose and object poses.

        Runs on the simulation thread; rebinds the world periodically so mid-run spawns
        show up.
        """
        if self.world is None:
            return
        if time.time() - self._last_bind > REBIND_INTERVAL_SECONDS:
            self._bind()
        frames = {
            str(connection.name): round(float(connection.position), 5)
            for connection in self._connections
        }
        base_pose: Optional[List[float]] = None
        object_poses: Dict[str, List[float]] = {}
        for name, body in self._bodies.items():
            if name == "__base__":
                base_pose = _pose_as_position_quaternion(body)
            else:
                object_poses[name] = _pose_as_position_quaternion(body)
        with self._lock:
            self.seq += 1
            self.state = {
                "seq": self.seq,
                "frames": frames,
                "base": base_pose,
                "objects": object_poses,
            }

    def get_state(self) -> Dict[str, Any]:
        """
        The newest world snapshot (safe to call from HTTP threads).
        """
        with self._lock:
            return dict(self.state)

    # %% plan tree ----------------------------------------------------------------
    def snapshot_plan(self) -> None:
        """
        Serialize the plan's nodes, each with the status its life cycle gives it.
        """
        statechart = self._plan
        if statechart is None:
            return
        nodes: List[Dict[str, Any]] = []
        order: List[str] = []
        for node in statechart.top_level_nodes:
            self._serialize_plan_node(node, None, nodes, order)
        with self._lock:
            self.plan_state = {"sig": "|".join(order), "nodes": nodes}
        for observer in self.plan_observers:
            observer(nodes)

    def _serialize_plan_node(
        self,
        node: StatechartNode,
        parent_id: Optional[str],
        nodes: List[Dict[str, Any]],
        order: List[str],
    ) -> None:
        """
        Serialize one plan node and the nodes below it.
        """
        node_id = "p%d" % id(node)
        entry: Dict[str, Any] = {
            PlanNodeKey.ID: node_id,
            PlanNodeKey.PARENT: parent_id,
            PlanNodeKey.KIND: type(node).__name__,
            PlanNodeKey.LABEL: type(node).__name__,
            PlanNodeKey.STATUS: LIFE_CYCLE_TO_STATUS[int(node.life_cycle_state)],
            PlanNodeKey.DERIVED: False,
        }
        self._add_designator_metadata(entry=entry, node=node)
        nodes.append(entry)
        order.append(node_id)
        for child in node.children:
            self._serialize_plan_node(child, node_id, nodes, order)

    def _add_designator_metadata(self, *, entry: Dict[str, Any], node: Any) -> None:
        """
        Add the arms a plan node moves, by name, and the published object it acts on, if
        any.
        """
        node_fields = vars(node)
        arms = node_fields.get("arm") or node_fields.get("arms")
        if arms is not None:
            entry["arm"] = ", ".join(
                arm.name.name if isinstance(arm, SemanticAnnotation) else str(arm)
                for arm in (arms if isinstance(arms, list) else [arms])
            )
        target = self._designator_target(node)
        if target:
            entry["target"] = target

    def _designator_target(self, designator: Any) -> Optional[str]:
        """
        Name of the published object a designator refers to, if any.
        """
        known = set(self._bodies)
        for value in vars(designator).values():
            name = getattr(value, "name", None)
            if name is None:
                continue
            basename = str(name).split("/")[-1]
            if basename in known:
                return basename
        return None

    def get_plan(self) -> Dict[str, Any]:
        """
        The newest plan snapshot (safe to call from HTTP threads).
        """
        with self._lock:
            return dict(self.plan_state)

    # %% motion statechart -----------------------------------------------------------
    def observe_chart(self, chart: Optional[Statechart]) -> None:
        """
        Publish the executing statechart's structure and node states.

        Called from the tick hook. The structure is re-serialized only when the executor
        compiled a new chart; the life-cycle / observation vectors are cheap and
        refreshed whenever they change.
        """
        if chart is None:
            return
        if chart is not self._chart or self._chart_structure is None:
            self._chart = chart
            self._chart_structure = self._serialize_chart_structure(chart)
            self._last_life_cycle = None
        structure = self._chart_structure
        if not structure:
            return
        life_cycle = [
            int(chart.life_cycle_state.data[index]) for index in structure["indices"]
        ]
        observations = [
            float(chart.observation_state.data[index]) for index in structure["indices"]
        ]
        if life_cycle == self._last_life_cycle:
            return
        self._last_life_cycle = life_cycle
        nodes = []
        for position, node in enumerate(structure["nodes"]):
            entry = dict(node)
            entry["life"] = LIFE_CYCLE_NAME.get(
                life_cycle[position], str(life_cycle[position])
            )
            entry["obs"] = self._observation_name(observations[position])
            nodes.append(entry)
        with self._lock:
            self.chart_state = {
                "sig": structure["sig"],
                "title": self._running_action_name(chart),
                "nodes": nodes,
                "edges": structure["edges"],
            }

    @staticmethod
    def _observation_name(observation: float) -> str:
        """
        Trinary observation value → name (0 false, 0.5 unknown, 1 true).
        """
        if observation >= 0.75:
            return "TRUE"
        if observation <= 0.25:
            return "FALSE"
        return "UNKNOWN"

    @staticmethod
    def _running_action_name(chart: Statechart) -> str:
        """
        :return: The name of the innermost running action's class, or ``''``.
        """
        running = [
            node
            for node in chart.nodes
            if isinstance(node, Action)
            and node.life_cycle_state == LifeCycleValues.RUNNING
        ]
        return type(running[-1]).__name__ if running else ""

    @staticmethod
    def _serialize_chart_structure(chart: Statechart) -> Optional[Dict[str, Any]]:
        """
        Nodes and transition edges of a statechart, as plain dicts.
        """
        nodes: List[Dict[str, Any]] = []
        indices: List[int] = []
        for node in chart.nodes:
            parent_index = (
                node.parent_node.index if node.parent_node is not None else None
            )
            nodes.append(
                {
                    "id": "s%d" % node.index,
                    "name": node.name,
                    "cls": type(node).__name__,
                    "parent": (
                        ("s%d" % parent_index) if parent_index is not None else None
                    ),
                }
            )
            indices.append(node.index)
        edges = []
        for source, target, transition in chart.rx_graph.edge_index_map().values():
            edges.append(
                {
                    "from": "s%d" % chart.rx_graph.get_node_data(source).index,
                    "to": "s%d" % chart.rx_graph.get_node_data(target).index,
                    "kind": transition.kind.name,
                }
            )
        signature = "|".join(node["id"] + ":" + node["name"] for node in nodes)
        return {"nodes": nodes, "edges": edges, "indices": indices, "sig": signature}

    def get_chart(self) -> Dict[str, Any]:
        """
        The newest statechart snapshot (safe to call from HTTP threads).
        """
        with self._lock:
            return dict(self.chart_state)


#: the process-wide bridge instance shared by hooks and HTTP handlers
BRIDGE = Bridge()
