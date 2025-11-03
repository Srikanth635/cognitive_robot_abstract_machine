from __future__ import annotations
import rustworkx as rx
from typing import Tuple

from typing_extensions import TYPE_CHECKING
from .connections import ActiveConnection


if TYPE_CHECKING:
    from ..world import World
    from .world_entity import Body, Connection


class BodyCollector(rx.visit.DFSVisitor):
    def __init__(self, world: World, collision_bodies_only: bool = False):
        self.world = world
        self.bodies = []
        self.collision_bodies_only = collision_bodies_only

    def discover_vertex(self, node_index: int, time: int) -> None:
        body = self.world.kinematic_structure[node_index]
        if not self.collision_bodies_only or (
            self.collision_bodies_only
            and isinstance(body, Body)
            and body.has_collision()
        ):
            self.bodies.append(body)

    def tree_edge(self, args: Tuple[int, int, Connection]) -> None:
        parent_index, child_index, e = args
        if (
            isinstance(e, ActiveConnection)
            and e.has_hardware_interface
            and not e.frozen_for_collision_avoidance
        ):
            raise rx.visit.PruneSearch()

class ConnectionCollector(rx.visit.DFSVisitor):
    def __init__(self, world: World):
        self.world = world
        self.connections = []

    def tree_edge(self, edge: Tuple[int, int, Connection]):
        """Called for each tree edge during DFS traversal"""
        self.connections.append(edge[2])  # edge[2] is the connection
