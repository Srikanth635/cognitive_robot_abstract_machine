from __future__ import absolute_import
from __future__ import annotations

import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum
from functools import wraps, lru_cache
from itertools import combinations_with_replacement
from typing import Dict, Tuple, OrderedDict, Union, Optional, Type, Set
from typing import TypeVar
from typing import Dict, Tuple, OrderedDict, Union, Optional, Generic, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import rustworkx as rx
import rustworkx.visit
import rustworkx.visualization
from lxml import etree
from typing_extensions import List

from .connections import ActiveConnection, PassiveConnection, FixedConnection
from .connections import HasUpdateState, Has1DOFState, Connection6DoF
from .degree_of_freedom import DegreeOfFreedom
from .exceptions import DuplicateViewError, AddingAnExistingViewError
from .exceptions import ViewNotFoundError
from .ik_solver import InverseKinematicsSolver
from .prefixed_name import PrefixedName
from .robots import AbstractRobot
from .spatial_types import spatial_types as cas
from .spatial_types.derivatives import Derivatives, DerivativeMap
from .spatial_types.math import inverse_frame
from .types import NpMatrix4x4
from .utils import IDGenerator, copy_lru_cache
from .world_entity import Body, Connection, View, CollisionCheckingConfig
from .world_state import WorldState

logger = logging.getLogger(__name__)

id_generator = IDGenerator()

ConnectionTypeVar = TypeVar('ConnectionTypeVar', bound=Connection)

T = TypeVar("T")


class PlotAlignment(IntEnum):
    HORIZONTAL = 0
    VERTICAL = 1


class ForwardKinematicsVisitor(rustworkx.visit.DFSVisitor):
    """
    Visitor class for collection various forward kinematics expressions in a world model.

    This class is designed to traverse a world, compute the forward kinematics transformations in batches for different
    use cases.
    1. Efficient computation of forward kinematics between any bodies in the world.
    2. Efficient computation of forward kinematics for all bodies with collisions for updating collision checkers.
    3. Efficient computation of forward kinematics as position and quaternion, useful for ROS tf.
    """

    compiled_collision_fks: cas.CompiledFunction
    compiled_all_fks: cas.CompiledFunction

    forward_kinematics_for_all_bodies: np.ndarray
    """
    A 2D array containing the stacked forward kinematics expressions for all bodies in the world.
    Dimensions are ((number of bodies) * 4) x 4.
    They are computed in batch for efficiency.
    """
    body_name_to_forward_kinematics_idx: Dict[PrefixedName, int]
    """
    Given a body name, returns the index of the first row in `forward_kinematics_for_all_bodies` that corresponds to that body.
    """

    def __init__(self, world: World):
        self.world = world
        self.child_body_to_fk_expr: Dict[PrefixedName, cas.TransformationMatrix] = {
            self.world.root.name: cas.TransformationMatrix()}
        self.tf: Dict[Tuple[PrefixedName, PrefixedName], cas.Expression] = OrderedDict()

    def connection_call(self, edge: Tuple[int, int, Connection]):
        """
        Gathers forward kinematics expressions for a connection.
        """
        connection = edge[2]
        map_T_parent = self.child_body_to_fk_expr[connection.parent.name]
        self.child_body_to_fk_expr[connection.child.name] = map_T_parent.dot(connection.origin_expression)
        self.tf[(connection.parent.name, connection.child.name)] = connection.origin_as_position_quaternion()

    tree_edge = connection_call

    def compile_forward_kinematics(self) -> None:
        """
        Compiles forward kinematics expressions for fast evaluation.
        """
        all_fks = cas.vstack([self.child_body_to_fk_expr[body.name] for body in self.world.bodies])
        tf = cas.vstack([pose for pose in self.tf.values()])
        collision_fks = []
        for body in sorted(self.world.bodies_with_enabled_collision, key=lambda b: b.name):
            if body == self.world.root:
                continue
            collision_fks.append(self.child_body_to_fk_expr[body.name])
        collision_fks = cas.vstack(collision_fks)
        params = [v.symbols.position for v in self.world.degrees_of_freedom]
        self.compiled_all_fks = all_fks.compile(parameters=params)
        self.compiled_collision_fks = collision_fks.compile(parameters=params)
        self.compiled_tf = tf.compile(parameters=params)
        self.idx_start = {body.name: i * 4 for i, body in enumerate(self.world.bodies)}

    def recompute(self) -> None:
        """
        Clears cache and recomputes all forward kinematics. Should be called after a state update.
        """
        self.compute_forward_kinematics_np.cache_clear()
        self.subs = self.world.state.positions
        self.forward_kinematics_for_all_bodies = self.compiled_all_fks.fast_call(self.subs)
        self.collision_fks = self.compiled_collision_fks.fast_call(self.subs)

    def compute_tf(self) -> np.ndarray:
        """
        Computes a (number of bodies) x 7 matrix of forward kinematics in position/quaternion format.
        The rows are ordered by body name.
        The first 3 entries are position values, the last 4 entires are quaternion values in x, y, z, w order.

        This is not updated in 'recompute', because this functionality is only used with ROS.
        :return: A large matrix with all forward kinematics.
        """
        return self.compiled_tf.fast_call(self.subs)

    @lru_cache(maxsize=None)
    def compute_forward_kinematics_np(self, root: Body, tip: Body) -> NpMatrix4x4:
        """
        Computes the forward kinematics from the root body to the tip body, root_T_tip.

        This method computes the transformation matrix representing the pose of the
        tip body relative to the root body, expressed as a numpy ndarray.

        :param root: Root body for which the kinematics are computed.
        :param tip: Tip body to which the kinematics are computed.
        :return: Transformation matrix representing the relative pose of the tip body with respect to the root body.
        """
        root = root.name
        tip = tip.name
        root_is_world = root == self.world.root.name
        tip_is_world = tip == self.world.root.name

        if not tip_is_world:
            i = self.idx_start[tip]
            map_T_tip = self.forward_kinematics_for_all_bodies[i:i + 4]
            if root_is_world:
                return map_T_tip

        if not root_is_world:
            i = self.idx_start[root]
            map_T_root = self.forward_kinematics_for_all_bodies[i:i + 4]
            root_T_map = inverse_frame(map_T_root)
            if tip_is_world:
                return root_T_map

        if tip_is_world and root_is_world:
            return np.eye(4)

        return root_T_map @ map_T_tip


class ResetStateContextManager:
    """
    A context manager for resetting the state of a given `World` instance.

    This class is designed to allow operations to be performed on a `World`
    object, ensuring that its state can be safely returned to its previous
    condition upon leaving the context. If no exceptions occur within the
    context, the original state of the `World` instance is restored, and the
    state change is notified.
    """

    def __init__(self, world: World):
        self.world = world

    def __enter__(self) -> None:
        self.state = deepcopy(self.world.state)

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[type]) -> None:
        if exc_type is None:
            self.world.state = self.state
            self.world.notify_state_change()


class WorldModelUpdateContextManager:
    """
    Context manager for updating the state of a given `World` instance.
    This class manages that updates to the world within the context of this class only trigger recomputations after all
    desired updates have been performed.
    """
    first: bool = True

    def __init__(self, world: World):
        self.world = world

    def __enter__(self):
        if self.world.world_is_being_modified:
            self.first = False
        self.world.world_is_being_modified = True
        return self.world

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.first:
            self.world.world_is_being_modified = False
            if exc_type is None:
                self.world._notify_model_change()


def modifies_world(func):
    """
    Decorator that marks a method as a modification to the state or model of a world.
    """

    @wraps(func)
    def wrapper(self: World, *args, **kwargs):
        with self.modify_world() as context_manager:
            result = func(self, *args, **kwargs)
            return result

    return wrapper


@dataclass
class World:
    """
    A class representing the world.
    The world manages a set of bodies and connections represented as a tree-like graph.
    The nodes represent bodies in the world, and the edges represent joins between them.
    """

    kinematic_structure: rx.PyDAG[Body] = field(default_factory=lambda: rx.PyDAG(multigraph=False), kw_only=True,
                                                repr=False)
    """
    The kinematic structure of the world.
    The kinematic structure is a tree shaped directed graph where the nodes represent bodies in the world,
    and the edges represent connections between them.
    """

    views: List[View] = field(default_factory=list, repr=False)
    """
    All views the world is aware of.
    """

    degrees_of_freedom: List[DegreeOfFreedom] = field(default_factory=list)

    state: WorldState = field(default_factory=WorldState)
    """
    2d array where rows are derivatives and columns are dof values for that derivative.
    """

    _model_version: int = 0
    """
    The version of the model. This increases whenever a change to the kinematic model is made. Mostly triggered
    by adding/removing bodies and connections.
    """

    _state_version: int = 0
    """
    The version of the state. This increases whenever a change to the state of the kinematic model is made. 
    Mostly triggered by updating connection values.
    """

    world_is_being_modified: bool = False
    """
    Is set to True, when a function with @modifies_world is called or world.modify_world context is used.
    """

    name: Optional[str] = None
    """
    Name of the world. May act as default namespace for all bodies and views in the world which do not have a prefix.
    """

    _disabled_collision_pairs: Set[Tuple[Body, Body]] = field(default_factory=lambda: set())
    """
    Collisions for these Body pairs is disabled.
    """

    _temp_disabled_collision_pairs: Set[Tuple[Body, Body]] = field(default_factory=lambda: set())
    """
    A set of Body pairs for which collisions are temporarily disabled.
    """

    def reset_temporary_collision_config(self):
        self._temp_disabled_collision_pairs = set()
        for body in self.bodies_with_collisions:
            body.reset_temporary_collision_config()

    @property
    def root(self) -> Body:
        """
        The root of the world is the unique node with in-degree 0.

        :return: The root of the world.
        """
        possible_roots = [node for node in self.bodies if self.kinematic_structure.in_degree(node.index) == 0]
        if len(possible_roots) == 1:
            return possible_roots[0]
        elif len(possible_roots) > 1:
            raise ValueError(f"More than one root found. Possible roots are {possible_roots}")
        else:
            raise ValueError(f"No root found.")

    def __hash__(self):
        return hash(id(self))

    @property
    def active_degrees_of_freedom(self) -> Set[DegreeOfFreedom]:
        dofs = set()
        for connection in self.connections:
            if isinstance(connection, ActiveConnection):
                dofs.update(set(connection.active_dofs))
        return dofs

    @property
    def passive_degrees_of_freedom(self) -> Set[DegreeOfFreedom]:
        dofs = set()
        for connection in self.connections:
            if isinstance(connection, PassiveConnection):
                dofs.update(set(connection.passive_dofs))
        return dofs

    def validate(self) -> bool:
        """
        Validate the world.

        The world must be a tree.
        :return: True if the world is valid, raises an AssertionError otherwise.
        """
        assert len(self.bodies) == (len(self.connections) + 1)
        assert rx.is_weakly_connected(self.kinematic_structure)
        return True

    @modifies_world
    def create_degree_of_freedom(self, name: PrefixedName, lower_limits: Optional[DerivativeMap[float]] = None,
                                 upper_limits: Optional[DerivativeMap[float]] = None) -> DegreeOfFreedom:
        """
        Create a degree of freedom in the world and return it.
        For dependent kinematics, DoFs must be created with this method and passed to the connection's conctructor.
        :param name: Name of the DoF.
        :param lower_limits: If the DoF is actively controlled, it must have at least velocity limits.
        :param upper_limits: If the DoF is actively controlled, it must have at least velocity limits.
        :return: The already registered DoF.
        """
        dof = DegreeOfFreedom(name=name, lower_limits=lower_limits, upper_limits=upper_limits, _world=self)
        initial_position = 0
        lower_limit = dof.lower_limits.position
        if lower_limit is not None:
            initial_position = max(lower_limit, initial_position)
        upper_limit = dof.upper_limits.position
        if upper_limit is not None:
            initial_position = min(upper_limit, initial_position)
        self.state[name].position = initial_position
        assert [dof for dof in self.degrees_of_freedom if dof.name == name].count(dof) == 0
        self.degrees_of_freedom.append(dof)
        return dof

    def modify_world(self) -> WorldModelUpdateContextManager:
        return WorldModelUpdateContextManager(self)

    def reset_state_context(self) -> ResetStateContextManager:
        return ResetStateContextManager(self)

    def reset_cache(self) -> None:
        self.clear_all_lru_caches()
        for dof in self.degrees_of_freedom:
            dof.reset_cache()

    def clear_all_lru_caches(self):
        for method_name in dir(self):
            try:
                method = getattr(self, method_name)
                if hasattr(method, 'cache_clear') and callable(method.cache_clear):
                    method.cache_clear()
            except AttributeError:
                # Skip attributes that can't be accessed
                pass

    def notify_state_change(self) -> None:
        """
        If you have changed the state of the world, call this function to trigger necessary events and increase
        the state version.
        """
        # self.compute_fk.cache_clear()
        # self.compute_fk_with_collision_offset_np.cache_clear()
        self._recompute_forward_kinematics()
        self._state_version += 1

    def _notify_model_change(self) -> None:
        """
        Call this function if you have changed the model of the world to trigger necessary events and increase
        the model version number.
        """
        if not self.world_is_being_modified:
            # self._fix_tree_structure()
            self.reset_cache()
            self.compile_forward_kinematics_expressions()
            # self._cleanup_unused_dofs()
            self.notify_state_change()
            self._model_version += 1
            self.validate()
            self.disable_non_robot_collisions()
            self.disable_collisions_for_adjacent_bodies()

    @property
    def bodies(self) -> List[Body]:
        """
        :return: A list of all bodies in the world.
        """
        return list(self.kinematic_structure.nodes())

    @property
    def bodies_with_collisions(self) -> List[Body]:
        """
        :return: A list of all bodies in the world that have collisions.
        """
        return [body for body in self.bodies if body.has_collision()]

    @property
    def connections(self) -> List[Connection]:
        """
        :return: A list of all connections in the world.
        """
        return list(self.kinematic_structure.edges())

    @property
    def frozen_connections(self) -> Set[Connection]:
        return set(c for c in self.connections
                   if isinstance(c, ActiveConnection) and (not c.is_controlled or c.frozen_for_collision_avoidance))

    @modifies_world
    def add_body(self, body: Body) -> None:
        """
        Add a body to the world.

        :param body: The body to add.
        """
        if body._world is self and body.index is not None:
            return
        elif body._world is not None and body._world is not self:
            raise NotImplementedError("Cannot add a body that already belongs to another world.")

        body.index = self.kinematic_structure.add_node(body)

        # write self as the bodys world
        body._world = self

    @modifies_world
    def add_connection(self, connection: Connection) -> None:
        """
        Add a connection and the bodies it connects to the world.

        :param connection: The connection to add.
        """
        self.add_body(connection.parent)
        self.add_body(connection.child)
        connection._world = self
        self.kinematic_structure.add_edge(connection.parent.index, connection.child.index, connection)

    def add_view(self, view: View, exists_ok: bool = False) -> None:
        """
        Adds a view to the current list of views if it doesn't already exist. Ensures
        that the `view` is associated with the current instance and maintains the
        integrity of unique view names.

        :param view: The view instance to be added. Its name must be unique within
            the current context.
        :param exists_ok: Whether to raise an error or not when a view already exists.

        :raises AddingAnExistingViewError: If exists_ok is False and a view with the same name and type already exists.
        """
        try:
            self.get_view_by_name(view.name)
            if not exists_ok:
                raise AddingAnExistingViewError(view)
        except ViewNotFoundError:
            view._world = self
            self.views.append(view)

    def get_connections_of_branch(self, root: Body) -> List[Connection]:
        """
        Collect all connections that are below root in the tree.

        :param root: The root body of the branch
        :return: List of all connections in the subtree rooted at the given body
        """

        # Create a custom visitor to collect connections
        class ConnectionCollector(rustworkx.visit.DFSVisitor):
            def __init__(self, world: 'World'):
                self.world = world
                self.connections = []

            def tree_edge(self, edge: Tuple[int, int, Connection]):
                """Called for each tree edge during DFS traversal"""
                self.connections.append(edge[2])  # edge[2] is the connection

        visitor = ConnectionCollector(self)
        rx.dfs_search(self.kinematic_structure, [root.index], visitor)

        return visitor.connections

    def get_bodies_of_branch(self, root: Body) -> List[Body]:
        """
        Collect all bodies that are below root in the tree.

        :param root: The root body of the branch
        :return: List of all bodies in the subtree rooted at the given body (including the root)
        """

        # Create a custom visitor to collect bodies
        class BodyCollector(rustworkx.visit.DFSVisitor):
            def __init__(self, world: World):
                self.world = world
                self.bodies = []

            def discover_vertex(self, node_index: int, time: int) -> None:
                """Called when a vertex is first discovered during DFS traversal"""
                body = self.world.kinematic_structure[node_index]
                self.bodies.append(body)

        visitor = BodyCollector(self)
        rx.dfs_search(self.kinematic_structure, [root.index], visitor)

        return visitor.bodies

    def get_view_by_name(self, name: Union[str, PrefixedName]) -> Optional[View]:
        """
        Retrieves a View from the list of view based on its name.
        If the input is of type `PrefixedName`, it checks whether the prefix is specified and looks for an
        exact match. Otherwise, it matches based on the name's string representation.
        If more than one body with the same name is found, an assertion error is raised.
        If no matching body is found, a `ValueError` is raised.

        :param name: The name of the view to search for. Can be a string or a `PrefixedName` object.
        :return: The `View` object that matches the given name.
        :raises ValueError: If multiple or no views with the specified name are found.
        :raises KeyError: If no view is found.
        """
        if isinstance(name, PrefixedName):
            if name.prefix is not None:
                matches = [view for view in self.views if view.name == name]
            else:
                matches = [view for view in self.views if view.name.name == name.name]
        else:
            matches = [view for view in self.views if view.name.name == name]
        if len(matches) > 1:
            raise DuplicateViewError(matches)
        if matches:
            return matches[0]
        raise ViewNotFoundError(name)

    def get_world_state_symbols(self) -> List[cas.Symbol]:
        """
        Constructs and returns a list of symbols representing the state of the system. The state
        is defined in terms of positions, velocities, accelerations, and jerks for each degree
        of freedom specified in the current state.

        :raises KeyError: If a degree of freedom defined in the state does not exist in
            the `degrees_of_freedom`.
        :returns: A combined list of symbols corresponding to the positions, velocities,
            accelerations, and jerks for each degree of freedom in the state.
        """
        positions = [self.get_degree_of_freedom_by_name(v_name).symbols.position for v_name in self.state]
        velocities = [self.get_degree_of_freedom_by_name(v_name).symbols.velocity for v_name in self.state]
        accelerations = [self.get_degree_of_freedom_by_name(v_name).symbols.acceleration for v_name in self.state]
        jerks = [self.get_degree_of_freedom_by_name(v_name).symbols.jerk for v_name in self.state]
        return positions + velocities + accelerations + jerks

    def get_views_by_type(self, view_type: Type[Generic[T]]) -> List[T]:
        """
        Retrieves all views of a specific type from the world.

        :param view_type: The class (type) of the views to search for.
        :return: A list of `View` objects that match the given type.
        """
        return [view for view in self.views if isinstance(view, view_type)]

    @modifies_world
    def remove_body(self, body: Body) -> None:
        if body._world is self and body.index is not None:
            self.kinematic_structure.remove_node(body.index)
            body._world = None
            body.index = None
        else:
            logger.debug("Trying to remove a body that is not part of this world.")

    @modifies_world
    def remove_connection(self, connection: Connection) -> None:
        """
        Removes a connection and deletes the corresponding degree of freedom, if it was only used by this connection.
        Might create disconnected bodies, so make sure to add a new connection or delete the child body.

        :param connection: The connection to be removed
        """
        remaining_dofs = set()
        for remaining_connection in self.connections:
            if remaining_connection == connection:
                continue
            remaining_dofs.update(remaining_connection.dofs)

        for dof in connection.dofs:
            if dof not in remaining_dofs:
                self.degrees_of_freedom.remove(dof)
                del self.state[dof.name]
        self.kinematic_structure.remove_edge(connection.parent.index, connection.child.index)

    @modifies_world
    def merge_world(self, other: World, root_connection: Connection = None) -> None:
        """
        Merge a world into the existing one by merging degrees of freedom, states, connections, and bodies.
        This removes all bodies and connections from `other`.

        :param other: The world to be added.
        :param root_connection: If provided, this connection will be used to connect the two worlds. Otherwise, a new Connection6DoF will be created
        :return: None
        """

        self_root = self.root
        other_root = other.root
        for dof in other.degrees_of_freedom:
            self.state[dof.name].position = other.state[dof.name].position
            self.state[dof.name].velocity = other.state[dof.name].velocity
            self.state[dof.name].acceleration = other.state[dof.name].acceleration
            self.state[dof.name].jerk = other.state[dof.name].jerk
            dof._world = self
        self.degrees_of_freedom.extend(other.degrees_of_freedom)

        # do not trigger computations in other
        other.world_is_being_modified = True
        for connection in other.connections:
            other.remove_body(connection.parent)
            other.remove_body(connection.child)
            self.add_connection(connection)

        for body in other.bodies:
            other.remove_body(body)
            self.add_body(body)
        other.world_is_being_modified = False

        connection = root_connection or Connection6DoF(parent=self_root, child=other_root, _world=self)
        self.add_connection(connection)

    @modifies_world
    def move_branch(self, branch_root: Body, new_parent: Body) -> None:
        old_connection = branch_root.parent_connection
        if isinstance(old_connection, FixedConnection):
            new_parent_T_root = self.compute_forward_kinematics(new_parent, branch_root)
            new_connection = FixedConnection(parent=new_parent,
                                             child=branch_root,
                                             _world=self,
                                             origin_expression=new_parent_T_root)
            self.add_connection(new_connection)
            self.remove_connection(old_connection)
        elif isinstance(old_connection, Connection6DoF):
            new_parent_T_root = self.compute_forward_kinematics(new_parent, branch_root)
            new_connection = Connection6DoF(parent=new_parent,
                                            child=branch_root,
                                            _world=self)
            self.add_connection(new_connection)
            self.remove_connection(old_connection)
            # fixme: probably don't do it like that?
            new_connection.origin = new_parent_T_root
        else:
            raise ValueError(f'Cannot move branch: "{branch_root.name}" is not connected with a FixedConnection')

    def merge_world_at_pose(self, other: World, pose: cas.TransformationMatrix) -> None:
        """
        Merge another world into the existing one, creates a 6DoF connection between the root of this world and the root
        of the other world.
        :param other: The world to be added.
        :param pose: world_root_T_other_root, the pose of the other world's root with respect to the current world's root
        """
        root_connection = Connection6DoF(parent=self.root, child=other.root, _world=self)
        root_connection.origin = pose
        self.merge_world(other, root_connection)
        self.add_connection(root_connection)

    def __str__(self):
        return f"{self.__class__.__name__} with {len(self.bodies)} bodies."

    def get_connection(self, parent: Body, child: Body) -> Connection:
        return self.kinematic_structure.get_edge_data(parent.index, child.index)

    def search_for_connections_of_type(self, connection_type: Union[Type[Connection], Tuple[Type[Connection], ...]]) \
            -> List[Connection]:
        return [c for c in self.connections if isinstance(c, connection_type)]

    def search_for_views_of_type(self, view_type: Union[Type[View], Tuple[Type[View], ...]]) \
            -> List[View]:
        return [v for v in self.views if isinstance(v, view_type)]

    @modifies_world
    def clear(self):
        """
        Clears all stored data and resets the state of the instance.
        """
        for body in list(self.bodies):
            self.remove_body(body)

        self.views.clear()
        self.degrees_of_freedom.clear()
        self.state = WorldState()

    def get_body_by_name(self, name: Union[str, PrefixedName]) -> Body:
        """
        Retrieves a body from the list of bodies based on its name.
        If the input is of type `PrefixedName`, it checks whether the prefix is specified and looks for an
        exact match. Otherwise, it matches based on the name's string representation.
        If more than one body with the same name is found, an assertion error is raised.
        If no matching body is found, a `ValueError` is raised.

        :param name: The name of the body to search for. Can be a string or a `PrefixedName` object.
        :return: The `Body` object that matches the given name.
        :raises ValueError: If multiple or no bodies with the specified name are found.
        """
        if isinstance(name, PrefixedName):
            if name.prefix is not None:
                matches = [body for body in self.bodies if body.name == name]
            else:
                matches = [body for body in self.bodies if body.name.name == name.name]
        else:
            matches = [body for body in self.bodies if body.name.name == name]
        if len(matches) > 1:
            raise ValueError(f'Multiple bodies with name {name} found')
        if matches:
            return matches[0]
        raise KeyError(f'Body with name {name} not found')

    def get_degree_of_freedom_by_name(self, name: Union[str, PrefixedName]) -> DegreeOfFreedom:
        """
        Retrieves a DegreeOfFreedom from the list of DegreeOfFreedom based on its name.
        If the input is of type `PrefixedName`, it checks whether the prefix is specified and looks for an
        exact match. Otherwise, it matches based on the name's string representation.
        If more than one body with the same name is found, an assertion error is raised.
        If no matching body is found, a `ValueError` is raised.

        :param name: The name of the DegreeOfFreedom to search for. Can be a string or a `PrefixedName` object.
        :return: The `DegreeOfFreedom` object that matches the given name.
        :raises ValueError: If multiple or no DegreeOfFreedom with the specified name are found.
        """
        if isinstance(name, PrefixedName):
            if name.prefix is not None:
                matches = [dof for dof in self.degrees_of_freedom if dof.name == name]
            else:
                matches = [dof for dof in self.degrees_of_freedom if dof.name.name == name.name]
        else:
            matches = [dof for dof in self.degrees_of_freedom if dof.name.name == name]
        if len(matches) > 1:
            raise ValueError(f'Multiple DegreeOfFreedom with name {name} found')
        if matches:
            return matches[0]
        raise KeyError(f'DegreeOfFreedom with name {name} not found')

    def get_connection_by_name(self, name: Union[str, PrefixedName]) -> Connection:
        """
        Retrieve a connection by its name.
        This method accepts either a string or a `PrefixedName` instance.
        It searches through the list of connections and returns the one
        that matches the given name. If the `PrefixedName` contains a prefix,
        the method ensures the name, including the prefix, matches an existing
        connection. Otherwise, it only considers the unprefixed name. If more than
        one connection matches the specified name, or if no connection is found,
        an exception is raised.

        :param name: The name of the connection to retrieve. Can be a string or
            a `PrefixedName` instance. If a prefix is included in `PrefixedName`,
            it will be used for matching.
        :return: The connection that matches the specified name.
        :raises ValueError: If multiple connections with the given name are found
            or if no connection with the given name exists.
        """
        if isinstance(name, PrefixedName):
            if name.prefix is not None:
                matches = [conn for conn in self.connections if conn.name == name]
            else:
                matches = [conn for conn in self.connections if conn.name.name == name.name]
        else:
            matches = [conn for conn in self.connections if conn.name.name == name]
        if len(matches) > 1:
            raise ValueError(f'Multiple connections with name {name} found')
        if matches:
            return matches[0]
        raise KeyError(f'Connection with name {name} not found')

    @lru_cache(maxsize=None)
    def compute_child_bodies(self, body: Body) -> List[Body]:
        """
        Computes the child bodies of a given body in the world.
        :param body: The body for which to compute child bodies.
        :return: A list of child bodies.
        """
        return list(self.kinematic_structure.successors(body.index))

    def compute_child_bodies_recursive(self, body: Body) -> List[Body]:
        """
        Computes all child bodies of a given body in the world recursively.
        :param body: The body for which to compute child bodies.
        :return: A list of all child bodies.
        """
        children = self.compute_child_bodies(body)
        for child in children:
            children.extend(self.compute_child_bodies_recursive(child))
        return children

    @lru_cache(maxsize=None)
    def compute_parent_body(self, body: Body) -> Body:
        """
        Computes the parent body of a given body in the world.
        :param body: The body for which to compute the parent body.
        :return: The parent body of the given body.
        """
        return next(iter(self.kinematic_structure.predecessors(body.index)))

    @lru_cache(maxsize=None)
    def compute_parent_connection(self, body: Body) -> Connection:
        """
        Computes the parent connection of a given body in the world.
        :param body: The body for which to compute the parent connection.
        :return: The parent connection of the given body.
        """
        return self.kinematic_structure.get_edge_data(self.compute_parent_body(body).index, body.index)

    @lru_cache(maxsize=None)
    def compute_chain_of_bodies(self, root: Body, tip: Body) -> List[Body]:
        if root == tip:
            return [root]
        shortest_paths = rx.all_shortest_paths(self.kinematic_structure, root.index, tip.index, as_undirected=False)

        if len(shortest_paths) == 0:
            raise rx.NoPathFound(f'No path found from {root} to {tip}')

        return [self.kinematic_structure[index] for index in shortest_paths[0]]

    @lru_cache(maxsize=None)
    def compute_chain_of_connections(self, root: Body, tip: Body) -> List[Connection]:
        body_chain = self.compute_chain_of_bodies(root, tip)
        return [self.get_connection(body_chain[i], body_chain[i + 1]) for i in range(len(body_chain) - 1)]

    @lru_cache(maxsize=None)
    def compute_split_chain_of_bodies(self, root: Body, tip: Body) -> Tuple[List[Body], List[Body], List[Body]]:
        """
        Computes the chain between root and tip. Can handle chains that start and end anywhere in the tree.
        :param root: The root body to start the chain from
        :param tip: The tip body to end the chain at
        :return: tuple containing
                    1. chain from root to the common ancestor (excluding common ancestor)
                    2. list containing just the common ancestor
                    3. chain from common ancestor to tip (excluding common ancestor)
        """
        if root == tip:
            return [], [root], []
        root_chain = self.compute_chain_of_bodies(self.root, root)
        tip_chain = self.compute_chain_of_bodies(self.root, tip)
        i = 0
        for i in range(min(len(root_chain), len(tip_chain))):
            if root_chain[i] != tip_chain[i]:
                break
        else:
            i += 1
        common_ancestor = tip_chain[i - 1]
        root_chain = self.compute_chain_of_bodies(common_ancestor, root)
        root_chain = root_chain[1:]
        root_chain = root_chain[::-1]
        tip_chain = self.compute_chain_of_bodies(common_ancestor, tip)
        tip_chain = tip_chain[1:]
        return root_chain, [common_ancestor], tip_chain

    @lru_cache(maxsize=None)
    def compute_split_chain_of_connections(self, root: Body, tip: Body) -> Tuple[List[Connection], List[Connection]]:
        """
        Computes split chains of connections between 'root' and 'tip' bodies. Returns tuple of two Connection lists:
        (root->common ancestor, tip->common ancestor). Returns empty lists if root==tip.

        :param root: The starting `Body` object for the chain of connections.
        :param tip: The ending `Body` object for the chain of connections.
        :return: A tuple of two lists: the first list contains `Connection` objects from the `root` to
            the common ancestor, and the second list contains `Connection` objects from the `tip` to the
            common ancestor.
        """
        if root == tip:
            return [], []
        root_chain, common_ancestor, tip_chain = self.compute_split_chain_of_bodies(root, tip)
        root_chain = root_chain + [common_ancestor[0]]
        tip_chain = [common_ancestor[0]] + tip_chain
        root_connections = []
        for i in range(len(root_chain) - 1):
            root_connections.append(self.get_connection(root_chain[i + 1], root_chain[i]))
        tip_connections = []
        for i in range(len(tip_chain) - 1):
            tip_connections.append(self.get_connection(tip_chain[i], tip_chain[i + 1]))
        return root_connections, tip_connections

    @property
    def layers(self) -> List[List[Body]]:
        return rx.layers(self.kinematic_structure, [self.root.index], index_output=False)

    def bfs_layout(self, scale: float = 1., align: PlotAlignment = PlotAlignment.VERTICAL) -> Dict[int, np.array]:
        """
        Generate a bfs layout for this circuit.

        :return: A dict mapping the node indices to 2d coordinates.
        """
        layers = self.layers

        pos = None
        nodes = []
        width = len(layers)
        for i, layer in enumerate(layers):
            height = len(layer)
            xs = np.repeat(i, height)
            ys = np.arange(0, height, dtype=float)
            offset = ((width - 1) / 2, (height - 1) / 2)
            layer_pos = np.column_stack([xs, ys]) - offset
            if pos is None:
                pos = layer_pos
            else:
                pos = np.concatenate([pos, layer_pos])
            nodes.extend(layer)

        # Find max length over all dimensions
        pos -= pos.mean(axis=0)
        lim = np.abs(pos).max()  # max coordinate for all axes
        # rescale to (-scale, scale) in all directions, preserves aspect
        if lim > 0:
            pos *= scale / lim

        if align == PlotAlignment.HORIZONTAL:
            pos = pos[:, ::-1]  # swap x and y coords

        pos = dict(zip([node.index for node in nodes], pos))
        return pos

    def plot_kinematic_structure(self, scale: float = 1., align: PlotAlignment = PlotAlignment.VERTICAL) -> None:
        """
        Plots the kinematic structure of the world.
        The plot shows bodies as nodes and connections as edges in a directed graph.
        """
        # Create a new figure
        plt.figure(figsize=(12, 8))

        pos = self.bfs_layout(scale=scale, align=align)

        rustworkx.visualization.mpl_draw(self.kinematic_structure, pos=pos, labels=lambda body: str(body.name),
                                         with_labels=True,
                                         edge_labels=lambda edge: edge.__class__.__name__)

        plt.title("World Kinematic Structure")
        plt.axis('off')  # Hide axes
        plt.show()

    def _travel_branch(self, body: Body, visitor: rustworkx.visit.DFSVisitor) -> None:
        """
        Apply a DFS Visitor to a subtree of the kinematic structure.

        :param body: Starting point of the search
        :param visitor: This visitor to apply.
        """
        rx.dfs_search(self.kinematic_structure, [body.index], visitor)

    def compile_forward_kinematics_expressions(self) -> None:
        """
        Traverse the kinematic structure and compile forward kinematics expressions for fast evaluation.
        """
        new_fks = ForwardKinematicsVisitor(self)
        self._travel_branch(self.root, new_fks)
        new_fks.compile_forward_kinematics()
        self._fk_computer = new_fks

    def _recompute_forward_kinematics(self) -> None:
        self._fk_computer.recompute()

    @copy_lru_cache()
    def compose_forward_kinematics_expression(self, root: Body, tip: Body) -> cas.TransformationMatrix:
        """
        :param root: The root body in the kinematic chain.
            It determines the starting point of the forward kinematics calculation.
        :param tip: The tip body in the kinematic chain.
            It determines the endpoint of the forward kinematics calculation.
        :return: An expression representing the computed forward kinematics of the tip body relative to the root body.
        """

        fk = cas.TransformationMatrix()
        root_chain, tip_chain = self.compute_split_chain_of_connections(root, tip)
        connection: Connection
        for connection in root_chain:
            tip_T_root = connection.origin_expression.inverse()
            fk = fk.dot(tip_T_root)
        for connection in tip_chain:
            fk = fk.dot(connection.origin_expression)
        fk.reference_frame = root
        fk.child_frame = tip
        return fk

    def compute_forward_kinematics(self, root: Body, tip: Body) -> cas.TransformationMatrix:
        """
        Compute the forward kinematics from the root body to the tip body.

        Calculate the transformation matrix representing the pose of the
        tip body relative to the root body.

        :param root: Root body for which the kinematics are computed.
        :param tip: Tip body to which the kinematics are computed.
        :return: Transformation matrix representing the relative pose of the tip body with respect to the root body.
        """
        return cas.TransformationMatrix(self.compute_forward_kinematics_np(root, tip))

    def compute_forward_kinematics_np(self, root: Body, tip: Body) -> NpMatrix4x4:
        """
        Compute the forward kinematics from the root body to the tip body, root_T_tip and return it as a 4x4 numpy ndarray.

        Calculate the transformation matrix representing the pose of the
        tip body relative to the root body, expressed as a numpy ndarray.

        :param root: Root body for which the kinematics are computed.
        :param tip: Tip body to which the kinematics are computed.
        :return: Transformation matrix representing the relative pose of the tip body with respect to the root body.
        """
        return self._fk_computer.compute_forward_kinematics_np(root, tip).copy()

    def compute_forward_kinematics_of_all_collision_bodies(self) -> np.ndarray:
        return self._fk_computer.collision_fks

    def transform(self, spatial_object: cas.SpatialType, target_frame: Body) -> cas.SpatialType:
        """
        Transform a given spatial object from its reference frame to a target frame.

        Calculate the transformation from the reference frame of the provided
        spatial object to the specified target frame. Apply the transformation
        differently depending on the type of the spatial object:

        - If the object is a Quaternion, compute its rotation matrix, transform it, and
          convert back to a Quaternion.
        - For other types, apply the transformation matrix directly.

        :param spatial_object: The spatial object to be transformed.
        :param target_frame: The target body frame to which the spatial object should
            be transformed.
        :return: The spatial object transformed to the target frame. If the input object
            is a Quaternion, the returned object is a Quaternion. Otherwise, it is the
            transformed spatial object.
        """
        target_frame_T_reference_frame = self.compute_forward_kinematics(root=target_frame,
                                                                         tip=spatial_object.reference_frame)
        if isinstance(spatial_object, cas.Quaternion):
            reference_frame_R = spatial_object.to_rotation_matrix()
            target_frame_R = target_frame_T_reference_frame @ reference_frame_R
            return target_frame_R.to_quaternion()
        else:
            return target_frame_T_reference_frame @ spatial_object

    def find_dofs_for_position_symbols(self, symbols: List[cas.Symbol]) -> List[DegreeOfFreedom]:
        result = []
        for s in symbols:
            for dof in self.degrees_of_freedom:
                if s == dof.symbols.position:
                    result.append(dof)
        return result

    def compute_inverse_kinematics(self, root: Body, tip: Body, target: cas.TransformationMatrix,
                                   dt: float = 0.05, max_iterations: int = 200,
                                   translation_velocity: float = 0.2, rotation_velocity: float = 0.2) \
            -> Dict[DegreeOfFreedom, float]:
        """
        Compute inverse kinematics using quadratic programming.

        :param root: Root body of the kinematic chain
        :param tip: Tip body of the kinematic chain
        :param target: Desired tip pose relative to the root body
        :param dt: Time step for integration
        :param max_iterations: Maximum number of iterations
        :param translation_velocity: Maximum translation velocity
        :param rotation_velocity: Maximum rotation velocity
        :return: Dictionary mapping DOF names to their computed positions
        """
        ik_solver = InverseKinematicsSolver(self)
        return ik_solver.solve(root, tip, target, dt, max_iterations, translation_velocity, rotation_velocity)

    def apply_control_commands(self, commands: np.ndarray, dt: float, derivative: Derivatives) -> None:
        """
        Updates the state of a system by applying control commands at a specified derivative level,
        followed by backward integration to update lower derivatives.

        :param commands: Control commands to be applied at the specified derivative
            level. The array length must match the number of free variables
            in the system.
        :param dt: Time step used for the integration of lower derivatives.
        :param derivative: The derivative level to which the control commands are
            applied.
        :return: None
        """
        if len(commands) != len(self.degrees_of_freedom):
            raise ValueError(
                f"Commands length {len(commands)} does not match number of free variables {len(self.degrees_of_freedom)}")

        self.state.set_derivative(derivative, commands)

        for i in range(derivative - 1, -1, -1):
            self.state.set_derivative(i, self.state.get_derivative(i) + self.state.get_derivative(i + 1) * dt)
        for connection in self.connections:
            if isinstance(connection, HasUpdateState):
                connection.update_state(dt)
        self.notify_state_change()

    def set_positions_1DOF_connection(self, new_state: Dict[Has1DOFState, float]) -> None:
        for connection, value in new_state.items():
            connection.position = value
        self.notify_state_change()

    def load_collision_srdf(self, file_path: str):
        """
        Creates a CollisionConfig instance from an SRDF file.

        Parse an SRDF file to configure disabled collision pairs or bodies for a given world.
        Process SRDF elements like `disable_collisions`, `disable_self_collision`,
        or `disable_all_collisions` to update collision configuration
        by referencing bodies in the provided `world`.

        :param file_path: The path to the SRDF file used for collision configuration.
        """
        SRDF_DISABLE_ALL_COLLISIONS: str = 'disable_all_collisions'
        SRDF_DISABLE_SELF_COLLISION: str = 'disable_self_collision'
        SRDF_MOVEIT_DISABLE_COLLISIONS: str = 'disable_collisions'

        if not os.path.exists(file_path):
            raise ValueError(f'file {file_path} does not exist')
        srdf = etree.parse(file_path)
        srdf_root = srdf.getroot()
        for child in srdf_root:
            if hasattr(child, 'tag'):
                if child.tag in {SRDF_MOVEIT_DISABLE_COLLISIONS, SRDF_DISABLE_SELF_COLLISION}:
                    body_a_srdf_name: str = child.attrib['link1']
                    body_b_srdf_name: str = child.attrib['link2']
                    body_a = self.get_body_by_name(body_a_srdf_name)
                    body_b = self.get_body_by_name(body_b_srdf_name)
                    if body_a not in self.bodies_with_collisions:
                        continue
                    if body_b not in self.bodies_with_collisions:
                        continue
                    self.add_disabled_collision_pair(body_a, body_b)
                elif child.tag == SRDF_DISABLE_ALL_COLLISIONS:
                    body = self.get_body_by_name(child.attrib['link'])
                    collision_config = CollisionCheckingConfig(disabled=True)
                    body.set_static_collision_config(collision_config)

    @property
    def controlled_connections(self) -> Set[ActiveConnection]:
        """
        A subset of the robot's connections that are controlled by a controller.
        """
        return set(c for c in self.connections if isinstance(c, ActiveConnection) and c.is_controlled)

    def is_controlled_connection_in_chain(self, root: Body, tip: Body) -> bool:
        root_part, tip_part = self.compute_split_chain_of_connections(root, tip)
        connections = root_part + tip_part
        for c in connections:
            if isinstance(c, ActiveConnection) and c.is_controlled and not c.frozen_for_collision_avoidance:
                return True
        return False

    def disable_collisions_for_adjacent_bodies(self):
        """
        Computes pairs of bodies that should not be collision checked because they have no controlled connections
        between them.

        When all connections between two bodies are not controlled, these bodies cannot move relative to each
        other, so collision checking between them is unnecessary.

        :return: Set of body pairs that should have collisions disabled
        """
        body_combinations = set(combinations_with_replacement(self.bodies_with_enabled_collision, 2))
        for body_a, body_b in list(body_combinations):
            if body_a == body_b:
                self.add_disabled_collision_pair(body_a, body_b)
                continue
            if self.is_controlled_connection_in_chain(body_a, body_b):
                continue
            self.add_disabled_collision_pair(body_a, body_b)

    @property
    def disabled_bodies(self) -> Set[Body]:
        return set(b for b in self.bodies_with_collisions if b.collision_config and b.collision_config.disabled)

    @property
    def bodies_with_enabled_collision(self) -> Set[Body]:
        return set(b for b in self.bodies_with_collisions if b.collision_config and not b.collision_config.disabled)

    @property
    def disabled_collision_pairs(self) -> Set[Tuple[Body, Body]]:
        return self._disabled_collision_pairs | self._temp_disabled_collision_pairs

    def add_disabled_collision_pair(self, body_a: Body, body_b: Body):
        """
        Disable collision checking between two bodies
        """
        pair = tuple(sorted([body_a, body_b], key=lambda b: b.name))
        self._disabled_collision_pairs.add(pair)

    def add_temp_disabled_collision_pair(self, body_a: Body, body_b: Body):
        """
        Disable collision checking between two bodies
        """
        pair = tuple(sorted([body_a, body_b], key=lambda b: b.name))
        self._temp_disabled_collision_pairs.add(pair)

    def avoid_collisions(self, view: View, threshold: float):
        """
        Will not enable collision checking for disabled bodies
        :param view:
        :param threshold:
        :return:
        """
        for body in view.bodies:
            body._collision_config.buffer_zone_distance = threshold

    def disable_collision_checking(self, view1: View, view2: Optional[View] = None):
        for body_a in view1.bodies:
            for body_b in view2.bodies:
                self.add_disabled_collision_pair(body_a, body_b)

    def get_directly_child_bodies_with_collision(self, connection: Connection) -> Set[Body]:
        """
        Collect all child Bodies until a movable connection is found.


        :param connection: The connection from the kinematic structure whose child bodies will be traversed.
        :return: A set of Bodies that are moved directly by only this connection.
        """

        class BodyCollector(rx.visit.DFSVisitor):
            def __init__(self, world: World):
                self.world = world
                self.bodies = set()

            def discover_vertex(self, node_index: int, time: int) -> None:
                body = self.world.kinematic_structure[node_index]
                if body.has_collision():
                    self.bodies.add(body)

            def tree_edge(self, args: Tuple[int, int, Connection]) -> None:
                parent_index, child_index, e = args
                if (isinstance(e, ActiveConnection)
                        and e.is_controlled
                        and not e.frozen_for_collision_avoidance):
                    raise rx.visit.PruneSearch()

        visitor = BodyCollector(self)
        rx.dfs_search(self.kinematic_structure, [connection.child.index], visitor)

        return visitor.bodies

    @lru_cache(maxsize=None)
    def get_controlled_parent_connection(self, body: Body) -> Connection:
        """
        Traverse the chain up until a controlled active connection is found.
        :param body: The body where the search starts.
        :return: The controlled active connection.
        """
        if body == self.root:
            raise ValueError(f"Cannot get controlled parent connection for root body {self.root.name}.")
        if body.parent_connection in self.controlled_connections:
            return body.parent_connection
        return self.get_controlled_parent_connection(body.parent_body)

    def compute_chain_reduced_to_controlled_joints(self, root: Body, tip: Body) -> Tuple[Body, Body]:
        """
        Removes root and tip links until they are both connected with a controlled connection.
        Useful for implementing collision avoidance.

        1. Compute the kinematic chain of bodies between root and tip.
        2. Remove all entries from link_a downward until one is connected with a connection from this view.
        2. Remove all entries from link_b upward until one is connected with a connection from this view.

        :param root: start of the chain
        :param tip: end of the chain
        :return: start and end link of the reduced chain
        """
        downward_chain, upward_chain = self.compute_split_chain_of_connections(root=root, tip=tip)
        chain = downward_chain + upward_chain
        for i, connection in enumerate(chain):
            if (isinstance(connection, ActiveConnection)
                    and connection.is_controlled
                    and not connection.frozen_for_collision_avoidance):
                new_root = connection
                break
        else:
            raise KeyError(f'no controlled connection in chain between {root} and {tip}')
        for i, connection in enumerate(reversed(chain)):
            if (isinstance(connection, ActiveConnection)
                    and connection.is_controlled
                    and not connection.frozen_for_collision_avoidance):
                new_tip = connection
                break
        else:
            raise KeyError(f'no controlled connection in chain between {root} and {tip}')

        if new_root in upward_chain:
            new_root_body = new_root.parent
        else:  # if new_root is in the downward chain, we need to "flip" it by returning its child
            new_root_body = new_root.child
        if new_tip in upward_chain:
            new_tip_body = new_tip.child
        else:  # if new_root is in the downward chain, we need to "flip" it by returning its parent
            new_tip_body = new_tip.parent
        return new_root_body, new_tip_body

    def disable_non_robot_collisions(self) -> None:
        robot_bodies = set()
        robot: AbstractRobot
        for robot in self.search_for_views_of_type(AbstractRobot):
            robot_bodies.update(robot.bodies_with_collisions)

        non_robot_bodies = set(self.bodies_with_enabled_collision) - robot_bodies
        for body_a in non_robot_bodies:
            for body_b in non_robot_bodies:
                self.add_disabled_collision_pair(body_a, body_b)

    def search_for_robot_with_body(self, body: Body) -> Optional[AbstractRobot]:
        robots = [v for v in self.views if isinstance(v, AbstractRobot) and body in v.bodies]
        if len(robots) == 1:
            return robots[0]
        if len(robots) > 1:
            raise ValueError(f"Found multiple robots with body {body.name}")
        return None
