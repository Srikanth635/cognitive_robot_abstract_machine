from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable

import numpy as np
import rclpy  # type: ignore
import semantic_world_msgs.msg
import sqlalchemy.orm
from ormatic.dao import to_dao
from rclpy.subscription import Subscription
from sqlalchemy import select

from ...orm.ormatic_interface import *
from ...prefixed_name import PrefixedName
from ...world import World


@dataclass
class WorldSynchronizer:
    """
    A ros node that synchronizes the world between different processes.
    """

    node: "rclpy.Node"
    """
    The rclpy node used to create the publishers and subscribers.
    """

    world: World
    """
    The world to synchronize.
    """

    publish: bool = True
    """
    Whether to publish the world state.
    """

    subscribe: bool = True
    """
    Whether to subscribe to the world state.
    """

    session: Optional[sqlalchemy.orm.Session] = None
    """
    Session used to communicate model changes through a database.
    """

    previous_world_state_data: np.ndarray = field(init=False, default=None)
    """
    The previous world state data used to check if something changed.
    """

    state_subscriber: Optional[Subscription] = field(default=None, init=False)
    """
    The subscriber to the world state.
    """

    model_change_subscriber: Optional[Subscription] = field(default=None, init=False)
    """
    The subscriber to the model changes.
    """

    reload_model_subscriber: Optional[Subscription] = field(default=None, init=False)
    """
    The subscriber to the reloading the model from the database.
    """

    reload_model_callback: Optional[Callable] = field(default=None, init=False)
    """
    Pointer to the callback used to signal the reload of the world model from the database.
    This is used to avoid calling the callback multiple times.
    """

    def __post_init__(self):
        """
        Initializes callbacks and subscriptions for publishing or subscribing to updates
        from the world's state and model.
        """

        if self.publish:
            self.world.state_change_callbacks.append(lambda: self.publish_state())

            if self.session:
                self.reload_model_callback = lambda: self.publish_reload_model()
                self.world.model_change_callbacks.append(self.reload_model_callback)
        self.update_previous_world_state()

        if self.subscribe:
            self.state_subscriber = self.node.create_subscription(
                semantic_world_msgs.msg.WorldState,
                topic="/semantic_world/world_state",
                callback=self.update_state,
                qos_profile=10,
            )

            if self.session:
                self.reload_model_subscriber = self.node.create_subscription(
                    semantic_world_msgs.msg.WorldModelReload,
                    topic="/semantic_world/reload_model",
                    callback=self.reload_model,
                    qos_profile=10,
                )

    def update_previous_world_state(self):
        """
        Update the previous world state to reflect the current world positions.
        """
        self.previous_world_state_data = np.copy(self.world.state.positions)

    @cached_property
    def state_publisher(self):
        return self.node.create_publisher(
            semantic_world_msgs.msg.WorldState,
            topic="/semantic_world/world_state",
            qos_profile=10,
        )

    @cached_property
    def reload_model_publisher(self):
        return self.node.create_publisher(
            semantic_world_msgs.msg.WorldModelReload,
            topic="/semantic_world/reload_model",
            qos_profile=10,
        )

    @cached_property
    def model_change_publisher(self):
        return self.node.create_publisher(
            semantic_world_msgs.msg.WorldModelModificationBlock,
            topic="/semantic_world/model_change",
            qos_profile=10,
        )

    def publish_state(self):
        """
        Publish the current world state to the ROS topic.
        """
        changes = {
            name: current_state
            for name, previous_state, current_state in zip(
                self.world.state.keys(),
                self.previous_world_state_data,
                self.world.state.positions,
            )
            if not np.allclose(previous_state, current_state)
        }

        if not changes:
            return

        msg = semantic_world_msgs.msg.WorldState(
            version=self.world._state_version,
            states=[
                semantic_world_msgs.msg.DegreeOfFreedomState(
                    name=semantic_world_msgs.msg.PrefixedName(
                        name=key.name, prefix=key.prefix
                    ),
                    position=value,
                )
                for key, value in changes.items()
            ],
        )
        self.state_publisher.publish(msg)
        # removed blocking sleep that stalled callbacks
        self.update_previous_world_state()

    def publish_reload_model(self):
        """
        Save the current world model to the database and publish the primary key to the ROS topic such that other
        processes can subscribe to the model changes and update their worlds.
        """
        dao: WorldMappingDAO = to_dao(self.world)
        self.session.add(dao)
        self.session.commit()
        message = semantic_world_msgs.msg.WorldModelReload(primary_key=dao.id)
        self.reload_model_publisher.publish(message)

    def reload_model(self, msg: semantic_world_msgs.msg.WorldModelReload):
        """
        Update the world with the new model by fetching it from the database.

        :param msg: The message containing the primary key of the model to be fetched.
        """
        query = select(WorldMappingDAO).where(WorldMappingDAO.id == msg.primary_key)
        new_world = self.session.scalars(query).one().from_dao()
        self._replace_world(new_world)
        self.world._notify_model_change([self.reload_model_callback])

    def _replace_world(self, new_world: World):
        """
        Replaces the current world with a new one, updating all relevant attributes.
        This method modifies the existing world state, kinematic structure, degrees
        of freedom, and views based on the `new_world` provided.

        If you encounter any issues with references to dead objects, it is most likely due to this method not doing
        everything needed.

        :param new_world: The new world instance to replace the current world.
        """
        self.world.state = new_world.state
        self.world.kinematic_structure = new_world.kinematic_structure
        self.world.degrees_of_freedom = new_world.degrees_of_freedom
        self.world.views = new_world.views

    def update_state(self, msg: semantic_world_msgs.msg.WorldState):
        """
        Update the world state with the provided message.

        :param msg: The message containing the new state information.
        """
        # Parse incoming states: WorldState has 'states' only
        indices = [
            self.world.state._index[
                PrefixedName(dof_state.name.name, dof_state.name.prefix)
            ]
            for dof_state in msg.states
        ]
        positions = [dof_state.position for dof_state in msg.states]

        if indices:
            self.world.state.data[0, indices] = np.asarray(positions, dtype=float)
            self.update_previous_world_state()
            self.world.notify_state_change()

    def publish_model_change(self):
        ...

    def apply_model_change(self, msg: semantic_world_msgs.msg.WorldModelModificationBlock):
        ...