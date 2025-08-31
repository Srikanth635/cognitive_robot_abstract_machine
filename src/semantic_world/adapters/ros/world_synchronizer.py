import json
import os
import time
import traceback
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable

import numpy as np
import rclpy  # type: ignore
import semantic_world_msgs.msg
import sqlalchemy.orm
from ormatic.dao import to_dao
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from sqlalchemy import select

from ...orm.ormatic_interface import *
from ...prefixed_name import PrefixedName
from ...world import World
from ...world_modification import (
    WorldModelModificationBlock,
    WorldModelModification,
    AddBodyModification,
    RemoveBodyModification,
    AddConnectionModification,
)


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

    state_publisher: Optional[Publisher] = field(init=False, default=None)
    """
    The publisher used to publish the world state.
    """

    state_subscriber: Optional[Subscription] = field(default=None, init=False)
    """
    The subscriber to the world state.
    """

    state_change_callback: Optional[Callable] = field(init=False, default=None)
    """
    The callback function to receive the world state changes from the world and publish them.
    """

    model_change_publisher: Optional[Publisher] = field(init=False, default=None)
    """
    The publisher used to publish model changes.
    """

    model_change_subscriber: Optional[Subscription] = field(default=None, init=False)
    """
    The subscriber to the model changes.
    """

    model_change_callback: Optional[Callable] = field(init=False, default=None)
    """
    The callback function to receive the model changes from the world and publish them.
    """

    reload_model_publisher: Optional[Publisher] = field(init=False, default=None)
    """
    The publisher used to reload model changes.
    """

    reload_model_subscriber: Optional[Subscription] = field(default=None, init=False)
    """
    The subscriber to the reloading the model from the database.
    """

    synchronizer_id: str = field(init=False)
    """
    The unique identifier for this synchronizer. Used to ensure a synchronizer does not apply its own changes.
    """

    def __post_init__(self):
        """
        Initializes callbacks and subscriptions for publishing or subscribing to updates
        from the world's state and model.
        """
        self.synchronizer_id = f"{os.getpid()}_{self.world.name}"
        if self.publish:
            self.state_change_callback = lambda: self.publish_state()
            self.world.state_change_callbacks.append(self.state_change_callback)
            self.model_change_callback = lambda: self.publish_model_change()
            self.world.model_change_callbacks.append(self.model_change_callback)

            self.state_publisher = self.node.create_publisher(
            semantic_world_msgs.msg.WorldState,
            topic="/semantic_world/world_state",
            qos_profile=10)

            self.reload_model_publisher = self.node.create_publisher(
            semantic_world_msgs.msg.WorldModelReload,
            topic="/semantic_world/reload_model",
            qos_profile=10)

            self.model_change_publisher = self.node.create_publisher(
            semantic_world_msgs.msg.WorldModelModificationBlock,
            topic="/semantic_world/model_change",
            qos_profile=10)


        self.update_previous_world_state()

        if self.subscribe:
            self.state_subscriber = self.node.create_subscription(
                semantic_world_msgs.msg.WorldState,
                topic="/semantic_world/world_state",
                callback=self.update_state,
                qos_profile=10,
            )

            self.model_change_subscriber = self.node.create_subscription(
                semantic_world_msgs.msg.WorldModelModificationBlock,
                topic="/semantic_world/model_change",
                callback=self.apply_model_change,
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
        self.world._notify_model_change()

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
        latest_changes = WorldModelModificationBlock.from_modifications(self.world._modifications[-1])
        msg = semantic_world_msgs.msg.WorldModelModificationBlock(source_id=self.synchronizer_id, version=self.world._model_version, modifications=[json.dumps(m.to_json()) for m in latest_changes.modifications])
        self.model_change_publisher.publish(msg)

    def apply_model_change(self, msg: semantic_world_msgs.msg.WorldModelModificationBlock):
        if msg.source_id != self.synchronizer_id:
            changes = WorldModelModificationBlock(modifications=[WorldModelModification.from_json(json.loads(m))
                                                                 for m in msg.modifications])
            changes(self.world)

    def close(self):
        """
        Clean up publishers, subscribers, and detach callbacks from the world to prevent leaks and cross-talk.
        """
        # Remove world callbacks if present
        if self.state_change_callback is not None:
            self.world.state_change_callbacks.remove(self.state_change_callback)
        if self.model_change_callback is not None:
            self.world.model_change_callbacks.remove(self.model_change_callback)

        # Destroy subscribers
        if self.state_subscriber is not None:
            self.node.destroy_subscription(self.state_subscriber)
            self.state_subscriber = None
        if self.model_change_subscriber is not None:
            self.node.destroy_subscription(self.model_change_subscriber)
            self.model_change_subscriber = None
        if self.reload_model_subscriber is not None:
            self.node.destroy_subscription(self.reload_model_subscriber)
            self.reload_model_subscriber = None

        # Destroy publishers
        if self.state_publisher is not None:
            self.node.destroy_publisher(self.state_publisher)
            self.state_publisher = None
        if self.model_change_publisher is not None:
            self.node.destroy_publisher(self.model_change_publisher)
            self.model_change_publisher = None
        if self.reload_model_publisher is not None:
            self.node.destroy_publisher(self.reload_model_publisher)
            self.reload_model_publisher = None

    def __del__(self):
        # Best-effort cleanup if the user forgot to call close()
        try:
            self.close()
        except Exception as e:
            traceback.print_exc()



