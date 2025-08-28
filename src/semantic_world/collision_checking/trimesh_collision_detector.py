from dataclasses import dataclass, field
from typing import Optional, Set, List

from trimesh.collision import CollisionManager

from .collision_detector import CollisionDetector, CollisionCheck, Collision
from ..world import World
from ..world_entity import Body


@dataclass
class TrimeshCollisionDetector(CollisionDetector):
    collision_manager: CollisionManager = field(default_factory=CollisionManager, init=False)
    """
    The collision manager from trimesh to handle collision detection
    """
    _last_synced_state: Optional[int] = field(default=None, init=False)
    """
    Last synced state version of the world
    """
    _last_synced_model: Optional[int] = field(default=None, init=False)
    """
    Last synced model version of the world
    """
    _manager_objects: set[Body] = field(default_factory=set, init=False)

    def sync_world_model(self) -> None:
        """
        Synchronize the collision checker with the current world model
        """
        if self._last_synced_model == self._world._model_version:
            return
        bodies_to_be_removed = self._manager_objects - set(self._world.bodies_with_enabled_collision)
        for body in bodies_to_be_removed:
            self.collision_manager.remove_object(body.name.name)
            self._manager_objects.remove(body)
        bodies_to_be_added = set(self._world.bodies_with_enabled_collision) - self._manager_objects
        for body in bodies_to_be_added:
            self.collision_manager.add_object(body.name.name, body.combined_collision_mesh, body.global_pose.to_np())
            self._manager_objects.add(body)
        self._last_synced_model = self._world._model_version

    def sync_world_state(self) -> None:
        """
        Synchronize the collision checker with the current world state
        """
        if self._last_synced_state == self._world._state_version:
            return
        for body in self._world.bodies_with_enabled_collision:
            self.collision_manager.set_transform(body.name.name, body.global_pose.to_np())
        self._last_synced_state = self._world._state_version

    def check_collisions(self,
                         collision_matrix: Optional[Set[CollisionCheck]] = None) -> List[Collision]:
        """
        Checks for collisions in the current world state. The collision manager from trimesh returns all collisions,
        which are then filtered based on the provided collision matrix. If there are multiple contacts between two bodies,
        only the first contact is returned.

        :param collision_matrix: An optional set of CollisionCheck objects to filter the collisions. If None is provided, all collisions are checked.
        :return: A list of Collision objects representing the detected collisions.
        """
        self.sync_world_model()
        self.sync_world_state()
        collisions = self.collision_manager.in_collision_internal(return_names=True, return_data=True)
        collision_pairs = [(self._world.get_kinematic_structure_entity_by_name(pair[0]),
                            self._world.get_kinematic_structure_entity_by_name(pair[1])) for pair in collisions[1]]
        result_set = set()
        for pair in collision_pairs:
            if (pair in [(c.body_a, c.body_b) for c in collision_matrix or []]
                or (pair[1], pair[0]) in [(c.body_a, c.body_b) for c in collision_matrix or []]
                or collision_matrix is None):
                for data in collisions[2]:
                    if data.names == {pair[0].name.name, pair[1].name.name} or data.names == {
                        pair[1].name.name, pair[0].name.name}:
                        result_set.add(
                            Collision(0.0, pair[0], pair[1], map_P_pa=data.point))

        return list(result_set)

    def check_collision_between_bodies(self, body_a: Body, body_b: Body) -> Optional[Collision]:
        collision = self.check_collisions({CollisionCheck(body_a, body_b, 0.0, self._world)})
        return collision[0] if collision else None

    def reset_cache(self):
        pass