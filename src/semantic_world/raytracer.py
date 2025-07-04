from typing import Tuple

import numpy as np
from trimesh import Scene
from trimesh.proximity import closest_point

from .geometry import Mesh
from .world_entity import Body


class RayTracer:

    def __init__(self, world):
        """
        Initializes the RayTracer with the given world.

        :param world: The world to use for ray tracing.
        """
        self.world = world
        self._last_world_model = -1
        self._last_world_state = -1
        self.collision_to_scene = {}

        self.scene = Scene()
        self.update_scene()

    def update_scene(self):
        """
        Updates the ray tracer scene with the current state of the world.
        This method should be called whenever the world changes to ensure the ray tracer has the latest information.
        """
        if self._last_world_model is not self.world._model_version:
            self.add_missing_bodies()
            self._last_world_model = self.world._model_version
        if self._last_world_state is not self.world._state_version:
            self.update_transforms()
            self._last_world_state = self.world._state_version

    def add_missing_bodies(self):
        """
        Adds all bodies from the world to the ray tracer scene that are not already present.
        """
        bodies_to_add = [body for body in self.world.bodies if body.name.name not in "\t".join(self.scene.graph.nodes)]
        for body in bodies_to_add:
            for i, collision in enumerate(body.collision):
                if isinstance(collision, Mesh):
                    self.collision_to_scene = self.scene.add_geometry(collision.mesh,
                                                                      node_name=body.name.name + f"_collision_{i}",
                                                                      parent_node_name="world",
                                                                      transform=self.world.compute_forward_kinematics_np(
                                                                          self.world.root,
                                                                          body) @ collision.origin.to_np())

    def update_transforms(self):
        """
        Updates the transforms of all bodies in the ray tracer scene.
        This is necessary to ensure that the ray tracing uses the correct positions and orientations.
        """
        for body in self.world.bodies:
            for i, collision in enumerate(body.collision):
                if isinstance(collision, Mesh):
                    transform = self.world.compute_forward_kinematics_np(self.world.root,
                                                                         body) @ collision.origin.to_np()
                    self.scene.graph[body.name.name + f"_collision_{i}"] = transform

    def compute_closest_vertex(self, body: Body, point: np.ndarray) -> Tuple[float, Mesh]:
        """
        Computes the closest vertex on a body to a given point in space.

        :param body: The body to compute the closest vertex for.
        :param point: The point in space to find the closest vertex to.
        :return: The closest vertex on the body to the given point.
        """
        vertex = None
        dist = np.linalg.norm(point - self.world.compute_forward_kinematics_np(self.world.root, body)[:3, 3])
        closest_collision = None
        for collision in body.collision:
            if isinstance(collision, Mesh):
                mesh = collision.mesh
                closest_vertex = mesh.kdtree.query(point, k=1)
                if dist < closest_vertex[0]:
                    dist = closest_vertex[0]
                    vertex = closest_vertex[1]
                    closest_collision = collision
        return vertex, closest_collision

    def compute_closest_points_approximate(self, body1: Body, body2: Body) -> Tuple:
        """
        Computes the closest points to any mesh in the scene in a cube defined by the two points.

       :param body1: The first body.
        :param body2: The second body.
        :return: The closest points between the two bodies in the scene.
        """
        self.update_scene()
        collision1 = body1.collision[0].mesh
        collision2 = body2.collision[0].mesh

        closest_point_body1, distance1, triangle_id = closest_point(collision1,
                                                                    [self.world.compute_forward_kinematics_np(
                                                                        self.world.root, body2)[:3, 3]])
        closest_point_body2, distance2, triangle_id = closest_point(collision2,
                                                                    [self.world.compute_forward_kinematics_np(
                                                                        self.world.root, body1)[:3, 3]])
        return closest_point_body1, closest_point_body2

    def compute_closest_points_multi(self, body: Body, others: list[Body]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the closest points to each given body respectively.
        """
        self.update_scene()
        other_origins = [self.world.compute_forward_kinematics_np(self.world.root, other)[:3, 3] for other in others]

        return closest_point(body.collision[0].mesh, other_origins)[:2]  # Returns only the closest points

    def create_segmentation_mask(self, camera_position: np.ndarray, target_position: np.ndarray,
                                 resolution: int = 512) -> np.ndarray:
        """
        Creates a segmentation mask for the ray tracer scene from the camera position to the target position.

        :param camera_position: The position of the camera.t
        :param target_position: The target position to look at.
        :param resolution: The resolution of the segmentation mask.
        :return: A segmentation mask as a numpy array.
        """
        self.update_scene()
        rays = self.create_camera_rays(camera_position, target_position, resolution=resolution)
        points, index_ray, index_tri = self.scene.to_mesh().ray.intersects_location(rays[:, :3], rays[:, 3:6],
                                                                                    multiple_hits=False,
                                                                                    return_index=False)

    def create_depth_map(self, camera_position: np.ndarray, target_position: np.ndarray,
                         resolution: int = 512) -> np.ndarray:
        """
        Creates a depth map for the ray tracer scene from the camera position to the target position.

        :param camera_position: The position of the camera.
        :param target_position: The target position to look at.
        :param resolution: The resolution of the depth map.
        :return: A depth map as a numpy array.
        """
        self.update_scene()
        rays = self.create_camera_rays(camera_position, target_position, resolution=resolution)
        points, index_ray, index_tri = self.scene.to_mesh().ray.intersects_location(rays[:, :3], rays[:, 3:6],
                                                                                    multiple_hits=False,
                                                                                    return_index=False)

    def create_camera_rays(self, camera_position: np.ndarray, target_position: np.ndarray,
                           resolution: int = 512) -> np.ndarray:
        """
        Creates camera rays for the ray tracer scene from the camera position to the target position.

        :param camera_position: The position of the camera.
        :param target_position: The target position to look at.
        :param resolution: The resolution of the camera rays.
        :return: Camera rays as a numpy array.
        """
        self.update_scene()
        self.scene.camera.resolution = (resolution, resolution)
        self.scene.camera.fov = 90.0
        self.scene.graph[self.scene.camera.name] = camera_position
        camera_transform = self.scene.camera.look_at(target_position)
        self.scene.graph[self.scene.camera.name] = camera_transform
        rays = self.scene.camera.to_rays()
        return rays
