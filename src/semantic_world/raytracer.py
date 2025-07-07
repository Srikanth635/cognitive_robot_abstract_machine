from typing import Tuple

import numpy as np
import trimesh
from trimesh import Scene

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
                self.collision_to_scene[collision] = self.scene.add_geometry(collision.mesh,
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
                transform = self.world.compute_forward_kinematics_np(self.world.root,
                                                                     body) @ collision.origin.to_np()
                self.scene.graph[body.name.name + f"_collision_{i}"] = transform

    def create_segmentation_mask(self, camera_position: np.ndarray,
                                 resolution: int = 512) -> np.ndarray:
        """
        Creates a segmentation mask for the ray tracer scene from the camera position to the target position.
<
        :param camera_position: The position of the camera.t
        :param resolution: The resolution of the segmentation mask.
        :return: A segmentation mask as a numpy array.
        """
        self.update_scene()
        ray_origins, ray_directions, pixels = self.create_camera_rays(camera_position, resolution=resolution)
        points, index_ray, index_tri = self.scene.to_mesh().ray.intersects_location(ray_origins, ray_directions,
                                                                                    multiple_hits=False)
        return points, index_ray, index_tri

    def create_depth_map(self, camera_position: np.ndarray,
                         resolution: int = 512) -> np.ndarray:
        """
        Creates a depth map for the ray tracer scene from the camera position to the target position.

        :param camera_position: The position of the camera.
        :param resolution: The resolution of the depth map.
        :return: A depth map as a numpy array.
        """
        self.update_scene()
        ray_origins, ray_directions, pixels = self.create_camera_rays(camera_position, resolution=resolution)
        # Code from the example in trimesh repo: examples/raytrace.py
        points, index_ray, index_tri = self.scene.to_mesh().ray.intersects_location(ray_origins, ray_directions,
                                                                                    multiple_hits=False)
        depth = trimesh.util.diagonal_dot(points - ray_origins[0], ray_directions[index_ray])
        pixel_ray = pixels[index_ray]

        # create a numpy array we can turn into an image
        # doing it with uint8 creates an `L` mode greyscale image
        a = np.zeros(self.scene.camera.resolution, dtype=np.uint8)

        # scale depth against range (0.0 - 1.0)
        depth_float = (depth - depth.min()) / np.ptp(depth)

        # convert depth into 0 - 255 uint8
        depth_int = (depth_float * 255).round().astype(np.uint8)
        # assign depth to correct pixel locations
        a[pixel_ray[:, 0], pixel_ray[:, 1]] = depth_int

        return a

    def create_camera_rays(self, camera_poe: np.ndarray,
                           resolution: int = 512, fov=90) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates camera rays for the ray tracer scene from the camera position to the target position. Places the camera
        at the given position and orientation view of the camera is along the x-axis.

        :param camera_poe: The position of the camera as a 4x4 transformation matrix.
        :param resolution: The resolution of the camera rays.
        :param fov: The field of view of the camera in degrees.
        :return: The origin points of the rays, the direction vectors of the rays, and the pixel coordinates.
        """
        self.update_scene()
        self.scene.camera.resolution = (resolution, resolution)
        rotate = trimesh.transformations.rotation_matrix(
            angle=np.radians(-90.0), direction=[0, 1, 0], point=self.scene.centroid
        )
        rotate_x = trimesh.transformations.rotation_matrix(
            angle=np.radians(180.0), direction=[1, 0, 0], point=self.scene.centroid
        )

        self.scene.camera.fov = (fov, fov)
        self.scene.camera.resolution  = [resolution, resolution]
        self.scene.graph[self.scene.camera.name] = camera_poe @ rotate_x @ rotate

        return self.scene.camera_rays()
