import logging
import os
from collections import deque
from dataclasses import dataclass
from typing import Optional, List

try:
    import bpy
    from bpy import types as bpy_types

    BlenderObject = bpy_types.Object
except ImportError:
    bpy = None
    BlenderObject = None
    logging.warn("bpy not found")

from ..connections import Connection6DoF
from ..geometry import TriangleMesh
from ..prefixed_name import PrefixedName
from ..spatial_types.spatial_types import TransformationMatrix
from ..world import World, Body

import numpy as np
import trimesh


def blender_mesh_to_trimesh(obj: BlenderObject,
                            apply_modifiers: bool = True,
                            preserve_world_transform: bool = False) -> trimesh.Trimesh:
    """
    Convert a Blender mesh object into a trimesh.Trimesh object.

    This function extracts the geometry information of a Blender mesh object,
    optionally applying modifiers and preserving the world transform when
    transforming vertex positions. The resulting data is returned as a
    trimesh.Trimesh object, suitable for use with the trimesh library.

    :param obj: The Blender object to convert. Must be of type 'MESH'.
    :type obj: BlenderObject
    :param apply_modifiers: If True, apply all modifiers to the Blender mesh before extracting geometry.
    :type apply_modifiers: bool
    :param preserve_world_transform: If True, preserve the object's world transformation when extracting vertex data.
    :type preserve_world_transform: bool
    :return: A trimesh.Trimesh object representing the geometry of the Blender mesh.
    :rtype: trimesh.Trimesh
    :raises TypeError: If the input `obj` is not of type 'MESH'.
    """
    if obj.type != 'MESH':
        raise TypeError(f"Object {obj.name} is not a mesh")

    # Evaluate through depsgraph if modifiers should be applied
    if apply_modifiers:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh()
    else:
        mesh = obj.to_mesh()

    try:
        # Fetch vertex positions
        if preserve_world_transform:
            mat = obj.matrix_world  # 4×4 transform
            verts = np.array([mat @ v.co for v in mesh.vertices], dtype=np.float64)
        else:
            verts = np.array([v.co[:] for v in mesh.vertices], dtype=np.float64)

        # Faces (vertex indices)
        faces = np.array([p.vertices[:] for p in mesh.polygons],
                         dtype=np.int64)

        # Optional: vertex normals, UVs, etc.
        # normals = np.array([p.normal[:] for p in mesh.polygons])
        # uv_layer = mesh.uv_layers.active.data

        tri_mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                                   process=True, validate=True)
        return tri_mesh
    finally:
        # Important: free the mesh datablock we created with to_mesh()
        if apply_modifiers:
            obj_eval.to_mesh_clear()
        else:
            obj.to_mesh_clear()


def get_min_z(root: BlenderObject) -> float:
    min_z = np.inf
    for x, y, z in root.bound_box:
        min_z = min(z, min_z)
    return min_z


@dataclass
class FBXParser:
    file_path: str
    """
    The path of the FBX file.
    """

    prefix: Optional[str] = None
    """
    The prefix for every name used in this world.
    """

    def __post_init__(self):
        if self.prefix is None:
            self.prefix = os.path.basename(self.file_path).split('.')[0]

    def parse(self) -> List[World]:
        """
        Parse the FBX file into a list of World objects.
        The fbx file is split at the top group level.
        """

        # === Clean the scene ===
        bpy.ops.wm.read_factory_settings(use_empty=True)

        # import the fbx
        bpy.ops.import_scene.fbx(filepath=self.file_path, axis_forward='X', axis_up='Z', global_scale=1.,
                                 # bake_space_transform=True,
                                 # automatic_bone_orientation=False,  # keep bones untouched
                                 )

        for obj in bpy.context.scene.objects:
            if obj.type != 'MESH':
                bpy.data.objects.remove(obj, do_unlink=True)
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True, isolate_users=True)
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

        return [self.parse_single_world(obj) for obj in bpy.context.scene.objects
                if obj.type == 'MESH' and obj.parent is None]

    def parse_single_world(self, root: BlenderObject) -> World:
        """
        Create a world from a group in the FBX file.

        :param root: The root of the group.
        :return: The world
        """

        # move the group to the center of the world
        root.location = (0, 0, 0)
        min_z = get_min_z(root)

        # create the resulting world and make a footprint as root
        world = World(primary_prefix=self.prefix)
        base_foot_print = Body(name=PrefixedName(f"{self.prefix}_footprint", self.prefix))
        world.add_body(base_foot_print)

        # initialize parent dict
        obj_to_body_map = {None: base_foot_print}

        # go through objects in bfs order
        object_queue = deque([root])
        with world.modify_world():
            while object_queue:
                obj = object_queue.popleft()

                name = PrefixedName(name=obj.name_full, prefix=self.prefix)

                # convert the mesh to trimesh
                mesh = blender_mesh_to_trimesh(obj)
                origin = np.array(obj.matrix_local)
                origin[2][3] -= min_z
                origin = TransformationMatrix(origin)
                shape = TriangleMesh(origin=origin, mesh=mesh)

                # create the body
                body = Body(name=name, visual=[shape], collision=[shape], )
                world.add_body(body)

                # memoize this and create a connection
                obj_to_body_map[obj] = body
                connection = Connection6DoF(parent=obj_to_body_map[obj.parent],
                                            child=body,
                                            _world=world)
                world.add_connection(connection)

                object_queue.extend(obj.children)

        return world
