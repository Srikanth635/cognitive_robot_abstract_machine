import logging
import os
from dataclasses import dataclass
from typing import Optional
try:
    import bpy
    from bpy import types as bpy_types
    BlenderObject = bpy_types.Object
except ImportError:
    bpy = object
    BlenderObject = object
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
    Convert a Blender mesh object to a trimesh.Trimesh.

    Parameters
    ----------
    obj : bpy.types.Object
        The Blender object to convert. Must be of type 'MESH'.
    apply_modifiers : bool, optional
        Evaluate the object through the dependency graph so that the mesh
        contains all active modifiers. Default: True.
    preserve_world_transform : bool, optional
        Vertex coordinates are exported in world space if True, else in
        object local space.

    Returns
    -------
    trimesh.Trimesh
        The converted trimesh mesh.
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

    def parse(self):

        world = World(primary_prefix=self.prefix)
        root = Body(name=PrefixedName("root", self.prefix))
        world.add_body(root)

        # === Clean the scene ===
        bpy.ops.wm.read_factory_settings(use_empty=True)

        # import the fbx
        bpy.ops.import_scene.fbx(filepath=self.file_path, axis_forward='Y', axis_up='Z', global_scale=0.01,
                                 # bake_space_transform=True,
                                 # automatic_bone_orientation=False,  # keep bones untouched
                                 )

        with world.modify_world():

            for obj in bpy.context.scene.objects:
                if obj.type != "MESH":
                    continue

                name = PrefixedName(name=obj.name_full , prefix=self.prefix)
                mesh = blender_mesh_to_trimesh(obj)
                origin = np.array(obj.matrix_local)
                origin = TransformationMatrix(origin)
                shape = TriangleMesh(origin=origin, mesh=mesh)

                body = Body(name=name, visual=[shape], collision=[shape], )
                world.add_body(body)
                connection = Connection6DoF(parent=root, child=body, _world=world)
                world.add_connection(connection)

        return world