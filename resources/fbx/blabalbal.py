import bpy

for armature in bpy.data.armatures:
    bpy.data.armatures.remove(armature)
for mesh in bpy.data.meshes:
    bpy.data.meshes.remove(mesh)
for from_obj in bpy.data.objects:
    bpy.data.objects.remove(from_obj)
for material in bpy.data.materials:
    bpy.data.materials.remove(material)
for camera in bpy.data.cameras:
    bpy.data.cameras.remove(camera)
for light in bpy.data.lights:
    bpy.data.lights.remove(light)
for image in bpy.data.images:
    bpy.data.images.remove(image)

file_path = "/home/tom_sch/semantic_world/resources/fbx/dressers_group.fbx"
bpy.ops.import_scene.fbx(filepath=file_path, axis_forward='Y', axis_up='Z')
for obj in bpy.context.scene.objects:
    if obj.type != 'MESH':
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True, isolate_users=True)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

for obj in bpy.context.scene.objects:
    if obj.parent is None and obj.type == 'MESH':
        obj.location = (0, 0, 0)
        bpy.ops.wm.stl_export(filepath=,
                          export_selected_objects=True,
                          forward_axis="Y",
                          up_axis="Z")