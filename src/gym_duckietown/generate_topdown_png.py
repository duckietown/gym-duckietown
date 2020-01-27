# NOTE: This script should be run from the blender interface, otherwise the bpy module will not be available
# Moreover, blender should be opened from the same place as this script is located

import bpy
import pathlib
import os

from mathutils import Vector

debug = False
standard_size = 1

# List collected from existing maps
scale_list = {
    "duckiebot": 0.12,
    "duckie": 0.06,
    "cone": 0.08,
    "barrier": 0.08,
    "sign": 0.18,  #includes all signs
    "stop": 0.18,
    "building": 0.6,
    "house": 0.5,
    "tree": 0.25,
    "truck": 0.2,
    "bus": 0.2
}

# Use cycles renderer
bpy.context.scene.render.engine = 'CYCLES'

# We should run this in the meshes directory (i.e. start blender from here in the command line)
obj_root = pathlib.Path('./meshes/')

# Set camera
bpy.data.objects['Camera'].location = [0,0,5]
bpy.data.cameras['Camera'].ortho_scale = 2
bpy.data.objects['Camera'].rotation_euler = [0,0,0]
bpy.data.cameras['Camera'].type = 'ORTHO'
bpy.context.scene.render.image_settings.color_mode ='RGBA'


# Before we start, make sure nothing is selected. The importer will select
# imported objects, which allows us to delete them after rendering.
bpy.ops.object.select_all(action='DESELECT')
render = bpy.context.scene.render

for obj_fname in sorted(obj_root.glob('*.obj')):
    
    # import obj
    bpy.ops.import_scene.obj(filepath=str(obj_fname))
    
    # Create parent hierarchy to resize
    o = bpy.data.objects.new( obj_fname.stem, None )
    bpy.context.scene.objects.link( o )
    for obj in bpy.context.selected_objects:
        obj.parent = o
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    
    # Generate bounding box of multi group
    xmin = ymin = zmin = float("inf")
    xmax = ymax = zmax = -float("inf")
    for obj in bpy.context.selected_objects:
        # Get bounding box vertices in world frame
        for corner in obj.bound_box:
            xval, yval, zval = obj.matrix_world * Vector(corner)
            xmin = xval if xval < xmin else xmin
            xmax = xval if xval > xmax else xmax
            ymin = yval if yval < ymin else ymin
            ymax = yval if yval > ymax else ymax
            zmin = zval if zval < zmin else zmin
            zmax = zval if zval > zmax else zmax

    # Bring to origin
    xmid = (xmin + xmax)/2
    ymid = (ymin + ymax)/2
    zmid = (zmin + zmax)/2
    bpy.ops.transform.translate(value=(-xmid,-ymid,-zmid))

    # Scale
    # Get the greatest dimenswion of x,y,z
    xdim = xmax - xmin
    ydim = ymax - ymin
    zdim = zmax - zmin
    
    dims = [xdim, ydim, zdim]
    
    gdim = max(dims)

    # get this dimension's scale
    newscale = standard_size / gdim
    
    # set oter dimensions scale to that
    bpy.ops.transform.resize(value=(newscale, newscale, newscale))
    
    render.filepath = '//pngs/obj-%s' % obj_fname.stem
    bpy.ops.render.render(write_still=True)

    if debug: continue
    
    # Cleaning:
    # Remember which meshes were just imported
    meshes_to_remove = []
    for ob in bpy.context.selected_objects:
        meshes_to_remove.append(ob.data)
        bpy.ops.object.delete()
    #  Remove the meshes from memory too
    for mesh in meshes_to_remove:
        bpy.data.meshes.remove(mesh)
    # Remove the groups
    for g in bpy.data.groups:
        bpy.data.groups.remove(g)

print("Done !")


# Disabled:

   # Create parent hierarchy to resize
    #o = bpy.data.objects.new( "empty", None )
    #bpy.context.scene.objects.link( o )
    #for obj in bpy.context.selected_objects:
    #    obj.parent = o
        
    # Create a group for easier manipulation
    #bpy.ops.group.create(name=obj_fname.stem)
    #for obj in bpy.context.selected_objects:
    #    #bpy.ops.object.group_link(group=obj_fname.stem)
    #    bpy.context.scene.objects.active=obj
    #    bpy.ops.object.group_link(group=obj_fname)
    


    
    #try:
    #    scale = scale_list[obj_fname.stem]
    #except KeyError:
    #    if obj_fname.stem.lower().startswith("sign"):
    #        scale = scale_list["sign"]
    #        #continue #Todo remove
    #    else:
    #        scale = 1
    #        print(f"Used default scale for object {obj_fname}")
    #
    #print(f"Used scale: {scale}")
    #
    #bpy.ops.transform.resize(value=(scale, scale, scale))
    
    
    # Scale
    # Get the greatest dimenswion of x,y,z
    #dims = list(bpy.context.active_object.dimensions)
    #gdim = dims.index(max(dims))
    
    # Set this dimension to 1
    #bpy.context.active_object.dimensions[gdim] = 1
    
    # get this dimension's scale
    #newscale = bpy.context.active_object.scale[gdim]
    
    # set oter dimensions scale to that
    #bpy.context.active_object.scale = (newscale,newscale,newscale)
    
    