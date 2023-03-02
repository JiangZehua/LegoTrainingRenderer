import bpy

# Load your .obj file
bpy.ops.import_scene.obj(filepath="LEGO'S.obj")

# Get the imported object
obj = bpy.context.selected_objects[0]

# Set it as active
bpy.context.view_layer.objects.active = obj

# Switch to edit mode
bpy.ops.object.mode_set(mode="EDIT")

# Split the object into parts whose meshes do not share any vertices
bpy.ops.mesh.separate(type="LOOSE")

# Print all the objects and the number of vertices in each
for obj in bpy.context.selected_objects:
    print(obj.name, len(obj.data.vertices))

# Switch back to object mode
bpy.ops.object.mode_set(mode="OBJECT")

# Get all the separate objects
objects = bpy.context.selected_objects

# Create a new blend file name
filename = "render.blend"

# Write all the objects into the blend file
with bpy.data.libraries.load(filename, link=False) as (data_from, data_to):
    data_to.objects = objects
    
for obj in data_to.objects:
    if obj is not None:
        # Make it real
        bpy.context.collection.objects.link(obj)
        
        # Save it as a separate .obj file with its name
        bpy.ops.export_scene.obj(filepath=obj.name + ".obj", use_selection=True)