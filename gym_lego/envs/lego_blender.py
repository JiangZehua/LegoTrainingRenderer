import bpy
import numpy as np

BRICK_SCALE = 85  # So that a standard height (3) brick is ~9.6mm as per real LEGO bricks.

def delete_scene_objects(scene=None, exclude={}):
    """Delete a scene and all its objects."""
    # if not scene:
        # Use current scene if no argument given
        # scene = bpy.context.scene
    # Select all objects in the scene
    for obj in scene.objects:
        if obj not in exclude:
            obj.select_set(True)
    # Delete selected objects
    bpy.ops.object.delete()
    # Remove orphaned data blocks
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)




class LegoScene:
    def __init__(self):
        scene = bpy.context.scene

        # Enter object mode
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except:
            pass

        # Create primitive cube
        # bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
        # obj = bpy.context.object

        # TODO: Spawn the first brick from scratch instead of relying ong it being already present in the scene.
        # Get the source "Brick" object
        # NOTE: This assumes that the "Brick" object is already present in the scene. If you error out here when running
        # outside of blender. Set render to False.
        src_brick = bpy.data.objects['Brick']

        # Place the brick out of sight
        src_brick.location = (-1, -1, -1)

        # Make brick invisible
        src_brick.hide_render = True

        # Print the X, Y, and Z inputs of the geometry node
        print(f'Source brick:')
        print('X: {}'.format(src_brick.modifiers['GeometryNodes'].node_group.inputs['X']))
        print('Y: {}'.format(src_brick.modifiers['GeometryNodes'].node_group.inputs['Y']))
        print('Z: {}'.format(src_brick.modifiers['GeometryNodes'].node_group.inputs['Depth']))

        # Print the value of the "Material" input
        print(f'Material: {src_brick.modifiers["GeometryNodes"].node_group.inputs["Material"]}')

        bpy.ops.object.select_all(action='DESELECT')
        delete_scene_objects(bpy.context.scene, exclude=[src_brick])

        # Add some basic lighting to the scene
        light_data = bpy.data.lights.new(name="Light", type='SUN')
        light = bpy.data.objects.new(name="Light", object_data=light_data)
        scene.collection.objects.link(light)
        light.location = (0, 0, 10)
        self.sun = light

        # Scale the brick # TODO: make this realistic
        src_brick.scale = (BRICK_SCALE, BRICK_SCALE, BRICK_SCALE)

        self.src_brick = src_brick

        # Add camera
        # bpy.ops.object.camera_add(location=(0, 4, 4), 
        #                   rotation=(-0.7853, 0, 0))
        # bpy.context.scene.camera = bpy.context.object


    def place_brick(self, loc, scale=(1, 1, 1)):
        scale = np.array(scale).astype(float)
        loc = np.array(loc).astype(float)
        loc[0] += scale[0] / 2
        loc[1] += scale[1] / 2
        loc = loc / np.array([125, 125, 312]) * BRICK_SCALE

        # Create a copy of the brick, and move it to a random location
        brick = self.src_brick.copy()
        brick.location = loc
        bpy.context.scene.collection.objects.link(brick)

        # Select the new plane and make it active and center the view
        bpy.context.view_layer.objects.active = brick
        brick.select_set(True)

        # Print the dimensions of the brick in millimeters
        # print('Brick dimensions: {} x {} x {} mm'.format(
        #     src_brick.dimensions[0] * 10,
        #     src_brick.dimensions[1] * 10,
        #     src_brick.dimensions[2] * 10))

        # Get the geometry node
        gnmod = None
        # print('momomo')
        # print(brick.modifiers)
        for gnmod in brick.modifiers:
            # print(gnmod)
            # print(gnmod.type)
            if gnmod.type == "NODES":
                break

        # Print the name of the geometry node
        # print(gnmod.name)
        
        # Print the node group's nodes
        inputs = gnmod.node_group.inputs
        # for input in inputs:
        #     print(input)

        x_id = inputs['X'].identifier
        y_id = inputs['Y'].identifier
        z_id = inputs['Depth'].identifier
        material_id = inputs['Material'].identifier

        # Set X and Y current value to random integers
        gnmod[x_id] = int(scale[0])
        gnmod[y_id] = int(scale[1])
        gnmod[z_id] = int(scale[2])

        # Print names of all available materials
        # for material in bpy.data.materials:
        #     print(material.name)

        # TODO: This geometry node allows us to set a translucent material. How can we do from inside this script?
        # lego_material = bpy.data.materials['Lego']
        # lego_translucent_material = bpy.data.materials['Lego_Translucent']

        # gnmod[material_id] = lego_translucent_material

        # Deselect the brick
        brick.select_set(False)


    def clear(self):
        delete_scene_objects(bpy.context.scene, exclude={self.src_brick, self.sun})

