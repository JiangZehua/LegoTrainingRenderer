import os
import sys
import subprocess
import bpy
import random


#### CONFIGURATION ####

# INSTALL = True
INSTALL = False

#######################


parent_dir = './'

# Print the python version
print(sys.version)

if INSTALL:
    # Load list of requirements from `requirements.txt`
    with open('requirements.txt') as f:
        required = f.read().splitlines()

    # Install required packages
    for r in required:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', r])

import numpy as np
import torch


# Get parent directory of this file


def main():

    # Create primitive cube
    # bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
    # obj = bpy.context.object

    # Add GeometryNodes
    # gnmod = obj.modifiers.new("GeometryNodes", "NODES")
    
    # Get the source "Brick" object
    src_brick = bpy.data.objects['Brick']

    # Place the brick out of sight
    src_brick.location = (0, 0, -100)

    delete_scene_objects(exclude=[src_brick])

    # Add some basic lighting to the scene
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))

    # Scale the brick # TODO: make this realistic
    src_brick.scale = (10, 10, 10)

    for i in range(100):
        # Create a copy of the brick, and move it to a random location
        brick = src_brick.copy()
        brick.location = (random.uniform(-3, 3), random.uniform(-2, 2), random.uniform(-2, 2))
        bpy.context.scene.collection.objects.link(brick)

        # Select the new plane and make it active and center the view
        bpy.context.view_layer.objects.active = brick
        brick.select_set(True)

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
        for input in inputs:
            print(input)

        x_id = inputs['X'].identifier
        y_id = inputs['Y'].identifier
        z_id = inputs['Depth'].identifier
        material_id = inputs['Material'].identifier

        # Set X and Y current value to random integers
        gnmod[x_id] = random.randint(1, 10)
        gnmod[y_id] = random.randint(1, 10)
        gnmod[z_id] = random.randint(1, 10)

    # Deselct all objects
    bpy.ops.object.select_all(action='DESELECT')

    

def delete_scene_objects(scene=None, exclude={}):
    """Delete a scene and all its objects."""
    if not scene:
        # Use current scene if no argument given
        scene = bpy.context.scene
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
            


if __name__ == '__main__':
    main()