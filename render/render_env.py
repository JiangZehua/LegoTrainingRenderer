import bpy
import os
import subprocess
import sys

sys.path.append(os.getcwd())
# FIXME: Imported files will not be updated after editing until we restart Blender. Properly follow guide on making basic
#     Blender add-on to fix this?
from render import utils

#### CONFIGURATION ####

INSTALL = True
# INSTALL = False

BRICK_SCALE = 85  # So that a standard height (3) brick is ~9.6mm as per real LEGO bricks.

from dataclasses import dataclass


@dataclass
class Config:
    map_shape = (10, 10, 20)
    brick_size_range = (6, 6, 3)
    render = True
    max_steps = 10000

#######################

if INSTALL:
    utils.install_requirements()

import os
from functools import partial
import random
import sys

import gym
import numpy as np
from stable_baselines3 import PPO
import torch

# Print current directory
print(os.getcwd())

# Add parent directory to path. HACK


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



# Get parent directory of this file. Special/odd behavior for blender.
parent_dir = './'


# @hydra.main(config_path=parent_dir + 'conf', config_name='config')
def main(cfg: Config):

    # for i in range(30):
    #     place_brick(bpy.context.scene, src_brick, (i, i//2, (i%2)*3), (2, 2, 3))
        
    env = LegoEnv(cfg)
    env.reset()
    done = False

    # List all possible actions in the multi-discrete action space. Use itertools
    # to generate all possible combinations of actions.
    import itertools

    # All possible actions
    actions = list(itertools.product(*[range(v) for v in env.action_space.nvec]))
    # Randomly shuffle the list of actions
    random.shuffle(actions)

    def timer_callback(scene, actions):
        if env.is_done():
            env.reset()
        else:
            if len(actions) == 0:
                # All possible actions
                actions = list(itertools.product(*[range(v) for v in env.action_space.nvec]))
                # Randomly shuffle the list of actions
                random.shuffle(actions)

            # action = env.action_space.sample()
            action = actions.pop(0)
            # print(action)

            obs, rew, done, info = env.step(action)

        # # Deselct all objects
        # bpy.ops.object.select_all(action='DESELECT')

        return 1e-10
        # return 0.1

    # Register a simple timer that prints the current time
    bpy.app.timers.register(partial(timer_callback, bpy.context.scene, actions))


class LegoScene:
    def __init__(self):
        scene = bpy.context.scene

        # Enter object mode
        bpy.ops.object.mode_set(mode='OBJECT')

        # Create primitive cube
        # bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
        # obj = bpy.context.object

        # TODO: Spawn the first brick from scratch instead of relying ong it being already present in the scene.
        # Get the source "Brick" object
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


class LegoEnv(gym.Env):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.grid = None
        self.bricks = {}
        self.scene = None
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.cfg.map_shape, dtype=np.uint8)
        self.action_space = gym.spaces.MultiDiscrete([self.cfg.map_shape[0], self.cfg.map_shape[1], self.cfg.map_shape[2], *self.cfg.brick_size_range])
        self.n_step = 0

        if self.cfg.render:
            self.scene = LegoScene()

    def step(self, action):
        self.n_step += 1
        # print(action)
        loc_x, loc_y, loc_z, sz_x, sz_y, sz_z = action
        sz_x, sz_y, sz_z = sz_x + 1, sz_y + 1, sz_z + 1
        # sz_x, sz_y, sz_z = 3, 3, 3
        loc = (loc_x, loc_y, loc_z)
        size = (sz_x, sz_y, sz_z)
        slice_idxs = tuple([slice(loc[i], loc[i] + size[i]) for i in (0, 1, 2)])

        # Weird.
        # slice_idxs = slice_idxs[2], slice_idxs[1], slice_idxs[0]

        # print(self.grid.unique())

        if not self.can_place(loc, size, slice_idxs):
            return self.observe(), -1, self.is_done(), self.get_info()

        self.bricks[self.brick_idx] = action
        self.grid[slice_idxs] = self.brick_idx
        # breakpoint()
        self.brick_idx += 1

        if self.cfg.render:
            self.scene.place_brick(loc, scale=size)


        return self.observe(), self.get_reward(), self.is_done(), self.get_info()

    def can_place(self, loc, size, slice_idxs):
        # Get the slice of the grid where we are about to place the block
        trg_slice = self.grid[slice_idxs]

        # The brick cannot overlap with any other brick.
        if not trg_slice.eq(0).all():
            coll_brick = 0
            i = 0
            while coll_brick == 0:
                coll_brick = trg_slice.unique()[i].int().item()
                i += 1

            # print(f'Collision at {loc} with {trg_slice.unique()}. Coll block: {self.bricks[coll_brick]}')
            return False

        # Either the brick must be placed on the ground, or, in the rows above and below the brick, there must be at 
        # least one brick (which this one can connect to).
        if loc[2] > 0:
            # Check the row above
            if not self.grid[loc[0], loc[1], loc[2] - 1].eq(0).all():
                return True

            # Check the row below
            elif loc[2] + size[2] < self.grid.shape[2] and not self.grid[loc[0], loc[1], loc[2] + size[2]].eq(0).all():
                return True

            return False

        return True
        

    def reset(self):
        self.n_step = 0

        # Index of the next brick to be placed.
        self.brick_idx = 1

        # A grid of piece IDs. Each piece on the grid has a unique ID. This grid is used for collision detection.
        self.grid = torch.zeros(self.cfg.map_shape)
        
        # A dict of piece IDs to location and size
        self.bricks = {}

        if self.cfg.render:
            self.scene.clear()

        return self.observe()

    def observe(self):
        occupancy = self.grid > 0
        return occupancy

    def get_reward(self):
        return 0

    def is_done(self):
        # print(self.n_step)
        return self.n_step >= self.cfg.max_steps

    def get_info(self):
        return {}
            


if __name__ == '__main__':
    cfg = Config()
    main(cfg)