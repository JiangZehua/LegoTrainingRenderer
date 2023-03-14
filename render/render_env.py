import bpy
import os
import subprocess
import sys

# Print python version
print(f'Python version: {sys.version}')

# HACK: Add parent directory to path
# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# root_dir = os.path.dirname(parent_dir)
# sys.path.append(root_dir)


# FIXME: Imported files will not be updated after editing until we restart Blender. Properly follow guide on making basic
#     Blender add-on to fix this?
import render
from render import utils
import gym_lego
from gym_lego.envs.lego_env import LegoEnv

import importlib
importlib.reload(render)
importlib.reload(gym_lego)

#### CONFIGURATION ####

# INSTALL = True
INSTALL = False

from dataclasses import dataclass


@dataclass
class Config:
    map_shape = (10, 10, 20)
    brick_size_range = (6, 6, 3)
    render = True
    max_steps = 10000

#######################

if INSTALL:
    render.utils.install_requirements()

import os
from functools import partial
import random
import sys

import gym
import numpy as np
from stable_baselines3 import PPO
import torch


# Add parent directory to path. HACK


# @hydra.main(config_path=parent_dir + 'conf', config_name='config')
def main(cfg: Config):

    # for i in range(30):
    #     place_brick(bpy.context.scene, src_brick, (i, i//2, (i%2)*3), (2, 2, 3))
        
    env = gym_lego.envs.lego_env.LegoEnvKwargs(render=True)
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

            obs, rew, done, info = env.step(action)
            env.render()

        # # Deselct all objects
        # bpy.ops.object.select_all(action='DESELECT')

        return 1e-10
        # return 0.1

    # Register a simple timer that prints the current time
    bpy.app.timers.register(partial(timer_callback, bpy.context.scene, actions))




if __name__ == '__main__':
    cfg = Config()
    main(cfg)