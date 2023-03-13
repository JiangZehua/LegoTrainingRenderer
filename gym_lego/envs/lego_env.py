from functools import partial
import gym
import numpy as np
import torch

from configs.config import Config


class LegoEnv(gym.Env):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.grid = None
        self.bricks = {}
        self.scene = None
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.cfg.map_shape, dtype=np.uint8)
        self.action_space = gym.spaces.MultiDiscrete([self.cfg.map_shape[0], self.cfg.map_shape[1], self.cfg.map_shape[2], *self.cfg.brick_size_range])
        self.n_step = 0
        self.loc, self.size = None, None
        self.has_placed = False

        if self.cfg.render:
            from gym_lego.envs.lego_blender import LegoScene
            self.scene = LegoScene()

    def step(self, action):
        self.n_step += 1
        # print(action)
        # print('poopo')
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
            self.has_placed = False
            return self.observe(), -1, self.is_done(), self.get_info()

        self.has_placed = True
        self.bricks[self.brick_idx] = action
        self.grid[slice_idxs] = self.brick_idx
        self.brick_idx += 1

        # if self.cfg.render:
        #     self.scene.place_brick(loc, scale=size)

        self.loc = loc
        self.size = size

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
            above_slice = [slice_idxs[0], slice_idxs[1], slice(loc[2] - 1, loc[2])]
            if not self.grid[above_slice].eq(0).all():
                return True

            # Check the row below
            below_slice = [slice_idxs[0], slice_idxs[1], slice(loc[2] + size[2], loc[2] + size[2] + 1)]
            if not self.grid[below_slice].eq(0).all():
                return True

            return False

        return True
        

    def reset(self):
        self.n_step = 0
        self.has_placed = False

        # Index of the next brick to be placed.
        self.brick_idx = 1

        # A grid of piece IDs. Each piece on the grid has a unique ID. This grid is used for collision detection.
        self.grid = torch.zeros(self.cfg.map_shape)
        
        # A dict of piece IDs to location and size
        self.bricks = {}

        self.loc, self.size = None, None

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

    def render(self, mode='human'):
        if not self.has_placed:
            return
        if self.loc is None or self.size is None:
            return
        self.scene.place_brick(self.loc, scale=self.size)

class LegoEnvKwargs(LegoEnv):
    def __init__(self, **kwargs):
        cfg = Config()
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        super().__init__(cfg=cfg)

            