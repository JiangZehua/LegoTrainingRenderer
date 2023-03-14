import bpy
import os
import subprocess
import sys

from render.auto_reload import DrawingClass

# HACK: Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = os.path.dirname(parent_dir)
sys.path.append(root_dir)

# FIXME: Imported files will not be updated after editing until we restart Blender. Properly follow guide on making basic
#     Blender add-on to fix this?
from render import utils

#### CONFIGURATION ####

# INSTALL = True
INSTALL = False

from dataclasses import dataclass, field


# @dataclass
# class Config:
#     map_shape = (10, 10, 20)
#     brick_size_range = (6, 6, 3)
#     render = True
#     max_steps = 10000

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

import gym_lego

# Get parent directory of this file. Special/odd behavior for blender.
parent_dir = './'

import argparse
import importlib
import os
import sys

import numpy as np
import torch as th
import yaml
from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.callbacks import tqdm
from stable_baselines3.common.utils import set_random_seed

importlib.reload(gym_lego.envs.lego_env)
importlib.reload(gym_lego)

@dataclass
class Config():
    # env: EnvironmentName = 'Lego-v0'
    env: EnvironmentName = 'LegoMaxFill-v0'

    folder: str = 'logs'
    algo: str = 'ppo'
    n_timesteps: int = 199
    num_threads: int = -1
    n_envs: int = 1
    exp_id: int = 0
    verbose: int = 1
    no_render: bool = False
    deterministic: bool = False
    device: str = 'auto'
    load_best: bool = True
    load_checkpoint: int = -1
    load_last_checkpoint: bool = False
    stochastic: bool = True
    norm_reward: bool = False
    seed: int = 4
    reward_log: str = ''
    gym_packages: list = field(default_factory=lambda: [])
    env_kwargs: dict = field(default_factory=lambda: {
        'render': True,
    })
    custom_objects: dict = field(default_factory=lambda: {})
    progress: bool = True
    random_actions: bool = False  # Ignore model predictions and sample random actions, for debugging



# Print all registered environments

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




import rl_zoo3.import_envs  # noqa: F401 pylint: disable=unused-import
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.load_from_hub import download_from_hub
from rl_zoo3.utils import StoreDict, get_model_path


def enjoy() -> None:  # noqa: C901
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--env", help="environment ID", type=EnvironmentName, default="Lego-v0")
    # parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    # parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    # parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    # parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    # parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    # parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    # parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    # parser.add_argument(
    #     "--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)"
    # )
    # parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    # parser.add_argument("--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="auto", type=str)
    # parser.add_argument(
    #     "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    # )
    # parser.add_argument(
    #     "--load-checkpoint",
    #     type=int,
    #     help="Load checkpoint instead of last model if available, "
    #     "you must pass the number of timesteps corresponding to it",
    # )
    # parser.add_argument(
    #     "--load-last-checkpoint",
    #     action="store_true",
    #     default=False,
    #     help="Load last checkpoint instead of last model if available",
    # )
    # parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    # parser.add_argument(
    #     "--norm-reward", action="store_true", default=False, help="Normalize reward if applicable (trained with VecNormalize)"
    # )
    # parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    # parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    # parser.add_argument(
    #     "--gym-packages",
    #     type=str,
    #     nargs="+",
    #     default=[],
    #     help="Additional external Gym environment package modules to import (e.g. gym_minigrid)",
    # )
    # parser.add_argument(
    #     "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    # )
    # parser.add_argument(
    #     "--custom-objects", action="store_true", default=False, help="Use custom objects to solve loading issues"
    # )
    # parser.add_argument(
    #     "-P",
    #     "--progress",
    #     action="store_true",
    #     default=False,
    #     help="if toggled, display a progress bar using tqdm and rich",
    # )

    args = Config()
    
    # Going through custom gym packages to let them register in the global registory
    # for env_module in args.gym_packages:
    #     importlib.import_module(env_module)

    # env_name: EnvironmentName = args.env
    env_name: EnvironmentName = EnvironmentName(args.env)
    algo = args.algo
    folder = args.folder

    try:
        _, model_path, log_path = get_model_path(
            args.exp_id,
            folder,
            algo,
            env_name,
            args.load_best,
            args.load_checkpoint,
            args.load_last_checkpoint,
        )
    except (AssertionError, ValueError) as e:
        # Special case for rl-trained agents
        # auto-download from the hub
        if "rl-trained-agents" not in folder:
            raise e
        else:
            print("Pretrained model not found, trying to download it from sb3 Huggingface hub: https://huggingface.co/sb3")
            # Auto-download
            download_from_hub(
                algo=algo,
                env_name=env_name,
                exp_id=args.exp_id,
                folder=folder,
                organization="sb3",
                repo_name=None,
                force=False,
            )
            # Try again
            _, model_path, log_path = get_model_path(
                args.exp_id,
                folder,
                algo,
                env_name,
                args.load_best,
                args.load_checkpoint,
                args.load_last_checkpoint,
            )

    print(f"Loading {model_path}")

    # Off-policy algorithm only support one env for now
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    if algo in off_policy_algos:
        args.n_envs = 1

    set_random_seed(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    is_atari = ExperimentManager.is_atari(env_name.gym_id)

    stats_path = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    log_dir = args.reward_log if args.reward_log != "" else None

    env = create_test_env(
        env_name.gym_id,
        n_envs=args.n_envs,
        stats_path=maybe_stats_path,
        seed=args.seed,
        log_dir=log_dir,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    kwargs = dict(seed=args.seed)
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))
        # Hack due to breaking change in v1.6
        # handle_timeout_termination cannot be at the same time
        # with optimize_memory_usage
        if "optimize_memory_usage" in hyperparams:
            kwargs.update(optimize_memory_usage=False)

    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version or args.custom_objects:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, device=args.device, **kwargs)

    obs = env.reset()

    # Deterministic by default except for atari games
    stochastic = args.stochastic or is_atari and not args.deterministic
    deterministic = not stochastic

    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    # For HER, monitor success rate
    successes = []
    lstm_states = None
    episode_start = np.ones((env.num_envs,), dtype=bool)

    generator = range(args.n_timesteps)
    if args.progress:
        if tqdm is None:
            raise ImportError("Please install tqdm and rich to use the progress bar")
        generator = tqdm(generator)

    generator = list(generator)

    class PlayState():
        def __init__(self):
            self.obs = obs
            self.lstm_states = lstm_states
            self.episode_start = episode_start
            self.deterministic = deterministic
            self.episode_reward = episode_reward
            self.ep_len = ep_len


    state = PlayState()

    def timer_callback(scene):
        if len(generator) == 0:
            if args.verbose > 0 and len(successes) > 0:
                print(f"Success rate: {100 * np.mean(successes):.2f}%")

            if args.verbose > 0 and len(episode_rewards) > 0:
                print(f"{len(episode_rewards)} Episodes")
                print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")

            if args.verbose > 0 and len(episode_lengths) > 0:
                print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")

            env.close()
            return 

        generator.pop(0)
        action, lstm_states = model.predict(
            state.obs,  # type: ignore[arg-type]
            state=state.lstm_states,
            episode_start=state.episode_start,
            deterministic=state.deterministic,
        )
        if args.random_actions:
            action = [env.action_space.sample()]

        obs, reward, done, infos = env.step(action)
        state.obs = obs

        state.episode_start = done

        if not args.no_render:
            env.render("human")

        state.episode_reward += reward[0]
        state.ep_len += 1

        if args.n_envs == 1:
            # For atari the return reward is not the atari score
            # so we have to get it from the infos dict
            if is_atari and infos is not None and args.verbose >= 1:
                episode_infos = infos[0].get("episode")
                if episode_infos is not None:
                    print(f"Atari Episode Score: {episode_infos['r']:.2f}")
                    print("Atari Episode Length", episode_infos["l"])

            if done and not is_atari and args.verbose > 0:
                # NOTE: for env using VecNormalize, the mean reward
                # is a normalized reward when `--norm_reward` flag is passed
                print(f"Episode Reward: {state.episode_reward:.2f}")
                print("Episode Length", state.ep_len)
                episode_rewards.append(state.episode_reward)
                episode_lengths.append(state.ep_len)
                state.episode_reward = 0.0
                state.ep_len = 0

            # Reset also when the goal is achieved when using HER
            if done and infos[0].get("is_success") is not None:
                if args.verbose > 1:
                    print("Success?", infos[0].get("is_success", False))

                if infos[0].get("is_success") is not None:
                    successes.append(infos[0].get("is_success", False))
                    episode_reward, ep_len = 0.0, 0


        # # Deselct all objects
        # bpy.ops.object.select_all(action='DESELECT')

        return 1e-10
        # return 0.1

    # Register a simple timer that prints the current time
    bpy.app.timers.register(partial(timer_callback, bpy.context.scene))



# @hydra.main(config_path=parent_dir + 'conf', config_name='config')
def main(cfg: Config):

    # for i in range(30):
    #     place_brick(bpy.context.scene, src_brick, (i, i//2, (i%2)*3), (2, 2, 3))
        
    env = gym.make('Lego-v0', render=True)
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


if __name__ == '__main__':

    context = bpy.context             
    dc = DrawingClass(context, "Draw This On Screen")

    # cfg = Config()
    # main(cfg)
    enjoy()