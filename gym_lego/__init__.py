from gym.envs.registration import register

from gym_lego.envs.lego_env import LegoEnvKwargs, LegoMaxFillEnv

register(
    id='Lego-v0',
    entry_point='gym_lego.envs.lego_env:LegoEnvKwargs',
    # max_episode_steps=1000,
    # reward_threshold=1.0,
)

register(
    id='LegoMaxFill-v0',
    entry_point='gym_lego.envs.lego_env:LegoMaxFillEnv',
)