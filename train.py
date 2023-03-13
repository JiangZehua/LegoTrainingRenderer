import rl_zoo3
from rl_zoo3.train import train

# Register LegoEnv
import gym
from render.render_env import LegoEnv  # fyi


if __name__ == "__main__":  # noqa: C901
    train()
