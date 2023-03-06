from stable_baselines3 import PPO

from configs.config import Config
from render.render_env import LegoEnv

def train_callback(*args, **kwargs):
    breakpoint()

def main(cfg: Config):
    env = LegoEnv(cfg=cfg)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50_000)

if __name__ == "__main__":
    cfg = Config()
    main(cfg)
