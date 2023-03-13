from functools import partial
import json
import os

from stable_baselines3 import PPO as PPO_SB3

from configs.config import Config
from render.render_env import LegoEnv


class PPO(PPO_SB3):
    def __init__(self, cfg: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.n_step = 0

    def save(self, save_dir, *args, **kwargs):
        path = os.path.join(save_dir, 'model')
        super().save(path, *args, **kwargs)

        # Convert config to a dict and save.
        cfg_dict = {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith('_')}
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            # yaml.dump(cfg)
            json.dump(cfg_dict, f)

    def load(self, load_dir, *args, **kwargs):
        cfg_path = os.path.join(load_dir, 'config.json')

        with open(cfg_path, 'rb') as f:
            cfg_dict = json.load(f)

        cfg = Config(**cfg_dict)
        model_path = os.path.join(load_dir, 'model')

        # Call this instead of super so that when, inside super's load function, when the model is re-initialized using
        # loaded data, we also feed 
        load_cls = partial(PPO_SB3.__init__, cfg=cfg)
        PPO_SB3.load(path=model_path)
        # super().load(path=model_path)


def init_config(cfg: Config):
    cfg.exp_dir = os.path.join(cfg.log_dir, 'latest')

def train_callback(*args, **kwargs):
    model: PPO = args[0]['self']
    cfg = model.cfg
    if model.n_step % cfg.save_freq == 0:
        model.save(save_dir=cfg.exp_dir)
    model.n_step += 1

def main(cfg: Config):
    init_config(cfg)
    if not os.path.isdir(cfg.log_dir):
        os.makedirs(cfg.log_dir, exist_ok=True)
    env = LegoEnv(cfg=cfg)
    model = PPO(cfg=cfg, policy="MlpPolicy", env=env, verbose=1)

    # Re-load model if it exists.
    if os.path.exists(cfg.exp_dir):
        model.load(load_dir = cfg.exp_dir)

    model.learn(total_timesteps=50_000, callback=train_callback)


if __name__ == "__main__":
    cfg = Config()
    main(cfg)
