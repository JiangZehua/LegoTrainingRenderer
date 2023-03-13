from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    map_shape: tuple = (5, 5, 5)
    render: bool= False
    max_steps: int = 100
    brick_size_range: tuple = (3, 3, 3)
    save_freq: int = 10
    log_dir: str = 'logs'
    exp_dir: Optional[str] = None  # This gets set automatically depending on other parameters