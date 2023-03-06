
from dataclasses import dataclass


@dataclass
class Config:
    map_shape = (5, 5, 5)
    render = False
    max_steps = 100
    brick_size_range = (3, 3, 3)