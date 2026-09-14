from __future__ import annotations
import numpy as np
from fruit_fly_experiments.games.pong import DOWN, NEUTRAL, UP

class RandomController:
    def __init__(self, seed: int = 1) -> None:
        self.rng = np.random.default_rng(seed)
    def act(self, frame: np.ndarray) -> int:
        return int(self.rng.choice([UP, NEUTRAL, DOWN]))
