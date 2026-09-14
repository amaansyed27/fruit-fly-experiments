from __future__ import annotations

import numpy as np

from fruit_fly_experiments.games.pong import DOWN, NEUTRAL, UP


class RandomController:
    """Uniform random baseline: no access to hidden Pong state."""

    def __init__(self, seed: int = 1) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, frame: np.ndarray) -> int:
        return int(self.rng.choice([UP, NEUTRAL, DOWN]))


class NeutralController:
    """Static-paddle control used to test whether simply staying centered is strong."""

    def act(self, frame: np.ndarray) -> int:
        return NEUTRAL


class MatchedRandomController:
    """Random control matched to the fly controller's observed action frequencies.

    The probabilities come from the completed 10-seed Experiment 001 fly runs:
    UP=2487/15000, NEUTRAL=10118/15000, DOWN=2395/15000. This controller
    ignores the image contents, so it tests whether the fly's benefit can be
    explained by its action-frequency bias alone rather than visual/connectome
    sensorimotor structure.
    """

    PROBABILITIES = np.array([2487, 10118, 2395], dtype=np.float64) / 15000.0

    def __init__(self, seed: int = 1) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, frame: np.ndarray) -> int:
        return int(self.rng.choice([UP, NEUTRAL, DOWN], p=self.PROBABILITIES))
