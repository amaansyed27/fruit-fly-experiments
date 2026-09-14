from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import numpy as np

from fruit_fly_experiments.brain.simulator import BrainSimulator, StepResult
from fruit_fly_experiments.games.pong import DOWN, NEUTRAL, UP
from fruit_fly_experiments.vision.encoder import EncodedVision, FlyVisualEncoder

STEERING_TYPES = ["DNa01", "DNa02", "DNa03", "DNa11", "DNb02", "DNg13"]

@dataclass(frozen=True)
class FlyDecision:
    action: int
    left_activity: float
    right_activity: float
    vision: EncodedVision
    brain: StepResult

class FlyController:
    def __init__(self, simulator: BrainSimulator, encoder: FlyVisualEncoder, trace_steps: int = 8, deadband: float = 0.08) -> None:
        self.simulator = simulator
        self.encoder = encoder
        neurons = simulator.graph.neurons
        self.left = neurons.indices_for_types(STEERING_TYPES, side="L")
        self.right = neurons.indices_for_types(STEERING_TYPES, side="R")
        if len(self.left) == 0 or len(self.right) == 0:
            # DNa02 is present bilaterally in MaleCNS v1.0; fail loudly if a changed
            # dataset/annotation policy removes the intended biological readout.
            raise ValueError("bilateral steering descending-neuron populations were not found")
        self.trace_steps = int(trace_steps)
        self.deadband = float(deadband)
        self.history = deque(maxlen=self.trace_steps)
        self.last_decision: FlyDecision | None = None

    def act(self, frame: np.ndarray) -> int:
        vision = self.encoder.encode(frame)
        result = self.simulator.step(vision.neuron_indices, vision.drive)
        fired = result.spikes
        left_now = float(np.isin(fired, self.left, assume_unique=False).sum() / max(1, len(self.left)))
        right_now = float(np.isin(fired, self.right, assume_unique=False).sum() / max(1, len(self.right)))
        self.history.append((left_now, right_now))
        weights = np.linspace(0.35, 1.0, len(self.history), dtype=np.float32)
        history = np.asarray(self.history, dtype=np.float32)
        left_activity = float(np.average(history[:, 0], weights=weights))
        right_activity = float(np.average(history[:, 1], weights=weights))
        diff = right_activity - left_activity
        action = NEUTRAL if abs(diff) < self.deadband else (DOWN if diff > 0 else UP)
        self.last_decision = FlyDecision(action, left_activity, right_activity, vision, result)
        return action
