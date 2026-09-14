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
    """Decode the fixed MaleCNS steering output without training a readout.

    Activity is compared *relatively* across bilateral steering DNs. This avoids
    throwing away biologically sparse output simply because its absolute firing
    fraction is small. The brain/connectome remains completely fixed.
    """

    def __init__(
        self,
        simulator: BrainSimulator,
        encoder: FlyVisualEncoder,
        trace_steps: int = 10,
        deadband: float = 0.18,
        min_activity: float = 0.001,
    ) -> None:
        self.simulator = simulator
        self.encoder = encoder
        neurons = simulator.graph.neurons
        self.left = neurons.indices_for_types(STEERING_TYPES, side="L")
        self.right = neurons.indices_for_types(STEERING_TYPES, side="R")
        if len(self.left) == 0 or len(self.right) == 0:
            raise ValueError("bilateral steering descending-neuron populations were not found")
        self.trace_steps = int(trace_steps)
        self.deadband = float(deadband)
        self.min_activity = float(min_activity)
        self.history = deque(maxlen=self.trace_steps)
        self.last_decision: FlyDecision | None = None

    def _decode(self, left_activity: float, right_activity: float) -> int:
        total = left_activity + right_activity
        if total < self.min_activity:
            return NEUTRAL
        balance = (right_activity - left_activity) / max(total, 1e-8)
        if abs(balance) < self.deadband:
            return NEUTRAL
        return DOWN if balance > 0 else UP

    def act(self, frame: np.ndarray) -> int:
        vision = self.encoder.encode(frame)
        result = self.simulator.step(vision.neuron_indices, vision.drive)
        fired = result.spikes
        left_now = float(np.isin(fired, self.left, assume_unique=False).sum() / max(1, len(self.left)))
        right_now = float(np.isin(fired, self.right, assume_unique=False).sum() / max(1, len(self.right)))
        self.history.append((left_now, right_now))

        # A short recency-weighted trace converts sparse DN spikes into a motor
        # command lasting long enough to move the Pong paddle between spike events.
        weights = np.linspace(0.30, 1.0, len(self.history), dtype=np.float32)
        history = np.asarray(self.history, dtype=np.float32)
        left_activity = float(np.average(history[:, 0], weights=weights))
        right_activity = float(np.average(history[:, 1], weights=weights))
        action = self._decode(left_activity, right_activity)

        self.last_decision = FlyDecision(action, left_activity, right_activity, vision, result)
        return action
