from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path


class RunLogger:
    FIELDS = [
        "timestamp", "controller", "seed", "step", "simulated_time_s", "action",
        "left_output", "right_output", "ball_x", "ball_y", "paddle_y", "opponent_y",
        "fly_score", "opponent_score", "rally_steps", "fly_hits", "fly_misses",
        "brain_latency_ms", "fps", "active_neurons", "vision_motion_energy",
    ]

    def __init__(self, root: Path, seed: int, controller: str, flush_every: int = 50) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_controller = controller.replace("_", "-")
        self.path = root / "runs" / f"run_{stamp}_{safe_controller}_seed{seed}.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.flush_every = max(1, int(flush_every))
        self._pending = 0

    def write(self, row: dict) -> None:
        self.writer.writerow({k: row.get(k, "") for k in self.FIELDS})
        self._pending += 1
        if self._pending >= self.flush_every:
            self.file.flush()
            self._pending = 0

    def close(self) -> None:
        if not self.file.closed:
            self.file.flush()
            self.file.close()
