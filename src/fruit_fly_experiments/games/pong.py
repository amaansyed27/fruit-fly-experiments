from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UP = -1
NEUTRAL = 0
DOWN = 1
ACTION_NAMES = {UP: "UP", NEUTRAL: "NEUTRAL", DOWN: "DOWN"}


@dataclass(frozen=True)
class PongSnapshot:
    ball_x: float
    ball_y: float
    fly_paddle_y: float
    opponent_paddle_y: float
    fly_score: int
    opponent_score: int
    rally_steps: int
    fly_hits: int
    fly_misses: int


class PongEnv:
    def __init__(self, seed: int = 1, width: int = 320, height: int = 180) -> None:
        self.width = int(width)
        self.height = int(height)
        self.rng = np.random.default_rng(seed)
        self.paddle_h = 34.0
        self.paddle_w = 5.0
        self.ball_r = 4.0
        self.paddle_speed = 150.0
        self.ball_speed = 135.0
        self.fly_x = 14.0
        self.opp_x = self.width - 14.0
        self.fly_score = 0
        self.opp_score = 0
        self.rally_steps = 0
        self.fly_hits = 0
        self.fly_misses = 0
        self.reset_round(direction=1)

    def reset_round(self, direction: int | None = None) -> None:
        self.fly_y = self.height / 2
        self.opp_y = self.height / 2
        self.ball_x = self.width / 2
        self.ball_y = float(self.rng.uniform(0.25, 0.75) * self.height)
        if direction is None:
            direction = -1 if self.rng.random() < 0.5 else 1
        angle = float(self.rng.uniform(-0.55, 0.55))
        self.ball_vx = direction * self.ball_speed * np.cos(angle)
        self.ball_vy = self.ball_speed * np.sin(angle)
        self.rally_steps = 0

    def snapshot(self) -> PongSnapshot:
        return PongSnapshot(self.ball_x, self.ball_y, self.fly_y, self.opp_y, self.fly_score, self.opp_score, self.rally_steps, self.fly_hits, self.fly_misses)

    def step(self, action: int, dt: float) -> PongSnapshot:
        self.fly_y = float(np.clip(self.fly_y + action * self.paddle_speed * dt, self.paddle_h / 2, self.height - self.paddle_h / 2))
        # Deterministic, deliberately imperfect opponent.
        error = self.ball_y - self.opp_y
        opp_dir = 0 if abs(error) < 7 else (1 if error > 0 else -1)
        self.opp_y = float(np.clip(self.opp_y + opp_dir * self.paddle_speed * 0.72 * dt, self.paddle_h / 2, self.height - self.paddle_h / 2))
        self.ball_x += self.ball_vx * dt
        self.ball_y += self.ball_vy * dt
        if self.ball_y < self.ball_r:
            self.ball_y = self.ball_r
            self.ball_vy = abs(self.ball_vy)
        elif self.ball_y > self.height - self.ball_r:
            self.ball_y = self.height - self.ball_r
            self.ball_vy = -abs(self.ball_vy)

        if self.ball_vx < 0 and self.ball_x - self.ball_r <= self.fly_x + self.paddle_w / 2:
            if abs(self.ball_y - self.fly_y) <= self.paddle_h / 2 + self.ball_r:
                self.fly_hits += 1
                self.ball_x = self.fly_x + self.paddle_w / 2 + self.ball_r
                self.ball_vx = abs(self.ball_vx) * 1.01
                self.ball_vy += (self.ball_y - self.fly_y) * 2.0
        elif self.ball_vx > 0 and self.ball_x + self.ball_r >= self.opp_x - self.paddle_w / 2:
            if abs(self.ball_y - self.opp_y) <= self.paddle_h / 2 + self.ball_r:
                self.ball_x = self.opp_x - self.paddle_w / 2 - self.ball_r
                self.ball_vx = -abs(self.ball_vx) * 1.01
                self.ball_vy += (self.ball_y - self.opp_y) * 1.6

        if self.ball_x < 0:
            self.fly_misses += 1
            self.opp_score += 1
            self.reset_round(direction=1)
        elif self.ball_x > self.width:
            self.fly_score += 1
            self.reset_round(direction=-1)
        else:
            self.rally_steps += 1
        return self.snapshot()

    def render_frame(self, sensor: bool = False, out_shape: tuple[int, int] = (96, 160)) -> np.ndarray:
        h, w = out_shape
        frame = np.zeros((h, w), dtype=np.float32)
        sx, sy = w / self.width, h / self.height
        y_shift = 0.0
        if sensor:
            # Egocentric camera: the fly-controlled paddle is centered. This changes
            # only rendering; the controller still receives pixels, never state values.
            y_shift = self.height / 2 - self.fly_y
        def yy(y: float) -> int:
            return int(np.clip((y + y_shift) * sy, 0, h - 1))
        def rect(cx: float, cy: float, rw: float, rh: float, value: float) -> None:
            x0 = max(0, int((cx - rw / 2) * sx)); x1 = min(w, int((cx + rw / 2) * sx) + 1)
            cy2 = cy + y_shift
            y0 = max(0, int((cy2 - rh / 2) * sy)); y1 = min(h, int((cy2 + rh / 2) * sy) + 1)
            if x0 < x1 and y0 < y1:
                frame[y0:y1, x0:x1] = value
        rect(self.fly_x, self.fly_y, self.paddle_w, self.paddle_h, 0.55)
        rect(self.opp_x, self.opp_y, self.paddle_w, self.paddle_h, 0.45)
        bx, by = int(self.ball_x * sx), yy(self.ball_y)
        rr = max(1, int(self.ball_r * min(sx, sy)))
        frame[max(0, by-rr):min(h, by+rr+1), max(0, bx-rr):min(w, bx+rr+1)] = 1.0
        frame[:, max(0, w//2-1):min(w, w//2+1)] = 0.12
        return frame
