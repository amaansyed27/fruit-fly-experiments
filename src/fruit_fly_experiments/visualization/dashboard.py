from __future__ import annotations

from collections import deque
import numpy as np

from fruit_fly_experiments.controllers.fly import STEERING_TYPES
from fruit_fly_experiments.games.pong import ACTION_NAMES, PongEnv


class Dashboard:
    def __init__(
        self,
        simulator,
        width: int = 1280,
        height: int = 720,
        sample_neurons: int = 18000,
        seed: int = 7,
    ) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError("pygame is required for the live demo") from exc

        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Experiment 001 — MaleCNS Pong")
        self.w, self.h = width, height
        self.font = pygame.font.SysFont("consolas", 17)
        self.small = pygame.font.SysFont("consolas", 14)
        self.big = pygame.font.SysFont("consolas", 28, bold=True)
        self.clock = pygame.time.Clock()
        self.rng = np.random.default_rng(seed)

        neurons = simulator.graph.neurons
        self.positions = neurons.positions_xyz().astype(np.float32, copy=True)
        valid = np.flatnonzero(np.isfinite(self.positions).all(axis=1))
        self.anatomical = len(valid) >= 100
        self.background_indices = (
            np.sort(self.rng.choice(valid, size=min(sample_neurons, len(valid)), replace=False))
            if self.anatomical
            else np.arange(min(sample_neurons, simulator.n), dtype=np.int64)
        )

        self.view_yaw_offset = 0.0
        self.view_pitch = np.deg2rad(-5.0)
        self._dragging = False
        self._view_yaw = 0.0

        if self.anatomical:
            self._orient_anatomy(neurons)
            shown = self.positions[valid]
            lo = np.nanpercentile(shown, 0.5, axis=0)
            hi = np.nanpercentile(shown, 99.5, axis=0)
            self.positions -= ((lo + hi) * 0.5).astype(np.float32)
            shown = self.positions[valid]
            radial = np.sqrt(shown[:, 0] ** 2 + shown[:, 1] ** 2)
            self.xy_radius = max(float(np.nanpercentile(radial, 99.5)), 1e-6)
            self.z_radius = max(float(np.nanpercentile(np.abs(shown[:, 2]), 99.5)), 1e-6)
            self.depth_radius = max(float(np.nanpercentile(np.abs(shown[:, 1]), 99.5)), 1e-6)
        else:
            sides = self.rng.choice([-1.0, 1.0], simulator.n)
            self.positions = np.column_stack(
                [
                    sides * (0.6 + 0.5 * self.rng.random(simulator.n))
                    + self.rng.normal(0, 0.14, simulator.n),
                    self.rng.normal(0, 0.25, simulator.n),
                    self.rng.normal(0, 1.0, simulator.n),
                ]
            ).astype(np.float32)
            self.xy_radius = 1.5
            self.z_radius = 2.5
            self.depth_radius = 1.0

        left = neurons.indices_for_types(STEERING_TYPES, side="L")
        right = neurons.indices_for_types(STEERING_TYPES, side="R")
        self.output_indices = np.unique(np.concatenate([left, right])).astype(np.int64)
        self.timeline = deque(maxlen=260)
        # Four frames are enough to make sparse spikes readable without letting the
        # overlay cost grow over the course of a demo.
        self.spike_history = deque(maxlen=4)

    def _orient_anatomy(self, neurons) -> None:
        ann = neurons.annotations
        side = np.full(neurons.size, "", dtype=object)
        for col in ("somaSide", "rootSide"):
            if col not in ann.columns:
                continue
            values = ann[col].fillna("").astype(str).str.upper().to_numpy()
            empty = side == ""
            side[empty] = values[empty]

        left = np.flatnonzero(np.isin(side, ["L", "LHS", "LEFT"]))
        right = np.flatnonzero(np.isin(side, ["R", "RHS", "RIGHT"]))
        if len(left) and len(right):
            lx = np.nanmedian(self.positions[left, 0])
            rx = np.nanmedian(self.positions[right, 0])
            if np.isfinite(lx) and np.isfinite(rx) and rx < lx:
                self.positions[:, 0] *= -1.0

        visual = neurons.indices_for_types(["L1", "R7", "R8", "LC10a"])
        visual = visual[np.isfinite(self.positions[visual, 2])] if len(visual) else visual
        all_z = self.positions[np.isfinite(self.positions[:, 2]), 2]
        if len(visual) and len(all_z):
            if np.nanmedian(self.positions[visual, 2]) > np.nanmedian(all_z):
                self.positions[:, 2] *= -1.0

    def pump(self) -> bool:
        for event in self.pg.event.get():
            if event.type == self.pg.QUIT:
                return False
            if event.type == self.pg.KEYDOWN:
                if event.key == self.pg.K_ESCAPE:
                    return False
                if event.key == self.pg.K_r:
                    self.view_yaw_offset = 0.0
                    self.view_pitch = np.deg2rad(-5.0)
            elif event.type == self.pg.MOUSEBUTTONDOWN and event.button == 1:
                self._dragging = True
            elif event.type == self.pg.MOUSEBUTTONUP and event.button == 1:
                self._dragging = False
            elif event.type == self.pg.MOUSEMOTION and self._dragging:
                self.view_yaw_offset += event.rel[0] * 0.010
                self.view_pitch = float(
                    np.clip(self.view_pitch + event.rel[1] * 0.006, -0.55, 0.55)
                )
        return True

    def draw(self, env: PongEnv, decision, fps: float) -> None:
        pg = self.pg
        self.screen.fill((12, 14, 18))
        margin = 22
        bottom_h = 120
        gap = 18
        left_w = int(self.w * 0.67)
        game_rect = pg.Rect(margin, margin, left_w - margin, self.h - bottom_h - margin * 2)
        brain_rect = pg.Rect(
            left_w + gap,
            margin,
            self.w - left_w - gap - margin,
            self.h - bottom_h - margin * 2,
        )
        timeline_rect = pg.Rect(margin, self.h - bottom_h, self.w - 2 * margin, bottom_h - margin)
        self._panel(game_rect)
        self._panel(brain_rect)
        self._panel(timeline_rect)
        self._draw_pong(env, game_rect)
        self._draw_brain(decision, brain_rect)
        self._draw_timeline(decision, timeline_rect, fps)
        pg.display.flip()

    def _panel(self, rect) -> None:
        self.pg.draw.rect(self.screen, (20, 23, 29), rect, border_radius=5)
        self.pg.draw.rect(self.screen, (48, 53, 64), rect, 1, border_radius=5)

    def _text(self, text, x, y, font=None, color=(225, 229, 238)) -> None:
        self.screen.blit((font or self.font).render(str(text), True, color), (x, y))

    def _draw_pong(self, env, rect) -> None:
        pg = self.pg
        self._text("PONG / FLY VIEW", rect.x + 16, rect.y + 14, self.big)
        arena = pg.Rect(rect.x + 16, rect.y + 58, rect.w - 32, rect.h - 76)
        pg.draw.rect(self.screen, (7, 9, 12), arena)
        sx, sy = arena.w / env.width, arena.h / env.height
        for yy in range(arena.y + 10, arena.bottom - 10, 20):
            pg.draw.rect(self.screen, (45, 49, 58), (arena.centerx - 1, yy, 2, 10))

        def y(v):
            return int(arena.y + v * sy)

        pg.draw.rect(
            self.screen,
            (235, 238, 245),
            (
                int(arena.x + env.fly_x * sx - 3),
                y(env.fly_y) - int(env.paddle_h * sy / 2),
                6,
                int(env.paddle_h * sy),
            ),
        )
        pg.draw.rect(
            self.screen,
            (135, 143, 158),
            (
                int(arena.x + env.opp_x * sx - 3),
                y(env.opp_y) - int(env.paddle_h * sy / 2),
                6,
                int(env.paddle_h * sy),
            ),
        )
        pg.draw.circle(
            self.screen,
            (244, 246, 250),
            (int(arena.x + env.ball_x * sx), y(env.ball_y)),
            max(3, int(env.ball_r * sy)),
        )
        self._text(
            f"FLY {env.fly_score}   :   {env.opp_score} OPP",
            arena.centerx - 75,
            arena.y + 8,
            self.small,
        )

    def _current_view(self) -> tuple[float, float]:
        t = self.pg.time.get_ticks() / 1000.0
        auto_yaw = np.deg2rad(24.0) * np.sin(t * 0.55)
        return auto_yaw + self.view_yaw_offset, self.view_pitch

    def _project_indices(self, indices: np.ndarray, inner):
        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) == 0:
            return (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=bool),
            )

        xyz = self.positions[indices]
        finite = np.isfinite(xyz).all(axis=1)
        xyz = np.where(finite[:, None], xyz, 0.0)

        yaw, pitch = self._view_yaw, self.view_pitch
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        x = xyz[:, 0] * cy - xyz[:, 1] * sy
        depth = xyz[:, 0] * sy + xyz[:, 1] * cy
        z = xyz[:, 2]
        depth2 = depth * cp - z * sp
        z2 = depth * sp + z * cp

        scale = min(
            inner.w * 0.45 / max(self.xy_radius, 1e-6),
            inner.h * 0.45 / max(self.z_radius, 1e-6),
        )
        depth_norm = np.clip(depth2 / max(self.depth_radius, 1e-6), -1.0, 1.0)
        perspective = 1.0 + 0.10 * depth_norm
        px = np.rint(inner.centerx + x * scale * perspective).astype(np.int32)
        py = np.rint(inner.centery + z2 * scale * perspective).astype(np.int32)
        visible = (
            finite
            & (px >= inner.left)
            & (px < inner.right)
            & (py >= inner.top)
            & (py < inner.bottom)
        )
        return px, py, depth_norm.astype(np.float32), visible

    def _draw_points(self, indices, inner, color, radius: int, max_points: int) -> None:
        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) == 0:
            return
        stride = max(1, int(np.ceil(len(indices) / max_points)))
        indices = indices[::stride]
        px, py, _, visible = self._project_indices(indices, inner)
        points = zip(px[visible].tolist(), py[visible].tolist())
        for point in points:
            self.pg.draw.circle(self.screen, color, point, radius)

    def _draw_brain_cloud(self, inner) -> None:
        px, py, depth, visible = self._project_indices(self.background_indices, inner)
        if not np.any(visible):
            return
        x = px[visible] - inner.x
        y = py[visible] - inner.y
        near = ((depth[visible] + 1.0) * 0.5).clip(0.0, 1.0)
        r = (40 + 72 * near).astype(np.uint8)
        g = (47 + 80 * near).astype(np.uint8)
        b = (58 + 92 * near).astype(np.uint8)
        pixels = np.zeros((inner.w, inner.h, 3), dtype=np.uint8)
        np.maximum.at(pixels[:, :, 0], (x, y), r)
        np.maximum.at(pixels[:, :, 1], (x, y), g)
        np.maximum.at(pixels[:, :, 2], (x, y), b)
        self.screen.blit(self.pg.surfarray.make_surface(pixels), inner.topleft)

    def _draw_brain(self, decision, rect) -> None:
        self._text("FLY BRAIN", rect.x + 16, rect.y + 14, self.big)
        subtitle = "MaleCNS anatomical 3D soma cloud · XYZ" if self.anatomical else "3D position fallback"
        self._text(subtitle, rect.x + 17, rect.y + 45, self.small, (139, 151, 170))
        inner = self.pg.Rect(rect.x + 12, rect.y + 70, rect.w - 24, rect.h - 182)

        self._view_yaw, _ = self._current_view()
        self._draw_brain_cloud(inner)

        spikes = np.asarray(decision.brain.spikes if decision else [], dtype=np.int64)
        if decision:
            self.spike_history.append(spikes.copy())
            history = list(self.spike_history)
            for age, old_spikes in enumerate(history):
                intensity = 90 + int(145 * (age + 1) / len(history))
                radius = 1 if age < len(history) - 1 else 2
                self._draw_points(old_spikes, inner, (intensity, intensity, intensity), radius, 300)

            self._draw_points(
                decision.vision.neuron_indices,
                inner,
                (88, 183, 255),
                3,
                180,
            )

            if len(self.output_indices):
                fired_mask = np.isin(self.output_indices, spikes, assume_unique=False)
                self._draw_points(
                    self.output_indices[~fired_mask],
                    inner,
                    (150, 105, 54),
                    2,
                    max(1, len(self.output_indices)),
                )
                self._draw_points(
                    self.output_indices[fired_mask],
                    inner,
                    (255, 190, 92),
                    4,
                    max(1, len(self.output_indices)),
                )

        self._text(
            "auto-orbit · drag mouse to rotate · R reset",
            rect.x + 16,
            rect.bottom - 122,
            self.small,
            (112, 123, 142),
        )
        y = rect.bottom - 100
        if decision:
            self._text(f"active neurons  {decision.brain.active_count:,}", rect.x + 16, y, self.small)
            y += 18
            self._text(
                f"visual drive    {len(decision.vision.neuron_indices):,}",
                rect.x + 16,
                y,
                self.small,
            )
            y += 18
            self._text(
                f"DN L/R          {decision.left_activity:.3f} / {decision.right_activity:.3f}",
                rect.x + 16,
                y,
                self.small,
            )
            y += 18
            self._text(f"action          {ACTION_NAMES[decision.action]}", rect.x + 16, y, self.small)
        self._text(
            "VISION  →  BRAIN  →  ACTION",
            rect.x + 16,
            rect.bottom - 24,
            self.small,
            (170, 180, 196),
        )

    def _draw_timeline(self, decision, rect, fps) -> None:
        action = 0 if decision is None else decision.action
        self.timeline.append(action)
        self._text(
            "EXPERIMENT 001 · MaleCNS v1.0 · fixed wiring · no training",
            rect.x + 14,
            rect.y + 10,
            self.small,
        )
        if decision:
            stats = (
                f"166,700 neurons   step {decision.brain.step_index:,}   "
                f"{1000 / max(decision.brain.latency_ms, 1e-6):.0f} brain steps/s   "
                f"{fps:.0f} sim steps/s   latency {decision.brain.latency_ms:.2f} ms"
            )
            self._text(stats, rect.x + 14, rect.y + 34, self.small)
        x0 = rect.x + 14
        base = rect.bottom - 22
        for j, a in enumerate(self.timeline):
            x = x0 + j * 4
            if x >= rect.right - 8:
                break
            self.pg.draw.line(
                self.screen,
                (150, 157, 170),
                (x, base),
                (x, base - int(a) * 12),
                2,
            )
