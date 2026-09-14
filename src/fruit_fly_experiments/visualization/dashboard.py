from __future__ import annotations
from collections import deque
import numpy as np

from fruit_fly_experiments.controllers.fly import STEERING_TYPES
from fruit_fly_experiments.games.pong import ACTION_NAMES, PongEnv


class Dashboard:
    def __init__(self, simulator, width: int = 1280, height: int = 720, sample_neurons: int = 60000, seed: int = 7) -> None:
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
        self.positions = neurons.positions_xyz()
        valid = np.flatnonzero(np.isfinite(self.positions).all(axis=1))
        self.anatomical = len(valid) >= 100
        self.background_indices = (
            np.sort(self.rng.choice(valid, size=min(sample_neurons, len(valid)), replace=False))
            if self.anatomical
            else np.arange(min(sample_neurons, simulator.n), dtype=np.int64)
        )

        if self.anatomical:
            # MaleCNS X/Z gives a recognizable frontal CNS projection: optic lobes,
            # central brain and the descending VNC. Coordinates are real EM-frame
            # soma/soma-tract positions; only the 2-D projection/orientation is visual.
            self.projected = self.positions[:, [0, 2]].astype(np.float32, copy=True)
            self._orient_anatomy(neurons)
            shown = self.projected[valid]
            lo = np.nanpercentile(shown, 0.5, axis=0)
            hi = np.nanpercentile(shown, 99.5, axis=0)
            self.bounds = (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]))
        else:
            # Synthetic fallback exists only for tiny test/custom graphs that do not
            # carry MaleCNS positions. Production MaleCNS runs should never use it.
            sides = self.rng.choice([-1.0, 1.0], simulator.n)
            self.projected = np.column_stack([
                sides * (0.6 + 0.5 * self.rng.random(simulator.n)) + self.rng.normal(0, 0.14, simulator.n),
                self.rng.normal(0, 1.0, simulator.n),
            ]).astype(np.float32)
            self.bounds = (-1.35, 1.35, -2.4, 2.4)

        output = set(neurons.indices_for_types(STEERING_TYPES, side="L").tolist())
        output.update(neurons.indices_for_types(STEERING_TYPES, side="R").tolist())
        self.output_indices = output
        self.timeline = deque(maxlen=260)
        self.spike_history = deque(maxlen=6)
        self._brain_background = None
        self._brain_background_size = None

    def _orient_anatomy(self, neurons) -> None:
        ann = neurons.annotations
        xz = self.projected

        # Put the fly's anatomical right on the viewer's right when side metadata
        # makes the orientation unambiguous.
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
            lx = np.nanmedian(xz[left, 0]); rx = np.nanmedian(xz[right, 0])
            if np.isfinite(lx) and np.isfinite(rx) and rx < lx:
                xz[:, 0] *= -1.0

        # Put the visual/optic end at the top of the panel. This changes only screen
        # orientation, never any simulation coordinate or connectivity.
        visual = neurons.indices_for_types(["L1", "R7", "R8", "LC10a"])
        visual = visual[np.isfinite(xz[visual, 1])] if len(visual) else visual
        all_z = xz[np.isfinite(xz[:, 1]), 1]
        if len(visual) and len(all_z):
            if np.nanmedian(xz[visual, 1]) > np.nanmedian(all_z):
                xz[:, 1] *= -1.0

    def pump(self) -> bool:
        for event in self.pg.event.get():
            if event.type == self.pg.QUIT or (event.type == self.pg.KEYDOWN and event.key == self.pg.K_ESCAPE):
                return False
        return True

    def draw(self, env: PongEnv, decision, fps: float) -> None:
        pg = self.pg
        self.screen.fill((12, 14, 18))
        margin = 22; bottom_h = 120; gap = 18
        left_w = int(self.w * 0.67)
        game_rect = pg.Rect(margin, margin, left_w - margin, self.h - bottom_h - margin*2)
        brain_rect = pg.Rect(left_w + gap, margin, self.w - left_w - gap - margin, self.h - bottom_h - margin*2)
        timeline_rect = pg.Rect(margin, self.h-bottom_h, self.w-2*margin, bottom_h-margin)
        self._panel(game_rect); self._panel(brain_rect); self._panel(timeline_rect)
        self._draw_pong(env, game_rect)
        self._draw_brain(decision, brain_rect)
        self._draw_timeline(decision, timeline_rect, fps, env)
        pg.display.flip()

    def _panel(self, rect):
        self.pg.draw.rect(self.screen, (20,23,29), rect, border_radius=5)
        self.pg.draw.rect(self.screen, (48,53,64), rect, 1, border_radius=5)

    def _text(self, text, x, y, font=None, color=(225,229,238)):
        self.screen.blit((font or self.font).render(str(text), True, color), (x,y))

    def _draw_pong(self, env, rect):
        pg=self.pg; self._text("PONG / FLY VIEW", rect.x+16, rect.y+14, self.big)
        arena = pg.Rect(rect.x+16, rect.y+58, rect.w-32, rect.h-76)
        pg.draw.rect(self.screen,(7,9,12),arena)
        sx, sy = arena.w/env.width, arena.h/env.height
        for yy in range(arena.y+10, arena.bottom-10, 20): pg.draw.rect(self.screen,(45,49,58),(arena.centerx-1,yy,2,10))
        def y(v): return int(arena.y+v*sy)
        pg.draw.rect(self.screen,(235,238,245),(int(arena.x+env.fly_x*sx-3),y(env.fly_y)-int(env.paddle_h*sy/2),6,int(env.paddle_h*sy)))
        pg.draw.rect(self.screen,(135,143,158),(int(arena.x+env.opp_x*sx-3),y(env.opp_y)-int(env.paddle_h*sy/2),6,int(env.paddle_h*sy)))
        pg.draw.circle(self.screen,(244,246,250),(int(arena.x+env.ball_x*sx),y(env.ball_y)),max(3,int(env.ball_r*sy)))
        self._text(f"FLY {env.fly_score}   :   {env.opp_score} OPP", arena.centerx-75, arena.y+8, self.small)

    def _point(self, idx: int, inner):
        x, z = self.projected[int(idx)]
        if not np.isfinite(x) or not np.isfinite(z):
            return None
        xmin, xmax, zmin, zmax = self.bounds
        xrange = max(xmax - xmin, 1e-6); zrange = max(zmax - zmin, 1e-6)
        scale = min(inner.w * 0.92 / xrange, inner.h * 0.92 / zrange)
        px = int(inner.centerx + (x - (xmin+xmax)/2.0) * scale)
        py = int(inner.centery + (z - (zmin+zmax)/2.0) * scale)
        return px, py

    def _build_brain_background(self, inner):
        surface = self.pg.Surface((inner.w, inner.h), self.pg.SRCALPHA)
        local = self.pg.Rect(0, 0, inner.w, inner.h)
        for i in self.background_indices:
            point = self._point(int(i), local)
            if point is None:
                continue
            x, y = point
            if 0 <= x < inner.w and 0 <= y < inner.h:
                surface.set_at((x, y), (86, 96, 112, 150))
        self._brain_background = surface
        self._brain_background_size = (inner.w, inner.h)

    def _draw_brain(self, decision, rect):
        self._text("FLY BRAIN", rect.x+16, rect.y+14, self.big)
        subtitle = "MaleCNS anatomical soma projection · X/Z" if self.anatomical else "position fallback"
        self._text(subtitle, rect.x+17, rect.y+45, self.small, (139,151,170))
        inner = self.pg.Rect(rect.x+12, rect.y+70, rect.w-24, rect.h-182)
        if self._brain_background is None or self._brain_background_size != (inner.w, inner.h):
            self._build_brain_background(inner)
        self.screen.blit(self._brain_background, inner.topleft)

        spikes = np.asarray(decision.brain.spikes if decision else [], dtype=np.int64)
        if decision:
            self.spike_history.append(spikes.copy())

            # Short activity persistence makes propagation legible without inventing
            # any extra neural activity. Older real spikes are simply drawn dimmer.
            history = list(self.spike_history)
            for age, old_spikes in enumerate(history):
                if len(old_spikes) == 0:
                    continue
                stride = max(1, len(old_spikes) // 900)
                intensity = 90 + int(130 * (age + 1) / len(history))
                radius = 1 if age < len(history)-1 else 2
                for i in old_spikes[::stride]:
                    point = self._point(int(i), inner)
                    if point is not None:
                        self.pg.draw.circle(self.screen, (intensity,intensity,intensity), point, radius)

            vision = np.asarray(decision.vision.neuron_indices, dtype=np.int64)
            stride = max(1, len(vision) // 350) if len(vision) else 1
            for i in vision[::stride]:
                point = self._point(int(i), inner)
                if point is not None:
                    self.pg.draw.circle(self.screen,(88,183,255),point,3)

            for i in self.output_indices:
                point = self._point(int(i), inner)
                if point is None:
                    continue
                firing = bool(np.any(spikes == int(i)))
                self.pg.draw.circle(self.screen,(255,190,92) if firing else (150,105,54),point,4 if firing else 2)

        y=rect.bottom-100
        if decision:
            self._text(f"active neurons  {decision.brain.active_count:,}",rect.x+16,y,self.small); y+=18
            self._text(f"visual drive    {len(decision.vision.neuron_indices):,}",rect.x+16,y,self.small); y+=18
            self._text(f"DN L/R          {decision.left_activity:.3f} / {decision.right_activity:.3f}",rect.x+16,y,self.small); y+=18
            self._text(f"action          {ACTION_NAMES[decision.action]}",rect.x+16,y,self.small)
        self._text("VISION  →  BRAIN  →  ACTION",rect.x+16,rect.bottom-24,self.small,(170,180,196))

    def _draw_timeline(self, decision, rect, fps, env):
        action=0 if decision is None else decision.action
        self.timeline.append(action)
        self._text("EXPERIMENT 001 · MaleCNS v1.0 · fixed wiring · no training",rect.x+14,rect.y+10,self.small)
        if decision:
            stats=f"166,700 neurons   step {decision.brain.step_index:,}   {1000/max(decision.brain.latency_ms,1e-6):.0f} brain steps/s   {fps:.0f} FPS   latency {decision.brain.latency_ms:.2f} ms"
            self._text(stats,rect.x+14,rect.y+34,self.small)
        x0=rect.x+14; base=rect.bottom-22
        for j,a in enumerate(self.timeline):
            x=x0+j*4
            if x >= rect.right-8:
                break
            self.pg.draw.line(self.screen,(150,157,170),(x,base),(x,base-int(a)*12),2)
