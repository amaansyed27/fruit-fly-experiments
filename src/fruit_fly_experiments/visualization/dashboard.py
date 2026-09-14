from __future__ import annotations
from collections import deque
import numpy as np

from fruit_fly_experiments.controllers.fly import STEERING_TYPES
from fruit_fly_experiments.games.pong import ACTION_NAMES, PongEnv


class Dashboard:
    def __init__(self, simulator, width: int = 1280, height: int = 720, sample_neurons: int = 1800, seed: int = 7) -> None:
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
        rng = np.random.default_rng(seed)
        n = simulator.n
        self.sample = np.sort(rng.choice(n, size=min(sample_neurons, n), replace=False))

        # Stable two-lobed projection for readability. Every screen coordinate is
        # attached to a real retained neuron index, but the projection is not anatomy.
        sides = rng.choice([-1.0, 1.0], n)
        xy = np.column_stack([
            0.5 + sides * (0.18 + 0.16 * rng.random(n)) + rng.normal(0, 0.055, n),
            0.5 + rng.normal(0, 0.24, n),
        ])
        self.xy = np.clip(xy, 0.05, 0.95).astype(np.float32)
        neurons = simulator.graph.neurons
        output = set(neurons.indices_for_types(STEERING_TYPES, side="L").tolist())
        output.update(neurons.indices_for_types(STEERING_TYPES, side="R").tolist())
        self.output_indices = output
        self.timeline = deque(maxlen=260)

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
        u, v = self.xy[int(idx)]
        return int(inner.x+u*inner.w), int(inner.y+v*inner.h)

    def _draw_brain(self, decision, rect):
        self._text("FLY BRAIN", rect.x+16, rect.y+14, self.big)
        inner = self.pg.Rect(rect.x+12, rect.y+58, rect.w-24, rect.h-170)
        spikes = np.asarray(decision.brain.spikes if decision else [], dtype=np.int64)
        sampled_active = set(np.intersect1d(self.sample, spikes, assume_unique=False).tolist())

        # Context cloud: real neuron IDs, projected coordinates.
        for i in self.sample:
            px, py = self._point(int(i), inner)
            active = int(i) in sampled_active
            self.pg.draw.circle(self.screen,(238,241,247) if active else (62,68,80),(px,py),3 if active else 1)

        if decision:
            # Ensure currently stimulated visual neurons and steering DNs remain visible
            # even if they were not selected into the background sample.
            vision = np.asarray(decision.vision.neuron_indices, dtype=np.int64)
            for i in vision[:220]:
                px, py = self._point(int(i), inner)
                self.pg.draw.circle(self.screen,(106,189,255),(px,py),3)
            active_extra = spikes[:350]
            for i in active_extra:
                px, py = self._point(int(i), inner)
                self.pg.draw.circle(self.screen,(244,246,250),(px,py),2)
            for i in self.output_indices:
                px, py = self._point(int(i), inner)
                firing = bool(np.any(spikes == int(i)))
                self.pg.draw.circle(self.screen,(255,190,92) if firing else (142,105,58),(px,py),4 if firing else 2)

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
        self._text("EXPERIMENT 001 · MaleCNS v1.0 · fixed wiring",rect.x+14,rect.y+10,self.small)
        if decision:
            stats=f"166,700 neurons   step {decision.brain.step_index:,}   {1000/max(decision.brain.latency_ms,1e-6):.0f} brain steps/s   {fps:.0f} FPS   latency {decision.brain.latency_ms:.2f} ms"
            self._text(stats,rect.x+14,rect.y+34,self.small)
        x0=rect.x+14; base=rect.bottom-22
        for j,a in enumerate(self.timeline):
            x=x0+j*4
            self.pg.draw.line(self.screen,(150,157,170),(x,base),(x,base-int(a)*12),2)
