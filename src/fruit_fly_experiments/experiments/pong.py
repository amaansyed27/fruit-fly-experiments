from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from fruit_fly_experiments.brain.loader import ConnectomeGraph, FILES, default_data_root
from fruit_fly_experiments.brain.simulator import BrainSimulator
from fruit_fly_experiments.controllers.fly import FlyController
from fruit_fly_experiments.controllers.random import MatchedRandomController, NeutralController, RandomController
from fruit_fly_experiments.games.pong import ACTION_NAMES, PongEnv
from fruit_fly_experiments.vision.encoder import FlyVisualEncoder, NoVisionEncoder
from .logging import RunLogger


def run_experiment(controller_name: str, seconds: float, seed: int, device: str, demo: bool, log: bool, data_root: Path | None = None) -> dict:
    root = data_root or default_data_root()
    env = PongEnv(seed=seed)
    sim = None
    if controller_name in {"fly", "fly-no-vision"}:
        graph = ConnectomeGraph.load(root / "processed")
        sim = BrainSimulator(graph, device=device, seed=seed)
        if controller_name == "fly":
            optic = root / "raw" / FILES["optic"]
            encoder = FlyVisualEncoder.from_data(graph.neurons, optic)
        else:
            encoder = NoVisionEncoder()
        controller = FlyController(sim, encoder)
    elif controller_name == "random":
        controller = RandomController(seed)
    elif controller_name == "neutral":
        controller = NeutralController()
    elif controller_name == "matched-random":
        controller = MatchedRandomController(seed)
    else:
        raise ValueError(controller_name)

    dashboard = None
    if demo:
        if sim is None:
            raise ValueError("live dashboard requires a MaleCNS controller")
        from fruit_fly_experiments.visualization.dashboard import Dashboard
        dashboard = Dashboard(sim)

    logger = RunLogger(Path("results/experiment_001_pong"), seed, controller_name) if log else None
    dt = 0.020
    steps = max(1, int(seconds / dt))
    started = perf_counter()
    latencies = []
    actions = []
    try:
        for step in range(steps):
            sensor_frame = env.render_frame(sensor=True)
            action = controller.act(sensor_frame)
            snapshot = env.step(action, dt)
            actions.append(action)
            decision = getattr(controller, "last_decision", None)
            brain_latency = decision.brain.latency_ms if decision else 0.0
            if decision:
                latencies.append(brain_latency)
            elapsed = perf_counter() - started
            fps = (step + 1) / max(elapsed, 1e-9)
            if logger:
                logger.write({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "controller": controller_name,
                    "seed": seed,
                    "step": step + 1,
                    "simulated_time_s": (step + 1) * dt,
                    "action": ACTION_NAMES[action],
                    "left_output": decision.left_activity if decision else "",
                    "right_output": decision.right_activity if decision else "",
                    "ball_x": snapshot.ball_x,
                    "ball_y": snapshot.ball_y,
                    "paddle_y": snapshot.fly_paddle_y,
                    "opponent_y": snapshot.opponent_paddle_y,
                    "fly_score": snapshot.fly_score,
                    "opponent_score": snapshot.opponent_score,
                    "rally_steps": snapshot.rally_steps,
                    "fly_hits": snapshot.fly_hits,
                    "fly_misses": snapshot.fly_misses,
                    "brain_latency_ms": brain_latency,
                    "fps": fps,
                    "active_neurons": decision.brain.active_count if decision else "",
                    "vision_motion_energy": decision.vision.motion_energy if decision else "",
                })
            if dashboard:
                if not dashboard.pump():
                    break
                dashboard.draw(env, decision, fps)
                dashboard.clock.tick(60)
    finally:
        if logger:
            logger.close()
        if dashboard:
            dashboard.pg.quit()

    snap = env.snapshot()
    return {
        "controller": controller_name,
        "seed": seed,
        "steps": len(actions),
        "fly_score": snap.fly_score,
        "opponent_score": snap.opponent_score,
        "mean_brain_latency_ms": float(np.mean(latencies)) if latencies else None,
        "actions": {name: int(sum(ACTION_NAMES[a] == name for a in actions)) for name in ACTION_NAMES.values()},
        "log": str(logger.path) if logger else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 001: MaleCNS controls a Pong paddle")
    parser.add_argument(
        "--controller",
        choices=["fly", "fly-no-vision", "random", "neutral", "matched-random"],
        default="fly",
    )
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()
    result = run_experiment(args.controller, args.seconds, args.seed, args.device, args.demo, not args.no_log)
    print(result)


if __name__ == "__main__":
    main()
