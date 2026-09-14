from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, sleep

import numpy as np

from fruit_fly_experiments.brain.loader import ConnectomeGraph, FILES, default_data_root
from fruit_fly_experiments.brain.simulator import BrainSimulator
from fruit_fly_experiments.controllers.fly import FlyController
from fruit_fly_experiments.controllers.random import MatchedRandomController, NeutralController, RandomController
from fruit_fly_experiments.games.pong import ACTION_NAMES, PongEnv
from fruit_fly_experiments.vision.encoder import (
    FlyVisualEncoder,
    NoVisionEncoder,
    ShuffledVisionEncoder,
    VisionSubsetEncoder,
)
from .logging import RunLogger


def run_experiment(
    controller_name: str,
    seconds: float,
    seed: int,
    device: str,
    demo: bool,
    log: bool,
    data_root: Path | None = None,
    shuffle_seed: int = 424242,
) -> dict:
    root = data_root or default_data_root()
    env = PongEnv(seed=seed)
    sim = None
    fly_modes = {
        "fly",
        "fly-no-vision",
        "fly-shuffled-vision",
        "fly-retina-only",
        "fly-lc10a-only",
    }
    if controller_name in fly_modes:
        graph = ConnectomeGraph.load(root / "processed")
        sim = BrainSimulator(graph, device=device, seed=seed)
        if controller_name == "fly-no-vision":
            encoder = NoVisionEncoder()
        else:
            optic = root / "raw" / FILES["optic"]
            base_encoder = FlyVisualEncoder.from_data(graph.neurons, optic)
            if controller_name == "fly-shuffled-vision":
                encoder = ShuffledVisionEncoder(base_encoder, graph.neurons.size, shuffle_seed=shuffle_seed)
            elif controller_name == "fly-retina-only":
                encoder = VisionSubsetEncoder(base_encoder, keep_retina=True, keep_lc10a=False)
            elif controller_name == "fly-lc10a-only":
                encoder = VisionSubsetEncoder(base_encoder, keep_retina=False, keep_lc10a=True)
            else:
                encoder = base_encoder
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
        # 30k real soma positions still give a dense anatomical cloud while cutting
        # the per-frame 3-D projection work in half compared with the original demo.
        dashboard = Dashboard(sim, sample_neurons=30000)

    logger = RunLogger(Path("results/experiment_001_pong"), seed, controller_name) if log else None
    dt = 0.020
    steps = max(1, int(seconds / dt))
    started = perf_counter()
    latencies = []
    actions = []
    # Run the simulation at 50 Hz but render the expensive 3-D dashboard at 30 Hz.
    # Headless experiments remain unthrottled.
    render_accumulator = 0.0
    render_hz = 30.0
    demo_closed_early = False
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
                    demo_closed_early = True
                    break

                render_accumulator += render_hz * dt
                if render_accumulator >= 1.0:
                    render_accumulator -= 1.0
                    dashboard.draw(env, decision, fps)

                # Pace demo mode to the actual 20 ms simulation timestep instead of
                # using pygame's old 60 FPS cap, which made a 50 Hz simulation run
                # too fast while simultaneously over-rendering the 3-D brain.
                deadline = started + (step + 1) * dt
                remaining = deadline - perf_counter()
                if remaining > 0:
                    sleep(remaining)
    finally:
        if logger:
            logger.close()
        if dashboard:
            dashboard.pg.quit()

    snap = env.snapshot()
    result = {
        "controller": controller_name,
        "seed": seed,
        "steps": len(actions),
        "fly_score": snap.fly_score,
        "opponent_score": snap.opponent_score,
        "mean_brain_latency_ms": float(np.mean(latencies)) if latencies else None,
        "actions": {name: int(sum(ACTION_NAMES[a] == name for a in actions)) for name in ACTION_NAMES.values()},
        "log": str(logger.path) if logger else None,
    }
    if controller_name == "fly-shuffled-vision":
        result["shuffle_seed"] = int(shuffle_seed)
    if demo_closed_early:
        result["demo_closed_early"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 001: MaleCNS controls a Pong paddle")
    parser.add_argument(
        "--controller",
        choices=[
            "fly",
            "fly-no-vision",
            "fly-shuffled-vision",
            "fly-retina-only",
            "fly-lc10a-only",
            "random",
            "neutral",
            "matched-random",
        ],
        default="fly",
    )
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=424242,
        help="entry-point permutation seed for --controller fly-shuffled-vision",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()
    result = run_experiment(
        args.controller,
        args.seconds,
        args.seed,
        args.device,
        args.demo,
        not args.no_log,
        shuffle_seed=args.shuffle_seed,
    )
    print(result)


if __name__ == "__main__":
    main()
