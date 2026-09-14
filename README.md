# Fruit Fly Experiments

A series of small experiments using the mapped fruit-fly connectome as a computational substrate.

The goal is to go beyond one-off demos and test **behaviour, learning, memory, transfer, and experience-dependent change** while keeping every experiment visual and easy to understand.

## Plan

We build one experiment at a time, directly on `main`.

| # | Experiment | Main question |
|---|---|---|
| 1 | **Pong** | Can connectome activity drive a simple real-time control task? |
| 2 | **Endless Runner** | Can it handle left/right/jump decisions in a changing environment? |
| 3 | **Learning / Plasticity** | Can reward-modulated synaptic changes improve performance over time? |
| 4 | **Fly School** | Can one persistent brain learn several tasks without being reset? |
| 5 | **Tiny Language** | Can arbitrary symbols become grounded in actions/rewards, then combine in unseen ways? |
| 6 | **Two Childhoods** | Do two identical starting brains diverge after different experiences? |
| 7 | **Drawing** | Can neural output progressively learn to control a cursor and reproduce shapes? |

More experiments can be added as the project develops.

## Presentation

Every experiment should be understandable from a short video.

Typical layout:

```text
┌──────────────────────────────┬───────────────────┐
│                              │ live brain view   │
│       game / environment     │ active neurons    │
│                              │ output / decision │
│                              │ reward / score    │
└──────────────────────────────┴───────────────────┘

        timeline / learning progress
```

Show the full loop clearly:

```text
sensory input → neural activity → output → action → reward → change
```

Useful overlays include score, active neurons, chosen action, reward, trial number, learning iteration, and performance over time.

## Research Plan

Each experiment should have:

- a clear hypothesis before training
- a fixed-connectome baseline
- a learning/plasticity version where relevant
- repeat runs with multiple seeds
- simple measurable outcomes such as score, survival time, error rate, learning speed, retention, and transfer
- ablations where possible to test whether behaviour really depends on the intended neural pathway or learning rule

Longer-term research directions:

- continual learning and catastrophic forgetting
- transfer between unrelated tasks
- memory retention
- experience-dependent behavioural divergence
- grounded symbolic learning and compositional generalisation
- comparison with conventional RL agents under the same environment and reward budget

## Scientific Caution

The connectome is a measured wiring map, not a complete digital copy of a living fly brain. Neuron dynamics, sensory encoding, plasticity, reward signalling, and other biological processes are simulated choices.

Claims should therefore stay precise: **simulated connectome behaviour**, not consciousness or a fully uploaded brain.
