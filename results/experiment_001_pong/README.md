# Experiment 001 — Pong

## Hypothesis

Activity propagated through the fixed MaleCNS v1.0 connectome may produce above-random sensorimotor behavior when pixel-derived visual activity is injected into mapped visual neurons and bilateral steering-descending activity controls a Pong paddle.

## Method

- Retain MaleCNS neurons with a non-empty `superclass`, excluding explicit glia: expected **166,700 neurons**.
- Retain all released directed edges between those neurons: expected **25,582,938 edges / 124,177,617 synaptic contacts**.
- Use synapse count as connection strength; GABA, glutamate and histamine are treated as inhibitory proxies, then each postsynaptic neuron's absolute incoming weights are normalized.
- Render an egocentric Pong sensor image and rotate its control axis into the fly's horizontal visual axis.
- Drive mapped L1/R7/R8 optic-column neurons from luminance/motion. LC10a receives a bright moving-target signal computed only from rendered pixels; no internal `ball_y`, ball velocity, or target coordinate enters the controller.
- Run the full fixed sparse graph with simplified leaky integrate-and-fire dynamics.
- Read bilateral activity from steering-related DNs (`DNa01`, `DNa02`, `DNa03`, `DNa11`, `DNb02`, `DNg13`). A short spike trace is compared by **relative left/right activity**, so sparse DN output is not discarded by an arbitrary large absolute threshold.
- The dashboard renders released MaleCNS `somaLocation`/`tosomaLocation` coordinates as an interactive **3-D XYZ soma cloud**. The coordinates are anatomical data; camera projection, colors, depth shading, orbit and activity persistence are visualization choices.
- **No model or readout is trained.** Connectome weights remain fixed throughout the experiment.

## Controls

Four controller modes are currently evaluated:

- `--controller fly` — fixed MaleCNS simulation and biological readout.
- `--controller random` — uniform random UP/NEUTRAL/DOWN.
- `--controller neutral` — always NEUTRAL; tests whether simply staying near the centre is strong.
- `--controller matched-random` — ignores pixels but samples actions using the aggregate MaleCNS action frequencies observed over seeds 1–10. This tests whether any advantage is explained by action-frequency bias rather than sensorimotor structure.

No learning or plasticity is enabled.

## Metrics

Score, rally duration, action distribution, output activity, brain-step latency, FPS, and run-level hit/score proxies. Use paired seeds for comparisons.

## Results

### 10-seed four-controller comparison

Each controller was evaluated for 30 s / 1500 simulation steps on the same ten Pong seeds. Lower opponent score means the fly-side paddle conceded fewer points.

| Seed | MaleCNS | Uniform random | Always neutral | Matched random |
|---:|---:|---:|---:|---:|
| 1 | 3 | 7 | 7 | 7 |
| 2 | 3 | 4 | 5 | 3 |
| 3 | 1 | 5 | 5 | 3 |
| 4 | 2 | 7 | 4 | 3 |
| 5 | 2 | 4 | 7 | 7 |
| 6 | 2 | 7 | 4 | 5 |
| 7 | 3 | 4 | 4 | 4 |
| 8 | 1 | 5 | 7 | 8 |
| 9 | 1 | 4 | 1 | 3 |
| 10 | 4 | 8 | 6 | 6 |
| **Mean conceded** | **2.2** | **5.5** | **5.0** | **4.9** |

The MaleCNS controller conceded fewer points than uniform random on **10/10 seeds**, fewer than always-neutral on **9/10 with 1 tie**, and fewer than matched-random on **9/10 with 1 tie**.

Relative to the controls, mean points conceded fell by:

- **60.0% vs uniform random** (5.5 → 2.2)
- **56.0% vs always-neutral** (5.0 → 2.2)
- **55.1% vs matched-random** (4.9 → 2.2)

Exact two-sided sign tests on paired direction give **p = 0.00195** vs uniform random and **p = 0.00391** vs both always-neutral and matched-random (ties excluded).

This rules out two simple explanations for the initial result: merely staying near the centre, and merely having a mostly-neutral action-frequency distribution. The timing and direction of MaleCNS-driven actions therefore contain more task-relevant structure than these controls under the current simulator/interface.

However, this still does **not** establish that the advantage is specifically caused by visually guided processing through the connectome. The next required control is to run the same MaleCNS dynamics and motor readout with the task visual input removed. If that no-vision controller performs similarly, the effect could come from intrinsic dynamics rather than visual sensorimotor processing.

All four controllers scored zero against the opponent in these runs, so the current evidence concerns **defensive survival / reduced misses**, not successful offensive Pong play.

### Aggregate fly action counts, seeds 1–10

| Action | Count | Share |
|---|---:|---:|
| UP | 2,487 | 16.58% |
| NEUTRAL | 10,118 | 67.45% |
| DOWN | 2,395 | 15.97% |

Run logs are intentionally gitignored; analyze individual logs locally with `python scripts/analyze_pong.py <run.csv>`.

### Local performance benchmark

Measured on Amaan's Windows laptop with an NVIDIA GeForce RTX 5060 Laptop GPU using the full retained MaleCNS graph (**166,700 neurons, 25,582,938 edges**):

| Device | Median brain step | Mean brain step | Steps/s |
|---|---:|---:|---:|
| CPU | 20.675 ms | 20.823 ms | 48.0 |
| RTX 5060 Laptop GPU (CUDA 13.0) | 1.735 ms | 1.761 ms | 568.0 |

The CUDA path is about **11.8× faster** by measured steps/s. PyTorch currently reports sparse CSR support as beta; the warning is informational and the benchmark completed successfully.

## Limitations

The wiring, neuron IDs, annotations, soma/soma-tract coordinates and synapse-count-derived strengths are biological data. Point-neuron dynamics, neurotransmitter sign simplification, pixel-to-neuron encoding, 90° control-axis rotation, LC10a target salience, relative DN motor decoding, and the Pong mapping of steering laterality to vertical paddle movement are engineered assumptions. This is not a biologically exact digital fly.
