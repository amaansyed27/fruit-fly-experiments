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

`--controller random` is the chance baseline. No learning or plasticity is enabled.

## Metrics

Score, rally duration, action distribution, output activity, brain-step latency, FPS, and run-level hit/score proxies. Use multiple seeds for comparisons.

## Results

### Seed 1 paired comparison

Both controllers were evaluated for 30 s / 1500 simulation steps on Pong seed 1.

| Controller | Fly score | Opponent score | UP | NEUTRAL | DOWN |
|---|---:|---:|---:|---:|---:|
| MaleCNS fly controller | 0 | 3 | 191 | 1189 | 120 |
| Random controller | 0 | 7 | 480 | 503 | 517 |

The MaleCNS controller conceded **3 points vs 7 for random** on this seed, but both scored zero. This is only one paired seed and is **not enough to claim above-random performance**. A multi-seed comparison is required before interpreting the result.

The fly-controller run had mean brain-step latency **2.841 ms** on CUDA. Run logs are intentionally gitignored; analyze locally with `python scripts/analyze_pong.py <run.csv>`.

### Local performance benchmark

Measured on Amaan's Windows laptop with an NVIDIA GeForce RTX 5060 Laptop GPU using the full retained MaleCNS graph (**166,700 neurons, 25,582,938 edges**):

| Device | Median brain step | Mean brain step | Steps/s |
|---|---:|---:|---:|
| CPU | 20.675 ms | 20.823 ms | 48.0 |
| RTX 5060 Laptop GPU (CUDA 13.0) | 1.735 ms | 1.761 ms | 568.0 |

The CUDA path is about **11.8× faster** by measured steps/s. PyTorch currently reports sparse CSR support as beta; the warning is informational and the benchmark completed successfully.

## Limitations

The wiring, neuron IDs, annotations, soma/soma-tract coordinates and synapse-count-derived strengths are biological data. Point-neuron dynamics, neurotransmitter sign simplification, pixel-to-neuron encoding, 90° control-axis rotation, LC10a target salience, relative DN motor decoding, and the Pong mapping of steering laterality to vertical paddle movement are engineered assumptions. This is not a biologically exact digital fly.
