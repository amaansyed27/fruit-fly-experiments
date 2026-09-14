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

Controller modes:

- `--controller fly` — fixed MaleCNS simulation with biological visual entry points and biological steering-DN readout.
- `--controller fly-no-vision` — same MaleCNS simulation/readout but with zero visual injection.
- `--controller fly-shuffled-vision` — same rendered-pixel encoder and drive values, but the driven visual-neuron identities are deterministically permuted to random retained neurons. This tests whether the biological visual entry points matter.
- `--controller random` — uniform random UP/NEUTRAL/DOWN.
- `--controller neutral` — always NEUTRAL; tests whether simply staying near the centre is strong.
- `--controller matched-random` — ignores pixels but samples actions using the aggregate MaleCNS action frequencies observed over seeds 1–10.

No learning or plasticity is enabled.

## Metrics

Score, rally duration, action distribution, output activity, brain-step latency, FPS, and run-level hit/score proxies. Use paired seeds for comparisons.

## Results

### 10-seed control comparison

Each controller was evaluated for 30 s / 1500 simulation steps on the same ten Pong seeds. Lower opponent score means the fly-side paddle conceded fewer points.

| Seed | MaleCNS | Uniform random | Always neutral | Matched random | MaleCNS no vision | Shuffled vision |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 7 | 7 | 7 | 7 | 7 |
| 2 | 3 | 4 | 5 | 3 | 5 | 6 |
| 3 | 1 | 5 | 5 | 3 | 5 | 6 |
| 4 | 2 | 7 | 4 | 3 | 4 | 6 |
| 5 | 2 | 4 | 7 | 7 | 7 | 5 |
| 6 | 2 | 7 | 4 | 5 | 4 | 6 |
| 7 | 3 | 4 | 4 | 4 | 4 | 8 |
| 8 | 1 | 5 | 7 | 8 | 7 | 7 |
| 9 | 1 | 4 | 1 | 3 | 1 | 6 |
| 10 | 4 | 8 | 6 | 6 | 6 | 7 |
| **Mean conceded** | **2.2** | **5.5** | **5.0** | **4.9** | **5.0** | **6.4** |

The normal MaleCNS controller conceded fewer points than uniform random on **10/10 seeds**, fewer than always-neutral on **9/10 with 1 tie**, fewer than matched-random on **9/10 with 1 tie**, fewer than no-vision MaleCNS on **9/10 with 1 tie**, and fewer than the first shuffled-vision permutation on **10/10 seeds**.

Relative to the controls, mean points conceded fell by:

- **60.0% vs uniform random** (5.5 → 2.2)
- **56.0% vs always-neutral** (5.0 → 2.2)
- **55.1% vs matched-random** (4.9 → 2.2)
- **56.0% vs MaleCNS no-vision** (5.0 → 2.2)
- **65.6% vs the first shuffled-vision permutation** (6.4 → 2.2)

Exact two-sided sign tests on paired direction give **p = 0.00195** vs uniform random and the first shuffled-vision permutation, and **p = 0.00391** vs always-neutral, matched-random, and no-vision (ties excluded).

The no-vision MaleCNS produced **1500/1500 NEUTRAL actions on every seed**, and its score pattern exactly matched the always-neutral control. Under the current dynamics/readout, removing visual drive therefore removes the movement that produced the MaleCNS advantage. This is evidence that **task visual input is necessary for the observed defensive behavior**.

The first shuffled-vision control used one fixed entry-point permutation (`shuffle_seed=424242`). It performed worse than the biological mapping on all ten Pong seeds, with **6.4 vs 2.2 mean points conceded**. That is consistent with the placement of visual drive in the real connectome mattering. However, this one permutation also produced a strong DOWN action bias, so one shuffled map is not enough for a robust placement claim. Multiple independent entry-point permutations are required next.

The strongest current conclusion is therefore **not** that a fruit-fly brain has learned Pong. Nothing was trained. Under this engineered simulator/interface, pixel-derived visual drive routed into biologically selected MaleCNS visual neurons and propagated through the fixed connectome produces defensive paddle behavior that outperforms multiple non-visual, random, distribution-matched, and first-pass shuffled-entry controls.

All evaluated controllers scored zero against the opponent in these runs, so the evidence concerns **defensive survival / reduced misses**, not successful offensive Pong play.

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
