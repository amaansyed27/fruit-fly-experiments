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
- `--controller fly-shuffled-vision` — same rendered-pixel encoder and drive values, but visual-neuron identities are deterministically permuted to random retained neurons.
- `--controller fly-retina-only` — retain only mapped L1/R7/R8 retinal-column drive.
- `--controller fly-lc10a-only` — retain only the LC10a moving-target drive.
- `--controller random` — uniform random UP/NEUTRAL/DOWN.
- `--controller neutral` — always NEUTRAL.
- `--controller matched-random` — ignores pixels but samples actions using the aggregate MaleCNS action frequencies observed over seeds 1–10.

No learning or plasticity is enabled.

## Metrics

Score, rally duration, action distribution, output activity, brain-step latency, FPS, and run-level hit/score proxies. Use paired seeds for comparisons.

## Results

### 10-seed baseline and ablation comparison

Each controller was evaluated for 30 s / 1500 simulation steps on the same ten Pong seeds. Lower opponent score means the fly-side paddle conceded fewer points.

| Seed | MaleCNS | Uniform random | Always neutral | Matched random | MaleCNS no vision |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 7 | 7 | 7 | 7 |
| 2 | 3 | 4 | 5 | 3 | 5 |
| 3 | 1 | 5 | 5 | 3 | 5 |
| 4 | 2 | 7 | 4 | 3 | 4 |
| 5 | 2 | 4 | 7 | 7 | 7 |
| 6 | 2 | 7 | 4 | 5 | 4 |
| 7 | 3 | 4 | 4 | 4 | 4 |
| 8 | 1 | 5 | 7 | 8 | 7 |
| 9 | 1 | 4 | 1 | 3 | 1 |
| 10 | 4 | 8 | 6 | 6 | 6 |
| **Mean conceded** | **2.2** | **5.5** | **5.0** | **4.9** | **5.0** |

The MaleCNS controller conceded fewer points than uniform random on **10/10 seeds**, fewer than always-neutral on **9/10 with 1 tie**, fewer than matched-random on **9/10 with 1 tie**, and fewer than the same MaleCNS simulation with visual input removed on **9/10 with 1 tie**.

Relative to the controls, mean points conceded fell by **60.0% vs uniform random**, **56.0% vs always-neutral**, **55.1% vs matched-random**, and **56.0% vs no-vision**. Exact two-sided sign tests on paired direction give **p = 0.00195** vs uniform random and **p = 0.00391** vs always-neutral, matched-random, and no-vision (ties excluded).

The no-vision MaleCNS produced **1500/1500 NEUTRAL actions on every seed**, exactly matching the always-neutral score pattern. Under the current dynamics/readout, task visual input is therefore necessary for the observed movement advantage.

### Replicated shuffled visual-entry control

The biological visual mapping was compared with **five independent random entry-point permutations**. Each permutation preserved the same rendered pixels, drive amplitudes, number of stimulated neurons, timing, full connectome and DN motor readout; only the connectome entry neurons changed.

| Entry mapping | Mean points conceded | Biological MaleCNS better on paired Pong seeds |
|---|---:|---:|
| **Biological mapping** | **2.2** | — |
| Shuffle 101 | 6.0 | 10/10 |
| Shuffle 202 | 6.1 | 10/10 |
| Shuffle 303 | 5.9 | 10/10 |
| Shuffle 404 | 5.9 | 10/10 |
| Shuffle 424242 | 6.4 | 10/10 |
| **Mean across shuffled mappings** | **6.06** | **50/50 individual paired comparisons** |

For every one of the five shuffled mappings, the biological visual mapping conceded fewer points on **all 10/10 paired Pong seeds**. Each individual 10-seed comparison therefore has an exact two-sided sign-test value of **p = 0.00195**. The shuffled mappings averaged **6.06 conceded points**, versus **2.2** for the biological mapping, a **63.7% reduction** for the biological entry points.

The 50 shuffled comparisons are not treated as 50 independent statistical observations because the same ten Pong seeds are reused across mappings. This is a reproducibility check across entry-point permutations, not a 50-trial independent test.

### Retina vs LC10a component ablation

The full visual controller was split into its two input components and rerun over the same ten Pong seeds.

| Seed | Full vision | Retina only (L1/R7/R8) | LC10a only |
|---:|---:|---:|---:|
| 1 | 3 | 7 | 3 |
| 2 | 3 | 5 | 3 |
| 3 | 1 | 5 | 1 |
| 4 | 2 | 4 | 2 |
| 5 | 2 | 7 | 2 |
| 6 | 2 | 4 | 2 |
| 7 | 3 | 4 | 3 |
| 8 | 1 | 7 | 1 |
| 9 | 1 | 1 | 1 |
| 10 | 4 | 6 | 4 |
| **Mean conceded** | **2.2** | **5.0** | **2.2** |

The **retina-only controller produced 1500/1500 NEUTRAL actions on every seed** and exactly matched the no-vision / always-neutral score pattern. By contrast, the **LC10a-only controller exactly reproduced the full controller's score and action counts on all ten seeds**.

Therefore, under the current simulator and interface, the mapped L1/R7/R8 retinal-column drive has **no measurable causal contribution** to the observed Pong behavior. The entire measured advantage comes from the engineered LC10a moving-target branch entering the biological LC10a neurons and then propagating through the fixed connectome to the DN readout.

This substantially narrows the interpretation. The result is **not evidence that the simulated fly retina or a rich fly visual system is solving Pong**. It is evidence that a pixel-derived target-direction signal, when routed through biologically selected LC10a entry neurons in the fixed MaleCNS network, produces defensive steering behavior that is lost when visual input is removed or when the same visual drive is sent to shuffled entry neurons.

One shuffled run (shuffle 101, Pong seed 9) scored a point against the opponent while the biological controller scored zero across the original ten runs. The current biological result is therefore specifically about **reduced misses / defensive survival**, not superior offensive scoring.

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

The wiring, neuron IDs, annotations, soma/soma-tract coordinates and synapse-count-derived strengths are biological data. Point-neuron dynamics, neurotransmitter sign simplification, pixel-to-neuron encoding, 90° control-axis rotation, LC10a target salience, relative DN motor decoding, and the Pong mapping of steering laterality to vertical paddle movement are engineered assumptions. The LC10a ablation shows that current behavior is entirely driven by the engineered LC10a target signal; the retinal-column pathway is inactive at the behavioral level in this setup. This is not a biologically exact digital fly.