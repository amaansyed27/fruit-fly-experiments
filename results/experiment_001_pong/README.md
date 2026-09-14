# Experiment 001 — Pong

## Hypothesis

Activity propagated through the fixed MaleCNS v1.0 connectome may produce above-random sensorimotor behavior when pixel-derived visual activity is injected into mapped visual neurons and bilateral steering-descending activity controls a Pong paddle.

## Method

- Retain MaleCNS neurons with a non-empty `superclass`, excluding explicit glia: expected **166,700 neurons**.
- Retain all released directed edges between those neurons: expected **25,582,938 edges / 124,177,617 synaptic contacts**.
- Use synapse count as connection strength; GABA, glutamate and histamine are treated as inhibitory proxies, then each postsynaptic neuron's absolute incoming weights are normalized.
- Render an egocentric Pong sensor image and rotate its control axis into the fly's horizontal visual axis.
- Drive mapped L1/R7/R8 optic-column neurons from luminance/motion and LC10a from coarse pixel motion. No internal `ball_y`, ball velocity, or target coordinate enters the controller.
- Run the full fixed sparse graph with simplified leaky integrate-and-fire dynamics.
- Read bilateral activity from steering-related DNs (`DNa01`, `DNa02`, `DNa03`, `DNa11`, `DNb02`, `DNg13`). Right-minus-left activity is translated to DOWN/UP; a deadband yields NEUTRAL.

## Controls

`--controller random` is the V0.1 chance baseline. No learning or plasticity is enabled.

## Metrics

Score, rally duration, action distribution, output activity, brain-step latency, FPS, and run-level hit/score proxies. Use multiple seeds for comparisons.

## Results

No results are committed. Run logs are intentionally gitignored; analyze locally with `python scripts/analyze_pong.py <run.csv>`.

## Limitations

The wiring, neuron IDs, annotations and synapse-count-derived strengths are biological data. Point-neuron dynamics, neurotransmitter sign simplification, frame-to-vision mapping, 90° control-axis rotation, LC10a motion drive, and the Pong mapping of steering laterality to vertical paddle movement are engineered assumptions. This is not a biologically exact digital fly.
