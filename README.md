# Fruit Fly Experiments

Small, visual experiments using the **MaleCNS v1.0 adult male Drosophila connectome** as a computational substrate.

## Experiment 001 — Pong

**Status: complete.** Fixed wiring, no learning.

```text
rendered Pong pixels
        ↓
pixel-derived visual signal
        ↓
real MaleCNS visual entry neurons (L1/R7/R8 + LC10a)
        ↓
166,700-neuron / 25.6M-edge connectome simulation
        ↓
real steering-related descending neurons
        ↓
UP / DOWN / NEUTRAL
        ↓
Pong paddle
```

The MaleCNS controller was tested across 10 paired seeds. It conceded **2.2 points on average**, versus **5.5 for uniform random**, **4.9 for matched-random**, and **5.0 with vision removed**. Five shuffled visual-entry mappings averaged **6.06 conceded points**.

Ablation testing showed an important limitation: the useful Pong behavior currently comes from the engineered **LC10a moving-target input**. The mapped L1/R7/R8 retinal pathway had no measurable behavioral contribution in this setup. So this is evidence that activity routed through biologically selected MaleCNS circuitry can produce useful steering — **not** that a simulated fly retina understood Pong.

No weights are trained or changed. This is **computational neuroscience / connectome simulation, not machine learning**.

### Performance

Measured locally on an NVIDIA GeForce RTX 5060 Laptop GPU with the full retained graph:

- CUDA: ~**568 brain steps/s** (~1.76 ms mean step)
- CPU: ~**48 brain steps/s** (~20.8 ms mean step)
- CUDA speedup: ~**11.8×**

### Run

```powershell
cd D:\Programming\03_Projects\personal-projects\04_Miscellaneous\fruit-fly-experiments
.\.venv\Scripts\Activate.ps1
python -m fruit_fly_experiments.experiments.pong --controller fly --device cuda --seconds 30 --demo
```

Full methods, controls, ablations and results are in `results/experiment_001_pong/README.md`.

## Planned experiments

1. **Pong** — fixed connectome sensorimotor control ✅
2. **Endless runner** — richer visual control task
3. **Learning / plasticity** — can performance improve with experience?
4. **Fly School** — sequential tasks without resetting the brain
5. **Tiny grounded language** — can symbols acquire meaning through experience?
6. **Two identical brains, different experiences** — does training history create different behavior?
7. **Drawing** — use connectome output to control a cursor

Each experiment should keep the same rule: clearly separate **measured biological structure** from **engineered simulation assumptions**, include controls, and avoid claims of consciousness or a biologically exact digital fly.

MaleCNS data is downloaded from Janelia and kept under `data/` (gitignored). Dataset: CC BY 4.0.
