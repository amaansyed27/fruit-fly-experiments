# Fruit Fly Experiments

Small, visual experiments using the **MaleCNS v1.0 adult male Drosophila connectome** as a computational substrate.

## Experiment 001 — Pong

**Status:** V0.1 implemented; fixed wiring, no learning. Full-dataset performance should be verified on the target machine after downloading MaleCNS.

```text
rendered Pong vision
        ↓
MaleCNS visual neurons (optic columns + LC10a)
        ↓
166,700-neuron retained connectome
        ↓
steering descending-neuron activity
        ↓
UP / DOWN / NEUTRAL paddle action
```

The wiring, neuron identities, annotations and synapse-count-derived strengths come from MaleCNS. Neuron dynamics, visual encoding and the Pong motor interface are simulation choices. This is **not** an uploaded or biologically exact fly brain.

### Run

```powershell
cd D:\Programming\03_Projects\personal-projects\04_Miscellaneous\fruit-fly-experiments
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
python scripts\fetch_data.py --build
python -m fruit_fly_experiments.experiments.pong --controller fly --device auto --seconds 30 --demo
```

Baseline and analysis:

```powershell
python -m fruit_fly_experiments.experiments.pong --controller random --seconds 30
python scripts\analyze_pong.py results\experiment_001_pong\runs\<run.csv>
python scripts\benchmark.py --device cpu
python scripts\benchmark.py --device cuda
pytest -q
```

If `torch.cuda.is_available()` is false, install a CUDA-enabled PyTorch build appropriate for the NVIDIA driver, then rerun with `--device cuda`.

MaleCNS data is downloaded from Janelia and kept under `data/` (gitignored). Dataset: CC BY 4.0. Project code is independent and does not vendor third-party simulator code.
