import numpy as np, pandas as pd
from scipy import sparse
from fruit_fly_experiments.brain.loader import ConnectomeGraph
from fruit_fly_experiments.brain.neurons import NeuronIndex
from fruit_fly_experiments.brain.simulator import BrainSimulator, resolve_device

def graph():
    ann=pd.DataFrame({"bodyId":[1,2],"type":["LC10a","DNa02"],"superclass":["visual_projection","descending_neuron"],"somaSide":["L","L"],"rootSide":["L","L"]})
    w=sparse.csr_matrix(np.array([[0,0],[1,0]],dtype=np.float32))
    return ConnectomeGraph(NeuronIndex(np.array([1,2]),ann),w,1,1,np.ones(2),{})

def test_cpu_seeded_reproducible_and_finite():
    a=BrainSimulator(graph(),device="cpu",seed=5,noise_std=.01); b=BrainSimulator(graph(),device="cpu",seed=5,noise_std=.01)
    for _ in range(20):
        ra=a.step(np.array([0]),.4); rb=b.step(np.array([0]),.4)
        assert np.array_equal(ra.spikes,rb.spikes); assert np.isfinite(a.voltage).all()

def test_device_auto_cpu_or_cuda(): assert resolve_device("auto") in {"cpu","cuda"}
