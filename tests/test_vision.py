import numpy as np, pandas as pd
from fruit_fly_experiments.brain.neurons import NeuronIndex
from fruit_fly_experiments.vision.encoder import FlyVisualEncoder

def test_encoder_outputs_real_graph_indices():
    ann=pd.DataFrame({"bodyId":[101,102,103,201,202],"type":["R7","L1","R8","LC10a","LC10a"],"superclass":["sensory"]*5,"somaSide":["L","L","L","L","R"],"rootSide":["L","L","L","L","R"]})
    neurons=NeuronIndex(np.array([101,102,103,201,202]),ann)
    optics=pd.DataFrame({"column":["ME_L_col_01_01"],"L1":[102],"R7":[101],"R8":[103]})
    enc=FlyVisualEncoder(neurons,optics)
    f0=np.zeros((20,30),dtype=np.float32); enc.encode(f0)
    f1=f0.copy(); f1[5:8,20:23]=1
    out=enc.encode(f1)
    assert np.all((out.neuron_indices>=0)&(out.neuron_indices<5)); assert out.motion_energy>0
