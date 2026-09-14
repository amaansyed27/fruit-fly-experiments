import numpy as np, pandas as pd
from scipy import sparse
from fruit_fly_experiments.brain.loader import ConnectomeGraph
from fruit_fly_experiments.brain.neurons import NeuronIndex
from fruit_fly_experiments.brain.simulator import BrainSimulator
from fruit_fly_experiments.vision.encoder import FlyVisualEncoder
from fruit_fly_experiments.controllers.fly import FlyController
from fruit_fly_experiments.games.pong import DOWN, NEUTRAL, UP


def make_controller(deadband=0.18, min_activity=0.001):
    ann=pd.DataFrame({"bodyId":[1,2,3,4],"type":["LC10a","LC10a","DNa02","DNa02"],"superclass":["visual_projection","visual_projection","descending_neuron","descending_neuron"],"somaSide":["L","R","L","R"],"rootSide":["L","R","L","R"]})
    # LC10a L->DN L; LC10a R->DN R
    w=sparse.csr_matrix(np.array([[0,0,0,0],[0,0,0,0],[1,0,0,0],[0,1,0,0]],dtype=np.float32))
    g=ConnectomeGraph(NeuronIndex(np.array([1,2,3,4]),ann),w,2,2,np.ones(4),{})
    sim=BrainSimulator(g,device="cpu",seed=1,tonic=0.0,gain=1.0)
    enc=FlyVisualEncoder(g.neurons,None)
    return FlyController(sim,enc,deadband=deadband,min_activity=min_activity)


def test_tiny_end_to_end_fly_controller():
    ctl=make_controller(deadband=0.0,min_activity=0.0)
    frame=np.zeros((20,40),dtype=np.float32); ctl.act(frame)
    frame[:,30:34]=1.0; action=ctl.act(frame)
    assert action in (-1,0,1); assert ctl.last_decision.brain.step_index==2


def test_relative_decoder_uses_sparse_bilateral_signal():
    ctl=make_controller()
    # The old absolute 0.08 threshold discarded activity at this scale.
    assert ctl._decode(0.000, 0.017) == DOWN
    assert ctl._decode(0.017, 0.000) == UP
    assert ctl._decode(0.017, 0.016) == NEUTRAL
    assert ctl._decode(0.0001, 0.0001) == NEUTRAL
