import numpy as np
from fruit_fly_experiments.games.pong import PongEnv, UP
from fruit_fly_experiments.controllers.random import RandomController

def test_controller_receives_only_pixels():
    env=PongEnv(seed=2); frame=env.render_frame(sensor=True); assert isinstance(frame,np.ndarray) and frame.ndim==2
    ctl=RandomController(2); action=ctl.act(frame); env.step(action,.02)

def test_seeded_environment_reproducible():
    a=PongEnv(seed=3); b=PongEnv(seed=3)
    for _ in range(40):
        sa=a.step(UP,.02); sb=b.step(UP,.02); assert sa==sb

def test_headless_random_run_without_brain_data(tmp_path, monkeypatch):
    from fruit_fly_experiments.experiments.pong import run_experiment
    monkeypatch.chdir(tmp_path)
    result=run_experiment("random",.2,1,"cpu",False,False)
    assert result["steps"]==10

def test_headless_fly_experiment_with_tiny_graph(tmp_path, monkeypatch):
    import pandas as pd
    from scipy import sparse
    from fruit_fly_experiments.brain.loader import ConnectomeGraph, FILES
    from fruit_fly_experiments.brain.neurons import NeuronIndex
    from fruit_fly_experiments.experiments.pong import run_experiment
    root=tmp_path/"data"; (root/"processed").mkdir(parents=True); (root/"raw").mkdir(parents=True)
    ann=pd.DataFrame({"bodyId":[1,2,3,4],"type":["LC10a","LC10a","DNa02","DNa02"],"superclass":["visual_projection","visual_projection","descending_neuron","descending_neuron"],"somaSide":["L","R","L","R"],"rootSide":["L","R","L","R"]})
    w=sparse.csr_matrix(np.array([[0,0,0,0],[0,0,0,0],[1,0,0,0],[0,1,0,0]],dtype=np.float32))
    meta={"retained_edges":2,"retained_synaptic_contacts":2}
    ConnectomeGraph(NeuronIndex(np.array([1,2,3,4]),ann),w,2,2,np.ones(4),meta).save(root/"processed")
    pd.DataFrame({"column":[],"L1":[],"R7":[],"R8":[]}).to_excel(root/"raw"/FILES["optic"],index=False)
    monkeypatch.chdir(tmp_path)
    result=run_experiment("fly",.1,1,"cpu",False,False,data_root=root)
    assert result["steps"]==5
