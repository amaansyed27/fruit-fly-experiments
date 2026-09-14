import numpy as np, pandas as pd
from fruit_fly_experiments.brain.loader import build_graph_from_frames, select_neurons

def fixture():
    ann=pd.DataFrame({"bodyId":[10,20,30,40],"superclass":["visual_projection","descending_neuron","descending_neuron",None],"type":["LC10a","DNa02","DNa02",None],"somaSide":["L","L","R",None],"rootSide":["L","L","R",None],"status":["Traced","Traced","Traced","Orphan"]})
    nt=pd.DataFrame({"body":[10,20,30],"predicted_nt":["acetylcholine","gaba","acetylcholine"],"predicted_nt_confidence":[.9,.9,.9]})
    batch={"body_pre":np.array([10,10,20,30,40]),"body_post":np.array([20,30,30,20,10]),"weight":np.array([5,4,3,2,99])}
    return ann,nt,iter([batch])

def test_retention_policy():
    ann,_,_=fixture(); out=select_neurons(ann); assert out.bodyId.tolist()==[10,20,30]

def test_graph_preserves_retained_edges_and_sign():
    ann,nt,batches=fixture(); g=build_graph_from_frames(ann,nt,batches)
    assert g.neurons.size==3; assert g.edge_count==4; assert g.synaptic_contacts==14
    # 20 is GABA, so its outgoing 20->30 edge is inhibitory after normalization.
    assert g.weights[2,1] < 0
