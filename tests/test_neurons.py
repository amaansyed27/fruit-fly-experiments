import numpy as np
import pandas as pd

from fruit_fly_experiments.brain.neurons import NeuronIndex


def test_positions_xyz_prefers_soma_and_falls_back_to_soma_tract():
    ann = pd.DataFrame({
        "bodyId": [1, 2, 3],
        "type": ["L1", "DNa02", "LC10a"],
        "superclass": ["sensory", "descending_neuron", "visual_projection"],
        "somaLocation": [np.array([1, 2, 3]), None, None],
        "tosomaLocation": [np.array([9, 9, 9]), [4, 5, 6], None],
    })
    neurons = NeuronIndex(np.array([1, 2, 3]), ann)
    xyz = neurons.positions_xyz()
    assert np.allclose(xyz[0], [1, 2, 3])
    assert np.allclose(xyz[1], [4, 5, 6])
    assert np.isnan(xyz[2]).all()
