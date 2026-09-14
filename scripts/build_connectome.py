from pathlib import Path
from fruit_fly_experiments.brain.loader import build_official_connectome
if __name__ == "__main__":
    g=build_official_connectome(Path("data"), strict=True)
    print(f"{g.neurons.size:,} neurons; {g.edge_count:,} edges; {g.synaptic_contacts:,} contacts")
