from __future__ import annotations
import argparse
from pathlib import Path
from fruit_fly_experiments.brain.loader import build_official_connectome, fetch_official_data

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--build", action="store_true", help="also build the retained sparse connectome")
    args=p.parse_args()
    paths=fetch_official_data(args.data_root)
    for key,path in paths.items(): print(f"{key:18} {path}")
    if args.build:
        graph=build_official_connectome(args.data_root, strict=True)
        print(f"built {graph.neurons.size:,} neurons / {graph.edge_count:,} directed edges / {graph.synaptic_contacts:,} contacts")
if __name__ == "__main__": main()
