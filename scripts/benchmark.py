from __future__ import annotations
import argparse, statistics, time
from pathlib import Path
from fruit_fly_experiments.brain.loader import ConnectomeGraph
from fruit_fly_experiments.brain.simulator import BrainSimulator

def main():
    p=argparse.ArgumentParser(); p.add_argument("--device",choices=["cpu","cuda","auto"],default="auto"); p.add_argument("--steps",type=int,default=200); a=p.parse_args()
    g=ConnectomeGraph.load(Path("data/processed")); sim=BrainSimulator(g,device=a.device,seed=1)
    for _ in range(10): sim.step()
    ms=[]
    for _ in range(a.steps): ms.append(sim.step().latency_ms)
    print(f"device={sim.device} neurons={sim.n:,} edges={g.edge_count:,} median_ms={statistics.median(ms):.3f} mean_ms={statistics.mean(ms):.3f} steps_per_s={1000/statistics.mean(ms):.1f}")
if __name__=="__main__": main()
