from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

def analyze(path: Path) -> dict:
    df=pd.read_csv(path)
    if not len(df):
        return {"rows":0,"fly_score":0,"opponent_score":0,"mean_rally_steps":0.0,"max_rally_steps":0,"paddle_hit_rate":None,"action_distribution":{},"mean_brain_latency_ms":None,"mean_fps":None}
    score_f=int(df.fly_score.max()); score_o=int(df.opponent_score.max())
    score_events=df[["fly_score","opponent_score"]].diff().fillna(0).abs().sum(axis=1)>0
    rallies=[]; start=0
    for i in df.index[score_events]: rallies.append(int(i-start)); start=int(i)+1
    if len(df)>start: rallies.append(len(df)-start)
    hits=int(df.fly_hits.max()) if "fly_hits" in df else 0
    misses=int(df.fly_misses.max()) if "fly_misses" in df else 0
    return {
        "rows":len(df), "fly_score":score_f, "opponent_score":score_o,
        "mean_rally_steps":float(sum(rallies)/len(rallies)) if rallies else 0.0,
        "max_rally_steps":max(rallies,default=0),
        "paddle_hit_rate":float(hits/(hits+misses)) if hits+misses else None,
        "action_distribution":df.action.value_counts(normalize=True).to_dict(),
        "mean_brain_latency_ms":float(df.brain_latency_ms.mean()) if "brain_latency_ms" in df else None,
        "mean_fps":float(df.fps.mean()) if "fps" in df else None,
    }

def main():
    p=argparse.ArgumentParser(); p.add_argument("run",type=Path); p.add_argument("--out",type=Path)
    a=p.parse_args(); result=analyze(a.run); print(json.dumps(result,indent=2))
    if a.out: a.out.write_text(json.dumps(result,indent=2),encoding="utf-8")
if __name__=="__main__": main()
