"""Lab-3 deliverable (R1): baseline suite on HomeGrid-Edge."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
from cogniwatt.env.homegrid import HomeGridEdge, TRAIN_DEVICES, HELDOUT_DEVICES
from cogniwatt.agents.baselines import default_baselines
from cogniwatt.analysis.evaluate import evaluate, summarise, outcome_table, fmt

N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
pd.set_option("display.width", 200)

env_fn = lambda: HomeGridEdge(devices=TRAIN_DEVICES, seed=0)
frames = [evaluate(b, env_fn, n_episodes=N, devices=list(TRAIN_DEVICES)) for b in default_baselines()]
df = pd.concat(frames, ignore_index=True)
df.to_csv("results/baselines.csv", index=False)

print(f"\n=== BASELINES on train devices  (n={N} paired episodes each) ===")
print(fmt(summarise(df)))
print("\n=== outcome distribution ===")
print(outcome_table(df).round(3).to_string())
print("\n=== greedy_value & oracle, per device ===")
sub = df[df.agent.isin(["greedy_value", "oracle", "always_big", "no_think"])]
print(fmt(summarise(sub, by="device")))
