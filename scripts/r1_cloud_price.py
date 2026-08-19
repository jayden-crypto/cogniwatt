"""Sensitivity of the channel portfolio to the price of a cloud call. Owner: R1.

Why this exists: after the roofline recalibration, local inference became
genuinely expensive (think_big is 1591 ms / 26 J on the M2 Air) while a cloud
call is 140 ms / 0.8 J. `lambda_cloud = 0.5` was introduced to stop the policy
degenerating to "always offload", and it does stop the *degenerate* case - but
the learned policy still draws only 5-20% of its evidence from the on-device
tiers. Since CogniWatt's claim is that the optimum is a genuine portfolio, the
price at which the local tiers re-enter that portfolio is a first-class
modelling question, not a tuning detail.

`GreedyValueAgent` scores channels from `env.cfg` at decision time, so it
re-optimises for whatever price it is handed. That makes this sweep possible
without retraining anything: it maps the *structure* of the optimal portfolio
against lambda_cloud, and bounds where the learned policy would land.

    python3 scripts/r1_cloud_price.py --prices 0 0.5 1 2 4 8 -n 400
"""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from cogniwatt.agents.baselines import GreedyValueAgent
from cogniwatt.analysis.evaluate import evaluate
from cogniwatt.env.homegrid import HomeGridEdge, RewardConfig, TRAIN_DEVICES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("-n", type=int, default=400)
    ap.add_argument("--out", default="results/cloud_price.csv")
    a = ap.parse_args()

    rows = []
    for lam in a.prices:
        rc = RewardConfig(lambda_cloud=lam)
        env_fn = lambda: HomeGridEdge(devices=TRAIN_DEVICES, reward=rc, seed=0)
        df = evaluate(GreedyValueAgent(threshold=0.85), env_fn,
                      n_episodes=a.n, devices=list(TRAIN_DEVICES))
        q = df[["n_tiny", "n_big", "n_offload", "n_ask"]].mean()
        tot = float(q.sum())
        rows.append({
            "lambda_cloud": lam,
            "success": df.success.mean(),
            "return": df.ret.mean(),
            "energy_J": df.energy_J.mean(),
            "tiny": q.n_tiny, "big": q.n_big,
            "cloud": q.n_offload, "ask": q.n_ask,
            "local_share": (q.n_tiny + q.n_big) / tot if tot else np.nan,
            "cloud_share": q.n_offload / tot if tot else np.nan,
        })
        print(f"  lambda_cloud={lam:<5} success={rows[-1]['success']:.3f} "
              f"local={rows[-1]['local_share']*100:4.0f}%  "
              f"cloud={rows[-1]['cloud_share']*100:4.0f}%", flush=True)

    out = pd.DataFrame(rows)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"\n-> {a.out}")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
