#!/usr/bin/env python3
"""Saturation ceiling per tier: the confidence it reaches with unlimited queries.

    python scripts/r2_ceiling.py --artefact data/dialogue_measured.npz

R1 reply 4 §2 asks the decision-relevant question directly: an irreversible
`place()` needs roughly a 0.85 posterior on the right referent, so a tier is
*usable* only if repeated querying can get there. Single-query accuracy alone
does not answer that, because evidence saturates -- and gamma alone does not
either, because a near-random channel has a high gamma and tells you nothing.
The ceiling combines them.

METHOD
------
Monte Carlo through R1's own belief filter (`cogniwatt.env.belief.Belief`) and
R1's own likelihood construction (`DialogueOracle.likelihood`), so this cannot
drift from what the environment actually does. Nothing in `env/` is modified --
it is imported and driven, which is the same thing training does.

Per trial: draw a true referent, then query one tier repeatedly. The n-th query
carries weight `gamma**(n-1)`, exactly as `evidence_weight` specifies, so the
posterior converges and the limit is the ceiling. Reported as the mean
posterior mass on the true referent, which is the quantity a `place()` decision
is actually gated on.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogniwatt.env.belief import BeliefState
from cogniwatt.interfaces.dialogue import load_precomputed

#: An irreversible place() at +10/-10 is a bad bet below this posterior.
THRESHOLD = 0.85


def ceiling(oracle, tier: str, k: int, n_queries: int = 40,
            trials: int = 4000, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    L = oracle.likelihood(tier, k)
    gamma = oracle.evidence_decay[tier]

    post_true = np.zeros(trials)
    first = np.zeros(trials)
    for t in range(trials):
        true_idx = int(rng.integers(0, k))
        b = BeliefState(k, 4)  # k_max=4, the env's maximum candidate count
        for n in range(n_queries):
            obs = int(rng.choice(k, p=L[true_idx]))
            b.update(obs, L, weight=gamma ** n)
            if n == 0:
                first[t] = b.probs[true_idx]
        post_true[t] = b.probs[true_idx]

    # Total weight a tier can ever accumulate: sum_n gamma^(n-1) = 1/(1-gamma).
    eff = 1.0 / (1.0 - gamma) if gamma < 1 else float("inf")
    return {"one_query_posterior": float(first.mean()),
            "ceiling": float(post_true.mean()),
            "reaches_threshold": bool(post_true.mean() >= THRESHOLD),
            "share_of_trials_above": float((post_true >= THRESHOLD).mean()),
            "effective_draws": float(eff)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artefact", default=os.path.join("data", "dialogue_measured.npz"))
    ap.add_argument("--tiers", nargs="+",
                    default=["think_tiny", "think_big", "offload"])
    ap.add_argument("--ks", nargs="+", type=int, default=[2, 3, 4])
    ap.add_argument("--queries", type=int, default=40)
    ap.add_argument("--trials", type=int, default=4000)
    ap.add_argument("--out", default=os.path.join("data", "r2_ceilings.json"))
    args = ap.parse_args()

    o = load_precomputed(args.artefact, strict=True)
    print(f"threshold for an irreversible place(): {THRESHOLD}\n")
    print(f"{'tier':<12}{'k':>2}{'acc':>8}{'gamma':>8}{'eff draws':>11}"
          f"{'1-query':>9}{'ceiling':>9}   reaches?")
    out: Dict[str, Dict[str, Dict]] = {}
    for tier in args.tiers:
        out[tier] = {}
        for k in args.ks:
            r = ceiling(o, tier, k, args.queries, args.trials)
            out[tier][str(k)] = r
            acc = o.accuracy(tier, k)
            print(f"{tier:<12}{k:>2}{acc:>8.3f}{o.evidence_decay[tier]:>8.3f}"
                  f"{r['effective_draws']:>11.2f}{r['one_query_posterior']:>9.3f}"
                  f"{r['ceiling']:>9.3f}   "
                  f"{'YES' if r['reaches_threshold'] else 'no'}")

    usable = [t for t in args.tiers
              if any(out[t][str(k)]["reaches_threshold"] for k in args.ks)]
    print(f"\ntiers that can ever support a confident place(): {usable or 'NONE'}")
    if not usable:
        print("  -> no available model grounds these expressions well enough to"
              "\n     act on, at any candidate count, with unlimited queries.")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"threshold": THRESHOLD, "queries": args.queries,
                   "trials": args.trials, "by_tier": out}, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
