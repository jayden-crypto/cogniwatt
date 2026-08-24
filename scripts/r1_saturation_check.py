"""Does "a cheap model cannot be chained up" survive R2's measured constants?

Owner: R1.

The project's central economic tension (README, docs/lab1_mdp_formalization.md,
interfaces/dialogue.py) is that evidence from a tier *saturates*, so a weak
model cannot be queried repeatedly up to the competence of a strong one. If it
could, the optimal policy would degenerate to "spam the cheapest channel" and
the portfolio framing collapses.

R2 measured the two local tiers and the ordering came out **backwards** from the
R1 placeholders:

    placeholder   think_tiny  acc 0.45   gamma 0.45   ->  1.82 effective draws
                  think_big   acc 0.72   gamma 0.55   ->  2.22
    measured      think_tiny  acc 0.414  gamma 0.834  ->  6.01 effective draws
                  think_big   acc 0.568  gamma 0.605  ->  2.53

The weak model is now both *less accurate* and *far less saturating*. R2's
reading (handback section 3.1) is that gamma measures variability rather than
information, and that the env composes accuracy and gamma correctly so nothing
needs to change. This checks that claim quantitatively: it runs the actual
belief filter and asks what confidence each tier can *ever* reach.

    python3 scripts/r1_saturation_check.py
"""
from __future__ import annotations

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from cogniwatt.env.belief import BeliefState
from cogniwatt.interfaces.dialogue import DialogueOracle
from cogniwatt.interfaces.ved import ASK, OFFLOAD, THINK_BIG, THINK_TINY

K_MAX = 4
N_QUERIES = 40
N_TRIALS = 4000

PLACEHOLDER = dict(
    acc={THINK_TINY: 0.45, THINK_BIG: 0.72, OFFLOAD: 0.88, ASK: 0.97},
    gam={THINK_TINY: 0.45, THINK_BIG: 0.55, OFFLOAD: 0.35, ASK: 0.15},
    by_k={})

# docs/R1_HANDBACK.md section 2. offload/ask remain R1 placeholders -- R2 has not
# measured them, which is section 3.2's warning and is flagged in every table below.
MEASURED = dict(
    acc={THINK_TINY: 0.414, THINK_BIG: 0.568, OFFLOAD: 0.88, ASK: 0.97},
    gam={THINK_TINY: 0.834, THINK_BIG: 0.605, OFFLOAD: 0.35, ASK: 0.15},
    by_k={THINK_TINY: {2: 0.529, 3: 0.396, 4: 0.317},
          THINK_BIG:  {2: 0.671, 3: 0.542, 4: 0.492}})


def ceiling(oracle: DialogueOracle, tier: str, k: int,
            n_queries: int = N_QUERIES, trials: int = N_TRIALS,
            seed: int = 0) -> dict:
    """Monte-Carlo the belief filter: query one tier repeatedly, forever.

    Returns the posterior probability the filter ends up placing on the TRUE
    referent, and how often the argmax is actually right -- which is what the
    agent's delivery decision depends on.
    """
    rng = np.random.default_rng(seed)
    lik = oracle.likelihood(tier, k)
    p_true, correct = np.empty(trials), np.empty(trials)
    for t in range(trials):
        true_idx = int(rng.integers(k))
        b = BeliefState(k, K_MAX)
        for n in range(n_queries):
            w = oracle.evidence_weight(tier, n)
            obs = oracle.sample_evidence(tier, true_idx, k, rng)
            b.update(obs, lik, weight=w)
        p_true[t] = b.probs[true_idx]
        correct[t] = float(b.argmax == true_idx)
    return {"p_true": p_true.mean(), "acc": correct.mean(),
            "eff_draws": 1.0 / (1.0 - oracle.evidence_decay[tier])}


def run(name: str, cfg: dict, ks=(2, 3, 4)) -> dict:
    o = DialogueOracle(tier_accuracy=dict(cfg["acc"]),
                       evidence_decay=dict(cfg["gam"]),
                       accuracy_by_k=dict(cfg["by_k"]))
    out = {}
    print(f"\n{'='*78}\n{name}\n{'='*78}")
    print(f"{'tier':12s} {'k':>2s} {'1-query acc':>12s} {'eff draws':>10s} "
          f"{'ceiling P(true)':>16s} {'ceiling acc':>12s}")
    for tier in (THINK_TINY, THINK_BIG, OFFLOAD, ASK):
        for k in ks:
            r = ceiling(o, tier, k)
            out[(tier, k)] = r
            print(f"{tier:12s} {k:>2d} {o.accuracy(tier, k):12.3f} "
                  f"{r['eff_draws']:10.2f} {r['p_true']:16.3f} {r['acc']:12.3f}")
    return out


def main():
    ph = run("R1 PLACEHOLDERS", PLACEHOLDER)
    ms = run("R2 MEASURED (local tiers only; offload/ask still placeholders)", MEASURED)

    print(f"\n{'='*78}\nDOES 'A CHEAP MODEL CANNOT BE CHAINED UP' SURVIVE?\n{'='*78}")
    print("Ceiling accuracy = what the tier reaches after unlimited queries.")
    print("The claim needs a WIDE gap between think_tiny's ceiling and think_big's.\n")
    print(f"{'k':>2s} | {'placeholder tiny':>16s} {'big':>7s} {'gap':>7s} "
          f"| {'measured tiny':>14s} {'big':>7s} {'gap':>7s}")
    for k in (2, 3, 4):
        pt, pb = ph[(THINK_TINY, k)]["acc"], ph[(THINK_BIG, k)]["acc"]
        mt, mb = ms[(THINK_TINY, k)]["acc"], ms[(THINK_BIG, k)]["acc"]
        print(f"{k:>2d} | {pt:16.3f} {pb:7.3f} {pb-pt:7.3f} "
              f"| {mt:14.3f} {mb:7.3f} {mb-mt:7.3f}")

    print(f"\n{'-'*78}")
    for k in (3, 4):
        pgap = ph[(THINK_BIG, k)]["acc"] - ph[(THINK_TINY, k)]["acc"]
        mgap = ms[(THINK_BIG, k)]["acc"] - ms[(THINK_TINY, k)]["acc"]
        print(f"k={k}: the tiny-vs-big ceiling gap moves {pgap:.3f} -> {mgap:.3f} "
              f"({(mgap-pgap)/pgap*100:+.0f}%)")


if __name__ == "__main__":
    main()
