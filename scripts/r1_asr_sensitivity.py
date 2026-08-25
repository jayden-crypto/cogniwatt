"""Where does the "human crowds out on-device inference" finding flip?

Owner: R1.

R2 measured ASR at WER 0.0337 (15 dB, faster-whisper small) but flagged that the
corpus is SAPI text-to-speech, so that figure is a *lower bound* on real speech
and the derived `ask` accuracy of 0.937 is an *upper bound*
(docs/R1_REPLY_2.md section 3.1). Reaching the ~0.85 where the human stops being
the best evidence buy needs WER ~0.124 -- 3.7x the measured value, which is
ordinary for a `small` model on real conversational speech.

R2's recommendation (3) was to sweep `asr_wer` and report where the conclusion
inverts, turning an unmeasured dependency into a stated one. That is this
script. It sweeps the channel quality and reports, per value, whether `ask`
still dominates the on-device tiers on expected information per unit reward.

This is an analytic sweep over the belief filter, not a training sweep: the
question is which channel a rational buyer prefers, which the cost model and the
oracle answer directly and without 2M steps per point.

    python3 scripts/r1_asr_sensitivity.py
"""
from __future__ import annotations

import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from cogniwatt.env.belief import BeliefState
from cogniwatt.env.homegrid import RewardConfig
from cogniwatt.interfaces.dialogue import load_precomputed
from cogniwatt.interfaces.ved import (ASK, OFFLOAD, THINK_BIG, THINK_TINY,
                                      VirtualEdgeDevice)

ARTEFACT = "data/dialogue_measured.npz"
K = 3
TRIALS = 3000
N_QUERIES = 12


def one_query_value(oracle, tier, k, rng, trials=TRIALS):
    """Expected posterior on the true referent after ONE query of `tier`."""
    lik = oracle.likelihood(tier, k)
    out = np.empty(trials)
    for t in range(trials):
        i = int(rng.integers(k))
        b = BeliefState(k, 4)
        b.update(oracle.sample_evidence(tier, i, k, rng), lik,
                 weight=oracle.evidence_weight(tier, 0))
        out[t] = b.probs[i]
    return float(out.mean())


def ceiling_value(oracle, tier, k, rng, n=N_QUERIES, trials=TRIALS):
    """Expected posterior after querying `tier` until it saturates."""
    lik = oracle.likelihood(tier, k)
    out = np.empty(trials)
    for t in range(trials):
        i = int(rng.integers(k))
        b = BeliefState(k, 4)
        for q in range(n):
            b.update(oracle.sample_evidence(tier, i, k, rng), lik,
                     weight=oracle.evidence_weight(tier, q))
        out[t] = b.probs[i]
    return float(out.mean())


def tier_cost(tier, cfg, device="m2_air"):
    """Reward cost of one query, matching HomeGridEdge._spend exactly."""
    ved = VirtualEdgeDevice(device, np.random.default_rng(0))
    c = ved.quote(tier)
    if not c.available:
        return float("inf")
    r = cfg.lambda_energy * c.energy_mJ / 1000.0 + cfg.lambda_time * c.latency_ms / 1000.0
    if tier == ASK:
        r += cfg.lambda_ask
    if tier == OFFLOAD:
        r += cfg.lambda_cloud
    return r


def cost_to_reach(oracle, tier, k, target, costs, rng, n_max=N_QUERIES,
                  trials=TRIALS):
    """Cheapest total reward cost to reach `target` expected confidence.

    This is the decision-relevant quantity, and the first version of this script
    got it wrong by ranking channels on information-per-unit-cost. That metric
    says `think_big` beats `ask` at every ASR quality -- which contradicts the
    trained policies, where think_big usage is 0.00. The reason is saturation: a
    channel with a ceiling below `target` can never reach it, at any price, so
    its efficiency per query is irrelevant. Returns inf for such a channel.
    """
    lik = oracle.likelihood(tier, k)
    run = np.zeros(trials)
    idx = np.array([int(rng.integers(k)) for _ in range(trials)])
    beliefs = [BeliefState(k, 4) for _ in range(trials)]
    for q in range(n_max):
        w = oracle.evidence_weight(tier, q)
        for t in range(trials):
            beliefs[t].update(oracle.sample_evidence(tier, idx[t], k, rng),
                              lik, weight=w)
        run = np.array([b.probs[i] for b, i in zip(beliefs, idx)])
        if run.mean() >= target:
            return (q + 1) * costs[tier], q + 1, float(run.mean())
    return float("inf"), None, float(run.mean())


def main():
    cfg = RewardConfig()
    base = load_precomputed(ARTEFACT, strict=True)
    measured = base.asr_wer
    costs = {t: tier_cost(t, cfg) for t in (THINK_TINY, THINK_BIG, OFFLOAD, ASK)}

    print(f"artefact: {ARTEFACT}   measured asr_wer = {measured:.4f}")
    print(f"reward cost per query (m2_air): "
          + "  ".join(f"{t}={costs[t]:.3f}" for t in costs))
    print(f"\nk={K}. 'value' = expected posterior on the true referent.")
    print("'per cost' = one-query value divided by that query's reward cost.\n")

    TARGET = 0.85
    print(f"Cheapest total reward cost to reach {TARGET:.2f} expected confidence.")
    print("inf = the channel saturates below the target and can never reach it.\n")
    hdr = (f"{'asr_wer':>8s} {'ask acc':>8s} | {'ask cost':>9s} {'n':>3s} | "
           f"{'big cost':>9s} {'n':>3s} | {'cloud cost':>11s} {'n':>3s} | {'cheapest':>10s}")
    print(hdr); print("-" * len(hdr))

    rows = []
    for wer in [measured, 0.05, 0.08, 0.10, 0.124, 0.15, 0.20, 0.30, 0.40]:
        o = load_precomputed(ARTEFACT, strict=True)
        o.asr_wer = float(wer)
        rng = np.random.default_rng(7)
        res = {t: cost_to_reach(o, t, K, TARGET, costs, rng)
               for t in (ASK, THINK_BIG, OFFLOAD, THINK_TINY)}
        best = min(res, key=lambda t: res[t][0])
        tag = "  <- measured" if wer == measured else ""
        def f(t):
            c, n, _ = res[t]
            return (f"{c:9.3f}" if np.isfinite(c) else f"{'inf':>9s}",
                    f"{n:>3d}" if n else f"{'-':>3s}")
        ac, an = f(ASK); bc, bn = f(THINK_BIG); oc, on = f(OFFLOAD)
        print(f"{wer:8.4f} {o.accuracy(ASK):8.4f} | {ac} {an} | {bc} {bn} | "
              f"{oc} {on} | {best:>10s}{tag}")
        rows.append({"asr_wer": wer, "ask_accuracy": o.accuracy(ASK),
                     "cost_ask": res[ASK][0], "cost_big": res[THINK_BIG][0],
                     "cost_offload": res[OFFLOAD][0], "cost_tiny": res[THINK_TINY][0],
                     "ceiling_big": res[THINK_BIG][2], "cheapest": best})

    flip = next((r for r in rows if r["cheapest"] != rows[0]["cheapest"]), None)
    print()
    if flip is None:
        print(f"NO FLIP over the swept range: `{rows[0]['cheapest']}` is the "
              f"cheapest route to {TARGET:.2f} confidence at every ASR quality tested.")
        print("The finding is robust to the ASR uncertainty R2 flagged.")
    else:
        print(f"FLIPS at asr_wer = {flip['asr_wer']:.3f} "
              f"(ask accuracy {flip['ask_accuracy']:.3f}), which is "
              f"{flip['asr_wer']/measured:.1f}x the measured value.")
    pathlib.Path("results/asr_sensitivity.json").write_text(
        json.dumps({"measured_asr_wer": measured, "k": K, "costs": costs,
                    "rows": rows}, indent=2))
    print("-> results/asr_sensitivity.json")


if __name__ == "__main__":
    main()
