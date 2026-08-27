#!/usr/bin/env python3
"""Freeze the measurements into the artefact R1 trains on (R2 Deliverable D).

    python scripts/r2_freeze.py --analysis data/r2_analysis.json --asr-wer 0.113
    python scripts/r2_freeze.py --verify-only        # re-check an existing artefact

This is the actual handoff back to R1. After it lands, R1 retrains and the
headline result stops being "structurally realistic" and becomes "grounded in
measurement".

THE ONE RULE THIS SCRIPT EXISTS TO ENFORCE
------------------------------------------
`sample_evidence` is on the PPO hot path and must never call a language model.
Everything measured offline gets flattened into eight numpy arrays here, and
training reads only those. See docs/R2_HANDOFF.md §1.1.

Schema (all keys written; the first four are required by `load_precomputed`):

    tiers          (T,)      str
    accuracy       (T,)      float   pooled over k
    evidence_decay (T,)      float   gamma per tier, in (0, 1]
    asr_wer        scalar    float
    k_values       (Kn,)     int
    accuracy_by_k  (T, Kn)   float
    n_samples      (T, Kn)   int
    meta_json      scalar    str

UNMEASURED TIERS
----------------
`ask` is the human channel; its accuracy is not something a model measurement
produces, and its gamma comes from the human-simulator pass. Any tier not
present in the analysis is simply left out of the artefact, and
`load_precomputed` keeps R1's placeholder for it -- which is the documented
behaviour, not a silent gap. The meta block records exactly which tiers are
measured and which are inherited, so no number in the final report is of
unknown provenance.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogniwatt.interfaces.dialogue import (EVIDENCE_DECAY, TIER_ACCURACY,
                                           load_precomputed)
from cogniwatt.interfaces.ved import ASK, OFFLOAD, THINK_BIG, THINK_TINY

DEFAULT_ANALYSIS = os.path.join("data", "r2_analysis.json")
DEFAULT_OUT = os.path.join("data", "dialogue_measured.npz")
KNOWN = (THINK_TINY, THINK_BIG, OFFLOAD, ASK)


def build(analysis: Dict, asr_wer: float, ask_accuracy: float,
          ask_gamma: float, extra_meta: Dict,
          ablations: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """Flatten the analysis into the artefact.

    `ablations` are configurations that were measured but are NOT environment
    tiers -- the Q8_0 build of `think_big`, for example, which exists to show
    that quantisation is not the limiting factor. They are dropped here and
    named in the meta block. The unknown-tier check below stays a hard error
    for everything else: silently admitting a name the env does not know is how
    an artefact ends up driving training with a tier that has no cost model.
    """
    ablations = list(ablations or [])
    ks = sorted({int(k) for t in analysis["accuracy"]
                 for k in analysis["accuracy"][t] if k != "pooled"})
    tiers: List[str] = [t for t in analysis["tiers"] if t in KNOWN]
    unknown = set(analysis["tiers"]) - set(KNOWN) - set(ablations)
    if unknown:
        raise ValueError(
            f"analysis contains tiers the env does not know: {unknown}. "
            f"If these are ablations rather than tiers, pass --ablation-tiers.")

    acc, decay, by_k, n_s = [], [], [], []
    for t in tiers:
        acc.append(float(analysis["accuracy"][t]["pooled"]["acc"]))
        decay.append(float(analysis["decay"][t]["gamma"]))
        by_k.append([float(analysis["accuracy"][t][str(k)]["acc"]) for k in ks])
        n_s.append([int(analysis["accuracy"][t][str(k)]["n"]) for k in ks])

    # The `ask` channel, when supplied. Its gamma is genuinely different in
    # kind: asking a person the same question twice tells you nothing, so it
    # should be the lowest of the four.
    if ask_accuracy is not None:
        tiers.append(ASK)
        acc.append(float(ask_accuracy))
        decay.append(float(ask_gamma))
        by_k.append([float(ask_accuracy)] * len(ks))
        n_s.append([0] * len(ks))

    for t, g in zip(tiers, decay):
        if not (0.0 < g <= 1.0):
            raise ValueError(f"evidence_decay[{t}]={g} outside (0, 1] -- "
                             f"load_precomputed will reject this artefact")
    for t, a in zip(tiers, acc):
        if not np.isfinite(a):
            raise ValueError(f"accuracy[{t}] is not finite")

    meta = {
        "produced": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_analysis": analysis.get("source"),
        "n_rows": analysis.get("n_rows"),
        "repeats": analysis.get("repeats"),
        "tiers_measured": [t for t in tiers if t != ASK or ask_accuracy is not None],
        "tiers_inherited_from_r1_placeholders":
            [t for t in KNOWN if t not in tiers],
        "gradient": analysis.get("gradient"),
        "gamma_direction_note": analysis.get("gamma_direction_note"),
        "rho": {t: analysis["decay"][t]["rho"] for t in analysis["decay"]},
        "estimator": "ICC(1) one-way ANOVA over repeats; gamma by least squares "
                     "against (1-g^n)/(1-g); CIs are cluster bootstrap over cores",
        "ablations_measured_not_tiers": {
            t: {"accuracy": analysis["accuracy"][t]["pooled"]["acc"],
                "gamma": analysis["decay"][t]["gamma"]}
            for t in ablations if t in analysis.get("accuracy", {})},
    }
    meta.update(extra_meta)

    return {
        "tiers": np.array(tiers),
        "accuracy": np.array(acc, dtype=np.float64),
        "evidence_decay": np.array(decay, dtype=np.float64),
        "asr_wer": np.float64(asr_wer),
        "k_values": np.array(ks, dtype=np.int64),
        "accuracy_by_k": np.array(by_k, dtype=np.float64),
        "n_samples": np.array(n_s, dtype=np.int64),
        "meta_json": json.dumps(meta),
    }


def verify(path: str) -> int:
    """Load it back the way R1 will, then drive the environment with it."""
    o = load_precomputed(path, strict=True)
    print(f"loaded {path}")
    print(f"  accuracy       {o.tier_accuracy}")
    print(f"  evidence_decay {o.evidence_decay}")
    print(f"  asr_wer        {o.asr_wer}")
    for t, by in o.accuracy_by_k.items():
        print(f"  {t:<12} by k: " + "  ".join(f"k={k}:{v:.3f}" for k, v in sorted(by.items())))

    # Per-k accuracy must fall as k rises, or the artefact contradicts the
    # env's premise that a bigger candidate set is harder.
    for t, by in o.accuracy_by_k.items():
        ks = sorted(by)
        if len(ks) >= 2 and by[ks[0]] < by[ks[-1]]:
            print(f"  WARNING {t}: accuracy rises with k "
                  f"(k={ks[0]}:{by[ks[0]]:.3f} < k={ks[-1]}:{by[ks[-1]]:.3f})")

    from cogniwatt.env.homegrid import HomeGridEdge, TRAIN_DEVICES
    env = HomeGridEdge(devices=TRAIN_DEVICES, oracle=o, seed=0)
    env.reset(seed=1)
    for tier, L in env._likelihoods.items():
        rows = L.sum(axis=1)
        assert np.allclose(rows, 1.0), f"{tier}: likelihood not row-stochastic"
    print("  env drives end to end; all likelihoods row-stochastic")
    print(f"  meta: {json.dumps(o.meta, indent=2)[:600]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", default=DEFAULT_ANALYSIS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--asr-recovery", type=float, default=None,
                    help="referent recovery at the operating point. This is what "
                         "the env should degrade the `ask` channel by -- it "
                         "consumes 'did ask deliver the right referent', not a "
                         "word error rate, and the two diverge sharply in noise "
                         "(0 dB: WER 0.426 vs recovery 0.917). Read from "
                         "data/r2_asr_curve.json when not given.")
    ap.add_argument("--asr-wer", type=float, default=None,
                    help="measured WER at the chosen operating point (r2_asr.py). "
                         "Omitted -> R1's placeholder, flagged in meta.")
    ap.add_argument("--ask-accuracy", type=float, default=None,
                    help="human-channel accuracy before ASR degradation")
    ap.add_argument("--ask-gamma", type=float, default=None,
                    help="gamma for the `ask` channel; should be the lowest tier")
    ap.add_argument("--profile", default=os.path.join("data", "r2_rtx3060_profile.json"),
                    help="hardware profile to reference in meta, if present")
    ap.add_argument("--ablation-tiers", nargs="*", default=["think_big_q8"],
                    help="measured configurations that are NOT env tiers. Dropped "
                         "from the arrays, recorded in meta.")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    if args.verify_only:
        return verify(args.out)

    with open(args.analysis, encoding="utf-8") as f:
        analysis = json.load(f)

    extra: Dict[str, object] = {}
    wer = args.asr_wer
    if wer is None:
        from cogniwatt.interfaces.dialogue import DEFAULT_ASR_WER
        wer = DEFAULT_ASR_WER
        extra["asr_wer_provenance"] = "R1 PLACEHOLDER -- not measured"
        print(f"WARNING: no --asr-wer given; using R1's placeholder {wer}")
    else:
        extra["asr_wer_provenance"] = "measured (scripts/r2_asr.py)"

    if args.ask_accuracy is None:
        extra["ask_provenance"] = "R1 PLACEHOLDER -- `ask` tier not measured"
        print(f"WARNING: `ask` tier not supplied; R1's placeholders stand "
              f"(accuracy {TIER_ACCURACY[ASK]}, gamma {EVIDENCE_DECAY[ASK]})")
    else:
        extra["ask_provenance"] = "measured (human simulator + ASR)"

    if os.path.exists(args.profile):
        with open(args.profile, encoding="utf-8") as f:
            prof = json.load(f)
        extra["hardware_profile"] = {"file": args.profile,
                                     "device": prof.get("device"),
                                     "device_measured": prof.get("device_measured"),
                                     "models": prof.get("models")}
    else:
        extra["hardware_profile"] = None

    # Emit `asr_recovery` alongside `asr_wer`. R1's loader treats it as
    # optional and takes precedence over the WER form, so shipping it removes
    # their bridge script; older loaders ignore the extra key.
    rec = args.asr_recovery
    if rec is None:
        curve = os.path.join("data", "r2_asr_curve.json")
        if os.path.exists(curve):
            with open(curve, encoding="utf-8") as f:
                c = json.load(f)
            rec = float(c["curve"][str(c["operating_snr"])]["recovery"])
            extra["asr_recovery_provenance"] = (
                f"measured, {c['operating_snr']} dB operating point")
    if rec is not None:
        extra["asr_recovery"] = rec

    arrays = build(analysis, wer, args.ask_accuracy,
                   args.ask_gamma if args.ask_gamma is not None
                   else EVIDENCE_DECAY[ASK], extra, args.ablation_tiers)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if rec is not None:
        arrays["asr_recovery"] = np.float64(rec)
    np.savez(args.out, **arrays)
    print(f"wrote artefact -> {args.out}\n")
    return verify(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
