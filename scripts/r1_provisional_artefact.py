"""Build a PROVISIONAL dialogue artefact from R2's handback tables.

Owner: R1.

R2 delivered docs/R1_HANDBACK.md but not the data files it references, so
`data/dialogue_measured.npz` is not in hand. Every measured number is however
tabulated in handback section 2, which is enough to reconstruct an artefact with
the same schema and answer the one question R2 asked that is blocking
(section 8.1: should R1 retrain now, given that measured local tiers are mixed
with placeholder cloud/ask?).

This is NOT a substitute for the real artefact:
  * offload and ask are R1 placeholders, unmeasured -- that is the whole point
    of the experiment this feeds;
  * n_samples is the authored-core count, not R2's per-cell counts;
  * it must be replaced the moment the real .npz arrives.

`meta_json` records all of that so no figure drawn from it can be mistaken for
a measured result.

    python3 scripts/r1_provisional_artefact.py
"""
from __future__ import annotations

import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data/dialogue_provisional.npz"

# docs/R1_HANDBACK.md section 2
TIERS = ["think_tiny", "think_big", "offload", "ask"]
K_VALUES = [2, 3, 4]
ACC_BY_K = {
    "think_tiny": [0.529, 0.396, 0.317],   # measured
    "think_big":  [0.671, 0.542, 0.492],   # measured
    "offload":    [0.880, 0.880, 0.880],   # R1 PLACEHOLDER - unmeasured
    "ask":        [0.970, 0.970, 0.970],   # R1 PLACEHOLDER - unmeasured
}
POOLED = {"think_tiny": 0.414, "think_big": 0.568, "offload": 0.88, "ask": 0.97}
GAMMA = {"think_tiny": 0.834, "think_big": 0.605,   # measured
         "offload": 0.35, "ask": 0.15}              # R1 PLACEHOLDER
ASR_WER = 0.08                                       # R1 PLACEHOLDER - unmeasured


def main():
    meta = {
        "provenance": "RECONSTRUCTED from docs/R1_HANDBACK.md section 2 tables, "
                      "not from R2's data files (which were not delivered)",
        "status": "PROVISIONAL - replace with data/dialogue_measured.npz on arrival",
        "tiers_measured": ["think_tiny", "think_big"],
        "tiers_inherited_from_r1_placeholders": ["offload", "ask"],
        "models": {"think_tiny": "Qwen2.5-0.5B-Instruct-Q8_0",
                   "think_big": "Llama-3.2-3B-Instruct-Q4_K_M"},
        "temperature": 0.7, "repeats": 10, "n_scenarios": 720,
        "parse_rate": 1.0,
        "gamma_method": "ICC(1) one-way ANOVA -> n_eff -> least squares vs geometric",
        "caveats": [
            "gamma ranks tiers backwards (tiny 0.834 > big 0.605); it measures "
            "variability, not information -- never quote it alone",
            "the big->offload gap is inflated because offload is a placeholder "
            "while think_big is measured (handback section 3.2)",
            "60/720 scenarios in the already_satisfied family are contested and "
            "not yet adjudicated (handback section 3.3)",
        ],
    }
    np.savez(
        OUT,
        tiers=np.array(TIERS),
        accuracy=np.array([POOLED[t] for t in TIERS], dtype=float),
        evidence_decay=np.array([GAMMA[t] for t in TIERS], dtype=float),
        asr_wer=np.float64(ASR_WER),
        k_values=np.array(K_VALUES, dtype=int),
        accuracy_by_k=np.array([ACC_BY_K[t] for t in TIERS], dtype=float),
        n_samples=np.array([[720, 720, 720]] * len(TIERS), dtype=int),
        meta_json=json.dumps(meta),
    )
    print(f"wrote {OUT}")

    from cogniwatt.interfaces.dialogue import load_precomputed
    o = load_precomputed(str(OUT), strict=True)
    print(f"  loads OK: think_big@k=4 = {o.accuracy('think_big', k=4):.3f}, "
          f"gamma = {o.evidence_decay}")
    from cogniwatt.env.homegrid import HomeGridEdge, TRAIN_DEVICES
    env = HomeGridEdge(devices=TRAIN_DEVICES, oracle=o, seed=0)
    env.reset(seed=1)
    print("  drives the environment end to end: OK")


if __name__ == "__main__":
    main()
