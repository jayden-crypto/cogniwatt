"""Add `asr_recovery` to R2's artefact from their own measured ASR curve.

Owner: R1. Temporary bridge.

R2 measured both WER and referent recovery per SNR
(`data/r2_asr_curve.json`) and argued in docs/R1_REPLY_2.md section 3.2 that the
env should consume recovery rather than (1 - WER), because the env's `ask`
action asks "was the right referent recovered", not "how many words were
transcribed correctly". R1 accepted that. But the shipped
`data/dialogue_measured.npz` carries only `asr_wer`.

Rather than hand-edit R2's artefact, this reads their curve file, takes the
recovery figure at the same operating SNR they chose for `asr_wer`, and writes a
new artefact with the key added. Provenance is recorded in `meta_json` so the
number is traceable to their measurement rather than appearing from nowhere.

**Delete this script once R2 emits `asr_recovery` from `r2_freeze.py`.**

    python3 scripts/r1_merge_recovery.py
"""
from __future__ import annotations

import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "data/dialogue_measured.npz"
CURVE = ROOT / "data/r2_asr_curve.json"
OUT = ROOT / "data/dialogue_measured_recovery.npz"


def main():
    d = np.load(SRC, allow_pickle=True)
    curve = json.loads(CURVE.read_text())
    snr = str(curve["operating_snr"])
    point = curve["curve"][snr]
    recovery, wer = float(point["recovery"]), float(point["wer"])

    stored_wer = float(d["asr_wer"])
    if abs(stored_wer - wer) > 1e-6:
        print(f"  note: artefact asr_wer={stored_wer:.4f} vs curve@{snr}dB "
              f"wer={wer:.4f} -- using the curve's operating point")

    meta = json.loads(str(d["meta_json"]))
    meta["asr_recovery_provenance"] = (
        f"R1 merged from data/r2_asr_curve.json at the operating SNR "
        f"({snr} dB) chosen by R2; recovery={recovery:.4f} vs "
        f"1-wer={1 - wer:.4f}. R2 recommended recovery over WER in "
        f"docs/R1_REPLY_2.md section 3.2. Remove this key once r2_freeze.py "
        f"emits asr_recovery directly.")

    payload = {k: d[k] for k in d.files if k != "meta_json"}
    payload["asr_recovery"] = np.float64(recovery)
    payload["meta_json"] = json.dumps(meta)
    np.savez(OUT, **payload)
    print(f"wrote {OUT}")

    from cogniwatt.interfaces.dialogue import load_precomputed
    a = load_precomputed(str(SRC), strict=True)
    b = load_precomputed(str(OUT), strict=True)
    print(f"  ask accuracy  (1 - wer) : {a.accuracy('ask'):.4f}")
    print(f"  ask accuracy  recovery  : {b.accuracy('ask'):.4f}"
          f"   ({(b.accuracy('ask') - a.accuracy('ask')) * 100:+.2f} pt)")


if __name__ == "__main__":
    main()
