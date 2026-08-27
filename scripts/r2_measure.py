#!/usr/bin/env python3
"""Query every tier on every scenario N times and store every raw response.

    # validate the whole pipeline against a simulator with known ground truth
    python scripts/r2_measure.py --mock --tiers think_tiny think_big offload

    # the real Lab-3 gradient check (cheap, 100 scenarios, 1 repeat)
    python scripts/r2_measure.py --scenarios data/r2_scenarios_pilot100.jsonl \
        --repeats 1 --tiers think_tiny think_big offload

    # the full Lab-5 campaign
    python scripts/r2_measure.py --repeats 10

This is the only script that pays for inference. Everything downstream --
accuracy, gamma, position bias, the failure taxonomy, the confusion matrix --
is derived from the single file it writes, so a re-analysis never re-pays.

DESIGN POINTS THAT PROTECT THE COMPARISON
-----------------------------------------
* **One prompt template for every tier** (`cogniwatt/dialogue/prompts.py`).
  There is deliberately no per-tier prompt argument.
* **The same permutation for every tier.** The candidate shuffle is seeded from
  (scenario_id, repeat) alone, so all three tiers see byte-identical prompts.
  If each tier drew its own shuffle, a tier that happens to draw easier
  positions would look more accurate, and position bias would be confounded
  with grounding ability.
* **Distinct seed per repeat.** Same prompt, different sample. Without this the
  local tiers return the identical token every time and gamma measures nothing.
* **Failed parses are recorded, never silently scored wrong.** A format
  violation and a grounding error are different failures (Deliverable F).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogniwatt.dialogue.backends import Query, make_backend
from cogniwatt.dialogue.prompts import (build_prompt, parse_choice,
                                        parse_reasoning, shuffled_view,
                                        stable_seed)
from cogniwatt.dialogue.schema import load_scenarios

DEFAULT_SCENARIOS = os.path.join("data", "r2_scenarios.jsonl")
DEFAULT_OUT = os.path.join("data", "r2_raw_responses.parquet")
TIERS = ("think_tiny", "think_big", "offload")


def build_queries(scenarios, repeats: int, tier: str,
                  mode: str = "direct") -> List[Dict]:
    """One row per (scenario, repeat). The permutation is tier-independent."""
    rows: List[Dict] = []
    for s in scenarios:
        descs = [c.desc for c in s.candidates]
        for i in range(repeats):
            # Seeded on (scenario, repeat) only -- NOT on tier. This is what
            # makes the cross-tier accuracy comparison a paired comparison, and
            # `stable_seed` is what makes it hold across separate process
            # invocations too (see prompts.stable_seed).
            rng = np.random.default_rng(stable_seed(s.id, i))
            shown, perm, shown_true = shuffled_view(descs, s.true_idx, rng)
            rows.append({
                "qid": f"{tier}|{s.id}|{i}",
                "tier": tier, "scenario_id": s.id, "repeat": i,
                "k": s.k, "core_id": s.core_id, "family": s.family,
                "difficulty": s.difficulty, "noun": s.noun,
                "receptacle": s.receptacle, "form_idx": s.form_idx,
                "instruction": s.instruction,
                "perm": json.dumps(perm), "shown_true_idx": shown_true,
                "true_idx": s.true_idx,
                "true_color": s.candidates[s.true_idx].color,
                "shown_colors": json.dumps([s.candidates[p].color for p in perm]),
                "prompt": build_prompt(s.instruction, shown, mode=mode),
                "prompt_mode": mode,
            })
    return rows


def run_tier(tier: str, rows: List[Dict], args) -> List[Dict]:
    be = make_backend(tier, mock=args.mock, models_dir=args.models_dir,
                      temperature=args.temperature, n_gpu_layers=args.n_gpu_layers,
                      seed=args.seed, live_cloud=args.live_cloud,
                      cloud_api=args.cloud_api)
    if hasattr(be, "max_tokens"):
        be.max_tokens = args.max_tokens or (320 if args.prompt_mode == "reasoning" else 64)
    meta = be.meta()
    print(f"  backend {meta}")
    qs = [Query(r["qid"], r["prompt"], r["k"], r["scenario_id"],
                r["shown_true_idx"], r["repeat"]) for r in rows]

    t0 = time.perf_counter()
    try:
        if tier == "offload" and args.cloud_api and not args.mock and not args.live_cloud:
            # One batch submission for the whole tier -- half price, and the
            # only sane way to move ~7k requests.
            responses = be.answer_batch(qs)
        else:
            responses = []
            for n, q in enumerate(qs, 1):
                responses.append(be.answer(q))
                if n % 200 == 0 or n == len(qs):
                    el = time.perf_counter() - t0
                    print(f"    {n}/{len(qs)}  {el:.0f}s  "
                          f"({n / max(el, 1e-9):.1f}/s)", flush=True)
    finally:
        be.close()

    out: List[Dict] = []
    for r, resp in zip(rows, responses):
        assert r["qid"] == resp.qid, "response/row misalignment"
        # A reasoning response names options while thinking, so its answer must
        # be read from the last line rather than the first match anywhere.
        parser = parse_reasoning if r.get("prompt_mode") == "reasoning" else parse_choice
        display_idx, status = parser(resp.text, r["k"])
        perm = json.loads(r["perm"])
        # Map the model's *display* index back to a dataset index. Skipping
        # this step is the single easiest way to silently score a shuffled
        # dataset wrong.
        dataset_idx = perm[display_idx] if (status == "ok"
                                            and display_idx is not None) else None
        rec = dict(r)
        rec.pop("prompt")  # recoverable from the dataset; keeps the file small
        rec.update({
            "raw_text": resp.text, "latency_ms": resp.latency_ms,
            "error": resp.error, "parse_status": status,
            "display_idx": display_idx, "dataset_idx": dataset_idx,
            "correct": (None if status != "ok"
                        else int(dataset_idx == r["true_idx"])),
            "usage_in": resp.usage.get("input"), "usage_out": resp.usage.get("output"),
            "backend": meta.get("backend"), "model_id": meta.get("model_id"),
        })
        out.append(rec)
    return out


def write_table(df: pd.DataFrame, path: str) -> str:
    """Parquet when pyarrow is present, gzipped CSV otherwise.

    The analysis reads whichever exists, so a missing pyarrow never blocks a
    measurement campaign -- which matters, because the campaign is the
    expensive part and the file format is not.
    """
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception as e:
        alt = path.replace(".parquet", ".csv.gz")
        df.to_csv(alt, index=False, compression="gzip")
        print(f"  (parquet unavailable: {type(e).__name__}; wrote CSV instead)")
        return alt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--tiers", nargs="+", default=list(TIERS))
    ap.add_argument("--repeats", type=int, default=10,
                    help="N in the decay protocol; 1 is enough for a gradient check")
    ap.add_argument("--limit", type=int, default=0, help="first N scenarios only")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="deployment temperature; recorded in meta")
    ap.add_argument("--models-dir", default=os.path.join("models"))
    ap.add_argument("--n-gpu-layers", type=int, default=-1,
                    help="-1 = all layers on GPU (required for a valid hardware anchor)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mock", action="store_true",
                    help="synthetic tiers with known accuracy and known rho")
    ap.add_argument("--prompt-mode", default="direct", choices=("direct", "reasoning"),
                    help="`direct` asks for a bare letter (~5 output tokens). "
                         "`reasoning` permits thinking first and reads the answer "
                         "off the last line; use --max-tokens to give it room.")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="output cap. Defaults to 64 direct / 320 reasoning.")
    ap.add_argument("--cloud-api", action="store_true",
                    help="route `offload` to the Anthropic backends instead of "
                         "the local large GGUF. Requires ANTHROPIC_API_KEY and "
                         "an approved budget; off by default.")
    ap.add_argument("--live-cloud", action="store_true",
                    help="offload via synchronous calls (full price) so latency is real")
    args = ap.parse_args()

    scenarios = load_scenarios(args.scenarios)
    if args.limit:
        scenarios = scenarios[:args.limit]
    n_calls = len(scenarios) * args.repeats * len(args.tiers)
    print(f"scenarios {len(scenarios)}  repeats {args.repeats}  "
          f"tiers {args.tiers}  -> {n_calls} calls")
    if args.mock:
        print("MOCK MODE: synthetic responses, no model is being queried")

    frames: List[pd.DataFrame] = []
    for tier in args.tiers:
        print(f"\n[{tier}]")
        rows = build_queries(scenarios, args.repeats, tier, args.prompt_mode)
        frames.append(pd.DataFrame(run_tier(tier, rows, args)))

    df = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    written = write_table(df, args.out)

    side = args.out.replace(".parquet", "_meta.json")
    with open(side, "w", encoding="utf-8") as f:
        json.dump({"scenarios": args.scenarios, "n_scenarios": len(scenarios),
                   "repeats": args.repeats, "tiers": args.tiers,
                   "temperature": args.temperature, "mock": args.mock,
                   "live_cloud": args.live_cloud, "seed": args.seed,
                   "n_gpu_layers": args.n_gpu_layers,
                   "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "rows": int(len(df))}, f, indent=2)

    print(f"\nwrote {len(df)} rows -> {written}")
    print(f"wrote run meta      -> {side}")
    ok = df["parse_status"].eq("ok")
    print(f"parse ok {ok.mean():.3f}   errors {df['error'].notna().mean():.3f}")
    for tier, g in df.groupby("tier"):
        first = g[g["repeat"] == 0]
        acc = first.loc[first["parse_status"].eq("ok"), "correct"].mean()
        print(f"  {tier:<11} first-repeat accuracy {acc:.3f}  (n={len(first)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
