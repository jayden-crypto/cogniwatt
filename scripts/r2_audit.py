#!/usr/bin/env python3
"""Human audit of the authored cores (R2 Deliverable A, the defensibility step).

    python scripts/r2_audit.py --export                  # blind sheet for annotators
    python scripts/r2_audit.py --annotate adit           # interactive pass
    python scripts/r2_audit.py --report                  # agreement + kappa + verdicts

WHY THE CORE IS THE AUDIT UNIT
------------------------------
Every generated scenario is a colour/position/phrasing permutation of one
authored core, so a core that survives audit certifies all 12 scenarios built
from it. Auditing 60 cores is therefore a *stronger* claim than auditing 100
loose scenarios, and a smaller job.

THE STANDARD
------------
docs/R2_HANDOFF.md §4: if two humans disagree on the answer, the scenario is
bad -- drop it. A dataset whose ground truth is not defensible invalidates
every number downstream, so this is a hard gate, not a formality.

WHAT COUNTS AS AN ANNOTATOR
---------------------------
Only a human pass counts toward inter-annotator agreement. A model annotation
is recorded with `kind: "model"` and is reported *separately*; it never enters
the kappa. This matters here because the cores were authored by a model, so a
model's agreement with them measures authorship memory, not resolvability.

PRESENTATION
------------
Each core is shown at full width -- target plus every distractor, shuffled --
which is the hardest form and audits all distractors at once. An annotator who
picks a distractor has found either a bad distractor or a bad core; the report
distinguishes the two by looking at *which* distractor drew the vote.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogniwatt.dialogue.authored import CORES, PREP

AUDIT_DIR = os.path.join("data", "r2_audit")
VERDICT = os.path.join("data", "r2_core_audit.json")
#: Fixed so every annotator sees an identical sheet and the ids in one person's
#: notes mean the same thing in another's.
SHUFFLE_SEED = 611


def presentation(core, rng) -> Dict[str, object]:
    """Blind view of a core: instruction plus all candidates, shuffled."""
    opts = [core.target] + list(core.distractors)
    perm = [int(i) for i in rng.permutation(len(opts))]
    return {
        "core_id": core.id,
        "instruction": core.instruction(
            "put the {noun} {prep} the {receptacle}"),
        "options": [opts[i][0] for i in perm],
        "rooms": [opts[i][1] for i in perm],
        "perm": perm,                       # perm[j] == 0 means shown j is the target
        "answer_shown_idx": perm.index(0),  # withheld from annotators
        "family": core.family,
        "difficulty": core.difficulty,
        "rationale": core.rationale,
    }


def build_sheet() -> List[Dict[str, object]]:
    rng = np.random.default_rng(SHUFFLE_SEED)
    return [presentation(c, rng) for c in CORES]


def cmd_export(path: str) -> int:
    sheet = build_sheet()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Core audit sheet\n\n")
        f.write("For each item, write the number of the object you think the "
                "speaker means.\nAnswer from the text alone. If you genuinely "
                "cannot choose, write `?` -- that is\na useful signal, not a "
                "failure.\n\nDo not discuss items with the other annotators "
                "until everyone has finished.\n\n---\n\n")
        for n, p in enumerate(sheet, 1):
            f.write(f"**{n}. [{p['core_id']}]**  “{p['instruction']}”\n\n")
            for j, (o, r) in enumerate(zip(p["options"], p["rooms"])):
                f.write(f"- `{j}` {o} ({r.replace('_', ' ')})\n")
            f.write("\nAnswer: `___`\n\n")
    print(f"wrote blind sheet ({len(sheet)} cores) -> {path}")
    print("The answer key is NOT in this file. Collect answers with "
          "`--annotate <name>` or as one `<name>.jsonl` per annotator in "
          f"{AUDIT_DIR}/.")
    return 0


def cmd_annotate(name: str, kind: str) -> int:
    sheet = build_sheet()
    os.makedirs(AUDIT_DIR, exist_ok=True)
    out = os.path.join(AUDIT_DIR, f"{name}.jsonl")
    done = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            done = {json.loads(l)["core_id"] for l in f if l.strip()}
        print(f"resuming: {len(done)} already annotated")

    print(f"\n{len(sheet) - len(done)} cores to go. "
          f"Enter an index, `?` if undecidable, or `q` to stop.\n")
    with open(out, "a", encoding="utf-8") as f:
        for n, p in enumerate(sheet, 1):
            if p["core_id"] in done:
                continue
            print(f"\n[{n}/{len(sheet)}]  “{p['instruction']}”")
            for j, (o, r) in enumerate(zip(p["options"], p["rooms"])):
                print(f"   {j}. {o}  ({r.replace('_', ' ')})")
            try:
                raw = input("   which one? ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nstopped");  return 0
            if raw == "q":
                print("stopped");  return 0
            choice: Optional[int] = None
            if raw not in ("?", ""):
                try:
                    choice = int(raw)
                except ValueError:
                    print("   (not a number -- recorded as undecidable)")
            note = input("   note (optional): ").strip()
            f.write(json.dumps({"core_id": p["core_id"], "annotator": name,
                                "kind": kind, "choice": choice,
                                "note": note}) + "\n")
            f.flush()
    print(f"\nwrote {out}")
    return 0


def _load() -> Dict[str, Dict[str, Dict]]:
    """{core_id: {annotator: record}}"""
    out: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    if not os.path.isdir(AUDIT_DIR):
        return out
    for fn in sorted(os.listdir(AUDIT_DIR)):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(AUDIT_DIR, fn), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[r["core_id"]][r["annotator"]] = r
    return out


def fleiss_kappa(table: np.ndarray) -> float:
    """table[i, c] = number of annotators assigning item i to category c.

    Requires the same number of ratings per item; items with a different count
    are dropped by the caller.
    """
    n = table.sum(axis=1)
    if len(set(n.tolist())) != 1 or n[0] < 2:
        return float("nan")
    N, nn = table.shape[0], int(n[0])
    p_j = table.sum(axis=0) / (N * nn)
    P_i = ((table ** 2).sum(axis=1) - nn) / (nn * (nn - 1))
    P_bar, Pe = P_i.mean(), float((p_j ** 2).sum())
    return float((P_bar - Pe) / (1 - Pe)) if Pe < 1 else float("nan")


def cmd_report(out_path: str, only: Optional[List[str]] = None) -> int:
    """`only` restricts the human pool -- the two-annotator sensitivity in R1's
    reply 7 (kappa 0.722 before the third reader) is reproduced with
    `--annotators reader_a,reader_b`. It is a sensitivity, not a way to drop an
    inconvenient rater: a pass is excluded for provenance, never for its numbers
    (see data/r2_audit_quarantine/README.md)."""
    sheet = {p["core_id"]: p for p in build_sheet()}
    anns = _load()
    if not anns:
        print(f"no annotations found in {AUDIT_DIR}/")
        print("run `--export` then `--annotate <name>` for each annotator.")
        return 1

    humans = sorted({r["annotator"] for v in anns.values() for r in v.values()
                     if r.get("kind") == "human"})
    if only:
        missing = set(only) - set(humans)
        if missing:
            raise SystemExit(f"--annotators names unknown annotators {sorted(missing)}; "
                             f"have {humans}")
        humans = [a for a in humans if a in only]
    models = sorted({r["annotator"] for v in anns.values() for r in v.values()
                     if r.get("kind") != "human"})
    print(f"human annotators: {humans or '(none)'}")
    print(f"model annotators: {models or '(none)'}  (reported separately, "
          f"excluded from kappa)\n")

    verdicts: Dict[str, Dict[str, object]] = {}
    n_cat = 4
    rows, row_ids = [], []
    for cid, p in sheet.items():
        recs = anns.get(cid, {})
        hv = [recs[a]["choice"] for a in humans if a in recs]
        key = p["answer_shown_idx"]
        agree_key = [c == key for c in hv if c is not None]
        undecidable = sum(1 for a in humans if a in recs
                          and recs[a]["choice"] is None)
        distinct = len({c for c in hv if c is not None})

        if not hv:
            status = "unaudited"
        elif undecidable:
            status = "undecidable"
        elif distinct > 1:
            status = "disagree"          # annotators split -> drop the core
        elif not all(agree_key):
            status = "key_wrong"         # annotators agree, but not with us
        else:
            status = "agree"

        # Which distractor drew the votes? A single bad distractor is fixable;
        # a core nobody can resolve is not.
        wrong_votes = Counter(c for c in hv if c is not None and c != key)
        verdicts[cid] = {
            "status": status, "n_human": len(hv), "key_shown_idx": key,
            "votes": hv, "wrong_votes": {str(k): v for k, v in wrong_votes.items()},
            "family": p["family"], "difficulty": p["difficulty"],
            "model_votes": {a: recs[a]["choice"] for a in models if a in recs},
            "notes": [recs[a].get("note") for a in recs if recs[a].get("note")],
        }
        if len(hv) >= 2 and all(c is not None for c in hv):
            counts = np.zeros(n_cat)
            for c in hv:
                if 0 <= c < n_cat:
                    counts[c] += 1
            rows.append(counts); row_ids.append(cid)

    by_status = Counter(v["status"] for v in verdicts.values())
    print("VERDICTS")
    for s in ("agree", "key_wrong", "disagree", "undecidable", "unaudited"):
        if by_status.get(s):
            print(f"  {s:<13}{by_status[s]:>4}")

    audited = [v for v in verdicts.values() if v["status"] != "unaudited"]
    if audited:
        keep = by_status.get("agree", 0)
        print(f"\n  retained {keep}/{len(audited)} audited cores "
              f"({keep / len(audited):.1%})")

    # Written into the verdict file as well as printed, so the freeze can
    # carry a *computed* kappa in meta_json rather than one copied from a
    # reply document.
    agreement: Dict[str, object] = {"n_annotators": len(humans),
                                    "annotators": humans,
                                    "n_fully_rated": len(rows),
                                    "raw_agreement": None, "fleiss_kappa": None}
    if rows:
        counts = np.array(rows)
        same = np.array([c.max() == c.sum() for c in counts])
        print(f"\nINTER-ANNOTATOR AGREEMENT ({len(humans)} humans, "
              f"{len(rows)} fully-rated cores)")
        print(f"  raw agreement   {same.mean():.3f}")
        k = fleiss_kappa(counts)
        print(f"  Fleiss kappa    {k:.3f}" if np.isfinite(k)
              else "  Fleiss kappa    n/a (uneven ratings per core)")
        agreement["raw_agreement"] = float(same.mean())
        agreement["fleiss_kappa"] = float(k) if np.isfinite(k) else None
    else:
        print("\nINTER-ANNOTATOR AGREEMENT: needs >= 2 human annotators "
              "with complete passes.")

    if models:
        print("\nMODEL ANNOTATOR (not evidence of resolvability -- the cores "
              "were model-authored)")
        for a in models:
            hits = [1 for cid, p in sheet.items()
                    if a in anns.get(cid, {})
                    and anns[cid][a]["choice"] == p["answer_shown_idx"]]
            tot = sum(1 for cid in sheet if a in anns.get(cid, {}))
            if tot:
                print(f"  {a}: agrees with key on {len(hits)}/{tot} "
                      f"({len(hits) / tot:.1%})")

    bad = [(c, v) for c, v in verdicts.items()
           if v["status"] in ("disagree", "key_wrong", "undecidable")]
    if bad:
        print(f"\nCORES TO FIX OR DROP ({len(bad)})")
        for cid, v in bad:
            print(f"  {cid:<20} {v['status']:<12} votes={v['votes']} "
                  f"key={v['key_shown_idx']}  ({v['family']}/{v['difficulty']})")
            for n in v["notes"]:
                print(f"      note: {n}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"annotators_human": humans, "annotators_model": models,
                   "by_status": dict(by_status), "agreement": agreement,
                   "verdicts": verdicts}, f, indent=2)
    print(f"\nwrote verdicts -> {out_path}")
    print("Feed to the build with: "
          "python scripts/r2_build_dataset.py --audit data/r2_core_audit.json "
          "--drop-unaudited")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", nargs="?", const=os.path.join("docs", "r2_audit_sheet.md"),
                    default=None, help="write the blind annotation sheet")
    ap.add_argument("--annotate", default=None, help="annotator name")
    ap.add_argument("--kind", default="human", choices=("human", "model"))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--annotators", default=None,
                    help="comma-separated subset of human annotators for a "
                         "sensitivity report (e.g. the two-rater run); "
                         "pair with --out so the main verdict file is untouched")
    ap.add_argument("--out", default=VERDICT)
    args = ap.parse_args()

    if args.export:
        return cmd_export(args.export)
    if args.annotate:
        return cmd_annotate(args.annotate, args.kind)
    if args.report:
        only = [a.strip() for a in args.annotators.split(",")] if args.annotators else None
        if only and args.out == VERDICT:
            raise SystemExit("--annotators is a sensitivity run; give it its own --out "
                             "so it cannot overwrite the three-rater verdicts")
        return cmd_report(args.out, only)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
