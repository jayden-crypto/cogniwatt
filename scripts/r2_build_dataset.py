#!/usr/bin/env python3
"""Expand the authored cores into the grounding dataset (R2 Deliverable A).

    python scripts/r2_build_dataset.py                  # full set -> data/r2_scenarios.jsonl
    python scripts/r2_build_dataset.py --pilot 100      # Lab-3 pilot subset
    python scripts/r2_build_dataset.py --check-only     # validate, write nothing

WHAT THIS GUARANTEES
--------------------
Everything below is *enforced by construction and then re-asserted*, because a
dataset with a hidden shortcut produces a beautiful accuracy gradient that means
nothing (docs/R2_HANDOFF.md §3, §11).

* **No position prior.** The correct answer's index is assigned round-robin per
  k, so it is exactly uniform. A model that always answers 0 scores exactly 1/k.
* **No colour prior.** The correct answer's colour is assigned round-robin over
  the env's six COLORS, so it is uniform to within one count. Colour carries
  zero information about the answer -- by design, since the env treats colour as
  an arbitrary label.
* **No length prior.** Descriptions come from the authored pool untouched except
  for the colour splice, which adds exactly one word to every candidate alike.
  The script reports how often the target is the longest candidate; at chance
  this is 1/k.
* **Balanced coverage.** 6 nouns x 10 pragmatic families x k in {2,3,4}, equal
  counts per cell.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogniwatt.dialogue.authored import (CORES, INSTRUCTION_FORMS, Core,
                                         colorize, validate_cores)
from cogniwatt.dialogue.schema import (COLORS, K_VALUES, Candidate, Scenario,
                                       save_scenarios, validate_scenario)

DEFAULT_OUT = os.path.join("data", "r2_scenarios.jsonl")
DEFAULT_REPORT = os.path.join("data", "r2_scenarios_report.json")
#: 60 cores x 4 variants x 3 values of k = 720 scenarios, 240 per k.
#: The handoff asks for >= 600 with 200 per k; the surplus is headroom for
#: dropping any core that fails the human audit without falling under quota.
VARIANTS_PER_CORE_PER_K = 4


def _distractor_subsets(core: Core, k: int) -> List[List[int]]:
    """Which distractors to show, as index lists, cycled across variants.

    For k=4 there is only one subset (all three), so those variants differ by
    colour, position and phrasing alone -- which is exactly the ablation that
    makes colour/position bias measurable.
    """
    n_needed = k - 1
    idx = range(len(core.distractors))
    return [list(c) for c in combinations(idx, n_needed)]


def audited_core_ids(cores: List[Core], verdict_path: str, drop_unaudited: bool
                     ) -> set:
    """Ids of the cores the human audit cleared.

    docs/R2_HANDOFF.md §4 is a hard gate: if annotators disagree on the answer,
    the core is bad and everything generated from it is bad with it.
    """
    with open(verdict_path, encoding="utf-8") as f:
        verdicts = json.load(f)["verdicts"]

    kept, dropped = set(), []
    for c in cores:
        status = verdicts.get(c.id, {}).get("status", "unaudited")
        if status == "agree" or (status == "unaudited" and not drop_unaudited):
            kept.add(c.id)
        else:
            dropped.append((c.id, status))

    print(f"audit: kept {len(kept)}/{len(cores)} cores")
    for cid, st in dropped:
        print(f"  dropped {cid:<20} ({st})")
    if not kept:
        raise SystemExit("audit dropped every core; nothing to build")
    return kept


def apply_audit(scenarios: List[Scenario], kept_core_ids: set) -> List[Scenario]:
    """Drop every scenario generated from a core that failed the audit.

    Runs *after* `build()` has assigned ids, for the same reason as
    `drop_contested`: ids are positional (`sc_0041`), so removing cores before
    expansion renumbers every survivor and silently breaks the join to the raw
    response table. The first version of this filter did exactly that -- with
    the 29 audited cores, all 348 surviving ids named a *different* scenario
    from the one the raw table had measured under that id (`sc_0000` became
    `mug_afford`; the raw table had it as `mug_already`, a dropped core), and
    `r2_analyze.py --scenarios` would have scored dropped-core responses as the
    audited set without a single error. `tests/test_r2_estimators.py` now pins
    id stability under the audit filter.
    """
    keep = [s for s in scenarios if s.core_id in kept_core_ids]
    print(f"audit: {len(scenarios)} -> {len(keep)} scenarios "
          f"({len({s.core_id for s in keep})} cores); ids unchanged")
    return keep


def build(variants: int = VARIANTS_PER_CORE_PER_K, seed: int = 20260820,
          cores: Optional[List[Core]] = None) -> List[Scenario]:
    validate_cores()
    cores = CORES if cores is None else cores
    rng = np.random.default_rng(seed)
    scenarios: List[Scenario] = []

    # Round-robin counters give exact uniformity rather than merely unbiased
    # sampling -- with 240 scenarios per k, sampling would still leave a visible
    # wobble that we would then have to argue is noise.
    pos_ctr: Dict[int, int] = {k: 0 for k in K_VALUES}
    col_ctr: Dict[int, int] = {k: 0 for k in K_VALUES}
    n = 0

    for k in K_VALUES:
        subsets = None
        for core in cores:
            subsets = _distractor_subsets(core, k)
            forms = core.forms()
            for v in range(variants):
                d_idx = subsets[v % len(subsets)]
                form = forms[v % len(forms)]
                instruction = core.instruction(form)

                # Colours: target's is round-robin; distractors take distinct
                # colours drawn at random from the rest.
                target_color = COLORS[col_ctr[k] % len(COLORS)]
                col_ctr[k] += 1
                pool = [c for c in COLORS if c != target_color]
                other_colors = list(rng.choice(pool, size=k - 1, replace=False))

                # Position: round-robin, so P(true_idx = j) is exactly 1/k.
                true_idx = pos_ctr[k] % k
                pos_ctr[k] += 1

                slots: List[Candidate] = [None] * k  # type: ignore[list-item]
                t_desc, t_room = core.target
                slots[true_idx] = Candidate(target_color,
                                            colorize(t_desc, core.noun, target_color),
                                            t_room)
                free = [j for j in range(k) if j != true_idx]
                # Shuffle which distractor lands in which free slot, so
                # distractor identity is not correlated with position either.
                order = list(rng.permutation(len(free)))
                for slot_pos, d_pos in zip(free, order):
                    desc, room = core.distractors[d_idx[d_pos]]
                    col = str(other_colors[d_pos])
                    slots[slot_pos] = Candidate(col, colorize(desc, core.noun, col), room)

                s = Scenario(
                    id=f"sc_{n:04d}", k=k, noun=core.noun,
                    receptacle=core.receptacle, instruction=instruction,
                    candidates=list(slots), true_idx=true_idx,
                    rationale=core.rationale, difficulty=core.difficulty,
                    family=core.family, audit="unaudited",
                    core_id=core.id, form_idx=forms.index(form),
                )
                validate_scenario(s)
                scenarios.append(s)
                n += 1
    return scenarios


#: Families whose distractors are *defined* as sitting at the named destination.
#: For these, an instruction that names the receptacle is genuinely ambiguous:
#: in "put the mug on the shelf", with a distractor described as already on the
#: shelf, "the mug on the shelf" parses as an identifying description as readily
#: as a destination. The "put the {noun} away" form names no receptacle and is
#: immune. Measured on Llama-3.2-3B: 0.875/0.833/0.722 error on the receptacle
#: forms vs 0.417 on the away form, where every other family moves only
#: 0.402 -> 0.333. See docs/r2_measurements.md §3c.
CONTESTED_FAMILIES = ("already_satisfied",)


def drop_contested(scenarios: List[Scenario]) -> List[Scenario]:
    """Remove scenarios whose instruction is ambiguous, keeping ids stable.

    Runs *after* id assignment on purpose. Ids are positional (`sc_0041`), so
    filtering before assignment would renumber everything and silently break
    the join to `data/r2_raw_responses.csv.gz`, which already holds responses
    for the full 720. The surviving ids are unchanged and the raw file stays
    complete -- it is the evidence for the exclusion.

    R1 chose dropping over re-keying (their reply §3.2): re-keying requires
    adjudicating a real ambiguity, and any adjudication is contestable by a
    reader for the same reason the scenario is. Dropping needs no adjudication.
    """
    keep, dropped = [], []
    for s in scenarios:
        away = s.form_idx == len(INSTRUCTION_FORMS)   # the "put the X away" slot
        (dropped if (s.family in CONTESTED_FAMILIES and not away) else keep).append(s)
    print(f"contested: dropped {len(dropped)} of {len(scenarios)} scenarios "
          f"({len(dropped) / max(1, len(scenarios)):.1%}) from "
          f"{sorted({d.core_id for d in dropped})}")
    return keep


def audit_report(scenarios: List[Scenario]) -> Dict[str, object]:
    """Everything a reviewer needs to believe the set has no shortcut."""
    rep: Dict[str, object] = {
        "n_scenarios": len(scenarios),
        "n_cores": len({s.core_id for s in scenarios}),
        "by_k": dict(Counter(s.k for s in scenarios)),
        "by_noun": dict(Counter(s.noun for s in scenarios)),
        "by_receptacle": dict(Counter(s.receptacle for s in scenarios)),
        "by_family": dict(Counter(s.family for s in scenarios)),
        "by_difficulty": dict(Counter(s.difficulty for s in scenarios)),
        "by_form_idx": dict(Counter(s.form_idx for s in scenarios)),
    }

    # Position of the correct answer, per k. Must be exactly 1/k.
    pos: Dict[int, Dict[int, int]] = {}
    for s in scenarios:
        pos.setdefault(s.k, Counter())[s.true_idx] += 1
    rep["true_idx_by_k"] = {k: dict(v) for k, v in sorted(pos.items())}

    # Colour of the correct answer. Must be flat across the six colours.
    rep["true_color"] = dict(Counter(s.candidates[s.true_idx].color for s in scenarios))

    # Length shortcut: how often is the target the longest description? At
    # chance this is 1/k. Ties are counted as not-longest, so this is the
    # conservative direction.
    longest: Dict[int, List[int]] = {}
    for s in scenarios:
        lens = [len(c.desc.split()) for c in s.candidates]
        hit = int(lens[s.true_idx] > max(l for j, l in enumerate(lens) if j != s.true_idx))
        longest.setdefault(s.k, []).append(hit)
    rep["target_is_longest_rate"] = {
        k: {"observed": round(float(np.mean(v)), 4), "chance": round(1.0 / k, 4)}
        for k, v in sorted(longest.items())}

    # A trivial "always answer 0" baseline, stated explicitly so the report can
    # quote the floor the tiers must beat.
    rep["always_zero_accuracy"] = {
        k: round(sum(1 for s in scenarios if s.k == k and s.true_idx == 0)
                 / max(1, sum(1 for s in scenarios if s.k == k)), 4)
        for k in sorted({s.k for s in scenarios})}
    return rep


def _no_detectable_prior(counts: Dict, what: str, alpha: float = 0.01) -> None:
    """Chi-square goodness-of-fit against uniform.

    The unfiltered build is exactly balanced by round-robin construction, so
    "max - min <= 1" was the natural check. Dropping a *non-random* subset
    (§ drop_contested) makes exact balance unattainable -- 60 removals leave
    k=2 at 108/112 -- and tightening or loosening a fixed count threshold to
    accommodate that would be choosing the guard to fit the data.

    So the guard now states what actually matters: an adversary looking at the
    answer key must not be able to *detect* a prior. Chi-square scales with n
    and answers exactly that question. A dataset that is exactly balanced
    trivially passes.
    """
    from scipy.stats import chisquare

    obs = np.array(list(counts.values()), dtype=float)
    if obs.sum() == 0:
        raise AssertionError(f"{what}: no observations")
    p = float(chisquare(obs).pvalue)
    dev = float(np.max(np.abs(obs / obs.sum() - 1.0 / len(obs))))
    assert p > alpha, (f"{what}: detectable prior (chi2 p={p:.4f}, max deviation "
                       f"{dev:.3f} from {1.0 / len(obs):.3f}) {dict(counts)}")


def assert_no_shortcuts(rep: Dict[str, object]) -> None:
    """Fail the build rather than ship a dataset with a detectable prior."""
    for k, counts in rep["true_idx_by_k"].items():  # type: ignore[union-attr]
        assert len(counts) == k, f"k={k}: correct answer never lands on some index"
        _no_detectable_prior(counts, f"k={k} position")
    _no_detectable_prior(rep["true_color"], "colour")  # type: ignore[arg-type]
    for k, d in rep["target_is_longest_rate"].items():  # type: ignore[union-attr]
        # Allow 10 points over chance before calling it a length shortcut.
        assert d["observed"] <= d["chance"] + 0.10, \
            f"k={k}: target is longest {d['observed']:.3f} vs chance {d['chance']:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--variants", type=int, default=VARIANTS_PER_CORE_PER_K)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--pilot", type=int, default=0,
                    help="write only N scenarios, stratified over k and family "
                         "(the Lab-3 gradient check)")
    ap.add_argument("--audit", default=None,
                    help="data/r2_core_audit.json from scripts/r2_audit.py; "
                         "every scenario from a core that failed the human "
                         "audit is dropped. Ids of survivors are unchanged, "
                         "so the existing raw responses still join.")
    ap.add_argument("--drop-unaudited", action="store_true",
                    help="also drop cores nobody has audited yet (strict mode "
                         "for the final dataset)")
    ap.add_argument("--drop-contested", action="store_true",
                    help="drop scenarios whose instruction is ambiguous "
                         "(already_satisfied + a receptacle-naming form). "
                         "Ids of survivors are unchanged, so the existing raw "
                         "responses still join.")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    if args.drop_unaudited and not args.audit:
        raise SystemExit('--drop-unaudited requires --audit')
    # Always expand the full core set first so ids stay aligned with the raw
    # response table; every exclusion below is a filter over assigned ids.
    scenarios = build(variants=args.variants, seed=args.seed)
    if args.audit:
        kept = audited_core_ids(list(CORES), args.audit, args.drop_unaudited)
        scenarios = apply_audit(scenarios, kept)
    if args.drop_contested:
        scenarios = drop_contested(scenarios)

    if args.pilot:
        # Stratify: walk the full set taking every m-th item, which preserves
        # the k / family / noun balance because build() emits in that order.
        m = max(1, len(scenarios) // args.pilot)
        scenarios = scenarios[::m][:args.pilot]
        args.out = args.out.replace(".jsonl", f"_pilot{args.pilot}.jsonl")
        args.report = args.report.replace(".json", f"_pilot{args.pilot}.json")

    rep = audit_report(scenarios)
    assert_no_shortcuts(rep)

    print(f"scenarios     {rep['n_scenarios']}  from {rep['n_cores']} cores")
    print(f"by k          {rep['by_k']}")
    print(f"by noun       {rep['by_noun']}")
    print(f"by receptacle {rep['by_receptacle']}")
    print(f"by family     {rep['by_family']}")
    print(f"by difficulty {rep['by_difficulty']}")
    print(f"true_idx/k    {rep['true_idx_by_k']}")
    print(f"true colour   {rep['true_color']}")
    print(f"longest-desc  {rep['target_is_longest_rate']}")
    print(f"always-0 acc  {rep['always_zero_accuracy']}")

    if args.check_only:
        print("\ncheck-only: nothing written")
        return 0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = save_scenarios(args.out, scenarios)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {n} scenarios -> {args.out}")
    print(f"wrote audit report  -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
