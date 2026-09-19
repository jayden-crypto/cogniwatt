# Reply to R2_REPLY_7 — re-frozen on the 29 cores, and the filter you asked me to run had a bug (Adit → Jayden)

Reads cold. §4 of your reply is done; the artefact is in the delta zip beside
this file. One thing you need to know before you retrain (§2), one choice I
made that you should be able to see (§3), and one small discrepancy for the
chapter (§4). Nothing is outstanding on my side after this.

---

## 1. Done, and your numbers reproduce

`data/dialogue_measured.npz` is frozen on the audited subset with all three of
your corrections:

1. cluster bootstrap over the **29 surviving cores** — intervals widened, as
   they should (`think_big` pooled ±0.07 → ±0.10);
2. γ and its ICC bootstrap recomputed on the retained rows, also over cores;
3. `meta_json` carries `audit` (κ, raw agreement, n_annotators,
   n_cores_retained, the rule) and `subsets` with pooled/per-k accuracy, γ and
   CIs for **both** `audited_348` and `all_660_contested_dropped`, with
   `trained_on: audited_348`.

Nothing was re-queried and nothing re-keyed. Every figure in your §1 came back
identical when recomputed from `data/r2_audit/*.jsonl` on this machine —
κ 0.592 on 47 cores, raw 0.617, 29/18/13; two-rater 39/60, κ 0.722, raw 0.800
including the `bowl_own` key-wrong; per-annotator 0.831 / 0.840 / 0.684;
pairwise 0.800 / 0.625 / 0.638. Your §2 accuracy table matches mine to three
decimals in every cell.

| tier | pooled | 95 % CI (29 cores) | γ | γ CI |
| --- | --- | --- | --- | --- |
| `think_tiny` | 0.397 | [0.342, 0.454] | 0.855 | [0.815, 0.906] |
| `think_big` | 0.500 | [0.397, 0.603] | 0.621 | [0.533, 0.714] |
| `offload` | 0.791 | [0.703, 0.873] | 0.444 | [0.319, 0.596] |
| `think_big_q8` (ablation, in meta only) | 0.480 | [0.369, 0.592] | 0.657 | [0.573, 0.750] |

Full write-up, family retention, the human-hard/model-hard table and the
methods-section limitation are in `docs/r2_measurements.md` **§3G**. Gates:
`tests/test_env.py` 13/13, `tests/test_r2_estimators.py` 18/18 — on a freshly
rebuilt machine (mine was reset; I rebuilt from your handback, which is a
useful reproducibility check in itself).

---

## 2. The command in your §4 would have produced wrong numbers — read this before running anything locally

`r2_build_dataset.py --audit` dropped failed cores **before** `build()`
assigned ids. Ids are positional (`sc_0041`), so with 29 cores kept, **all 348
surviving ids named a different scenario** from the one the raw table had
measured under that id — `sc_0000` became `mug_afford`; the raw table has it
as `mug_already`, a dropped core. `r2_analyze.py --scenarios` would then have
scored dropped-core responses as the audited set. No error, no warning, a
plausible 348-row result.

`--drop-contested` had always done this correctly (filter after ids, and its
docstring says why); the audit path just did not follow the same rule. Fixed:
the build now always expands all 60 cores, and every exclusion is a filter
over assigned ids. `test_audit_filter_keeps_ids_aligned_with_raw_table` pins
it against both a synthetic verdict file and the real raw table.

**If you ran `--audit` locally and have a `data/r2_scenarios.jsonl` from it,
that file is misaligned — replace it with the one in the delta.** Your §2
numbers match mine exactly, so however you produced them you did not go
through the broken path; this is about not doing so in future.

---

## 3. One choice you should be able to see: contested forms are *not* dropped on top of the audit

Your §4 command and your 348 count apply the audit alone, so that is what the
artefact does. But the shipped artefact you train on today is the 660-item set
with the contested `already_satisfied` receptacle forms removed (§3d), and
going 660 → 348 silently un-applies that rule for the local tiers. Two facts
worth having in front of you:

- **For `offload` and `think_big_q8` it is moot** — they were only ever
  measured on the 660, so their audited subset is 321 items (107 per k), audit
  *and* contested drop already. That is why `n_samples` in the artefact is
  107 for `offload` and 116 for the local tiers.
- **For the local tiers the difference is small and one-directional.**
  Dropping the contested forms too (321 for every tier): `think_big` 0.500 →
  0.526, `think_tiny` 0.397 → 0.405, spread 0.395 → 0.386. Nothing else moves.

The audit itself confirmed the contested rule's premise, incidentally:
`mug_already` was shown on the sheet with the receptacle form *"put the mug on
the shelf"* and came back undecidable. I followed your spec. If you would
rather the artefact carry 321 everywhere for consistency with what R1 has been
training on, it is a one-flag re-freeze (`--drop-contested` on the build) and
I will send it the same day — say so.

---

## 4. Small discrepancy in your §2.1, for the chapter

My audit-passed / audit-failed split (first repeat, every raw row):

| tier | audit-passed | audit-failed |
| --- | --- | --- |
| `think_tiny` | 0.397 | 0.430 |
| `think_big` | 0.500 | 0.632 |
| `think_big_q8` | 0.480 | 0.699 |
| `offload` | 0.791 | 0.773 |

Same direction as yours, same conclusion — models no worse on the ambiguous
cores, the 3B and Q8 better — but your `think_big` cells are 0.537 / 0.623 and
mine 0.500 / 0.632, so we used different denominators. Mine are the ones the
artefact's `subsets` block records; the chapter should carry one set, and I do
not mind which. If yours were on the 660 rows only, that would explain it.

---

## 5. What is in the delta (drop over the tree; paths are repo-relative)

| file | what changed |
| --- | --- |
| `data/dialogue_measured.npz` | **the artefact to retrain on** |
| `data/r2_scenarios.jsonl` | 348 audited, ids unchanged |
| `data/r2_scenarios_660_contested_dropped.jsonl` | the previous shipped set, kept under its real name |
| `data/r2_analysis_audited.json` | analysis behind the artefact (`r2_analysis_real.json` untouched = the 660 reference) |
| `data/r2_core_audit.json` | your verdicts, unchanged, plus an `agreement` block (κ etc.) the freeze reads |
| `data/r2_core_audit_2rater.json` | the two-rater sensitivity, reproducible via `r2_audit.py --report --annotators adit,annotator2 --out …` |
| `data/r2_scenarios_report.json` | shortcut guards on the 348: position/colour undetectable, target-is-longest at chance |
| `scripts/r2_build_dataset.py` | the §2 fix |
| `scripts/r2_freeze.py` | `--audit`, `--reference-analysis NAME=PATH`, `--subset-name` |
| `scripts/r2_audit.py` | writes `agreement`; `--annotators` filter (refuses to overwrite the main verdict file) |
| `tests/test_r2_estimators.py` | id-stability test; the ≥600 quota now checked on the unfiltered build, not the shipped file |
| `docs/r2_measurements.md` | §3G, plus the stale "before expansion" sentence corrected |
| `results/figures/r2_fig{1,2,3}_*.png` | regenerated on the audited subset |

Not regenerated: `docs/r2_dialogue_chapter.html` — §3G is the source for the
chapter text and I would rather you fold it in beside your §2.1 than have two
renderings drift.

Reproduce from the tree:

```
python scripts/r2_audit.py --report
python scripts/r2_build_dataset.py --audit data/r2_core_audit.json --drop-unaudited
python scripts/r2_analyze.py --raw data/r2_raw_all.csv.gz --scenarios data/r2_scenarios.jsonl --out data/r2_analysis_audited.json --figures
python scripts/r2_freeze.py --analysis data/r2_analysis_audited.json --subset-name audited_348 \
    --reference-analysis all_660_contested_dropped=data/r2_analysis_real.json \
    --audit data/r2_core_audit.json --asr-wer 0.033724146224146226
```

---

## 6. Status

| # | Item | State |
| --- | --- | --- |
| 6 | re-freeze on the audited subset | **done — this delta** |
| — | 321-everywhere variant | on request (§3) |
| — | §2.1 denominator | your call (§4) |

Retrain on it and the measurement phase is closed on both sides.
