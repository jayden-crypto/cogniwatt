# R2 reply 8 — delta merged, §4 was our error, and yes to the 321 re-freeze (Jayden → Adit)

**Paste this whole file to your Claude as the opening prompt.** Reads cold.

Delta merged and everything reproduces on this machine: `test_env.py` 13/13,
`test_r2_estimators.py` 18/18, `r2_audit.py --report` gives 29/60, κ 0.592, raw
0.617, 29/18/13. The artefact carries `audit`, both `subsets`, and
`asr_recovery` — `r1_merge_recovery.py` and `dialogue_measured_recovery.npz`
are deleted, we train on your file directly.

Two answers and one note.

---

## 1. §4 — the discrepancy is ours, and it is not a denominator

Your guess was that ours were on the 660 rows. They were not. We recomputed
both ways and neither repeats nor row count explains it:

| computed on | `think_big` passed / failed |
| --- | --- |
| all 10 repeats | 0.499 / 0.631 |
| first repeat only | 0.500 / 0.632 |

Both give your numbers. The actual cause: **our 0.537 / 0.623 was computed
against the two-rater verdict file**, before the third pass existed.

| verdict file | `think_big` passed / failed |
| --- | --- |
| 2-rater (39 agree) | 0.541 / 0.619 |
| **3-rater (29 agree)** | **0.500 / 0.632** |

Same method, same rows, stale verdicts. **Your numbers are the ones for the
chapter** — ours describe a 39-core subset that no longer exists. Carry the
`subsets` block's figures and drop ours entirely.

---

## 2. §3 — yes, please re-freeze at 321 everywhere

We verified your figures exactly (`think_big` 0.500 → 0.526, `think_tiny`
0.397 → 0.405, spread 0.395 → 0.386). But the framing undersells the reason, so
here is the version we would put in the chapter.

The current artefact does not merely apply two different exclusion rules — it
**scores different tiers on different items**:

| tier | scenarios in the audited set |
| --- | --- |
| `offload` | 321 |
| `think_big_q8` | 321 |
| `think_big` | **348** |
| `think_tiny` | **348** |
| common to all four | **321** |

So `think_big` and `think_tiny` are each scored on 27 scenarios `offload` was
never shown. The **gradient spread is the project's headline comparison**, and
right now it is computed across non-identical item sets: 0.395 on each tier's
own subset, 0.386 on the common 321.

The difference is small. The exposure is not: *"were your tiers scored on the
same questions?"* is the first thing a reader checks about a gradient claim, and
"no, they differ by 27 items" is a bad answer to a question that costs 0.009 of
spread to make unnecessary.

Both exclusions are independently justified — the audit by inter-annotator
disagreement, the contested forms by the receptacle-form ambiguity your own
audit then corroborated when `mug_already` came back undecidable under exactly
that form. Applying both to every tier is the defensible position.

**So: `--drop-contested` on the build, 321 everywhere, please.** Keep the 348
figures in `subsets` as the audit-only reference, the same way
`all_660_contested_dropped` is kept now. Then every number in the chapter comes
from one item set and the `n_samples` array is uniform.

---

## 3. On your §2 bug

Worth recording in the chapter's methods, not just the changelog. Positional ids
being assigned after the filter is the same failure mode we have now hit four
times between us — `lambda_cloud` absent from a stored reward config,
`net_quality` not reaching the evaluation env, the dialogue artefact not
reaching it either, and now ids misaligned against the raw table. Every one was
silent, produced plausible numbers, and was caught by a check rather than by
anything breaking.

For the record on the local side: we never went through the broken path. Our
audited-subset numbers came from joining `r2_raw_all.csv.gz` to the verdict file
directly, never through `r2_build_dataset.py`, which is why they matched yours
while the bug was live. There is no misaligned `r2_scenarios.jsonl` here to
replace; we took yours from the delta.

Your `test_audit_filter_keeps_ids_aligned_with_raw_table` is the right shape of
test — it pins the invariant against the real raw table rather than a
reconstruction, which is what makes it able to fail.

---

## 4. Status

| # | Item | State |
| --- | --- | --- |
| 6 | re-freeze on the audited subset | done — your delta |
| — | §2.1 denominator | **resolved, §1 — use yours** |
| — | 321-everywhere variant | **requested, §2** |
| — | R1 retrain on the audited artefact | **done, §5** |

After the 321 re-freeze lands we retrain once more and the measurement phase is
closed. Nothing else is outstanding on either side.

---

## 5. Retrained on your audited artefact — the result holds and sharpens

Canonical config, 4 seeds per condition, evaluated in the world each was trained
in:

| condition | PPO | `greedy_value` | `always_ask` | return gap |
| --- | --- | --- | --- | --- |
| nominal | 0.968 ± 0.001 / +0.289 | 0.964 / −0.437 | 0.968 / +0.311 | −0.023 |
| **offline + alone** | **0.451 ± 0.011 / −8.07** | 0.306 / −24.61 | 0.374 / −9.28 | **+1.216** |

**Nominal stays degenerate**, exactly as before and for the reason your
measurement established: one `ask` reaches confidence, no model tier can, so
every agent converges on "ask once" and ties at ~0.968. On-device share 8 %.

**Offline and alone, the learned policy wins decisively** — +14.5 points of
success over the strong heuristic, +7.7 over `always_ask`, and +1.2 return over
the best baseline. On-device share rises to 44 %, with `think_tiny` at 0.84
calls per episode.

Note the absolute numbers fell against the pre-audit artefact (0.612 → 0.451 in
that condition). That is your audit working as intended: the surviving 29 cores
are harder for the models (`think_big` 0.603 → 0.500), so the task got genuinely
harder once the indefensible items were removed. The *ordering* is unchanged and
the margins widened.

The 321 re-freeze will move these slightly — `think_big` 0.500 → 0.526 lifts the
weakest usable tier — so we will rerun both conditions on it and report from
that. We do not expect the direction to change.

One last thing worth saying: your side of this has been measured, cross-checked
and corrected at a standard well above what the brief asked for, and three of
the project's four most load-bearing corrections came from your measurements
contradicting our assumptions. The chapter should say so plainly.
