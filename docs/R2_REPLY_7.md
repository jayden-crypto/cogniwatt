# R2 reply 7 — the audit is in, and it costs us half the dataset (Jayden → Adit)

**Paste this whole file to your Claude as the opening prompt.** Reads cold.

Three human annotators, blind sheet, no answer key. The gate you set in
`R2_HANDOFF.md` §4 — *"if two humans disagree on the answer, the scenario is
bad, drop it"* — removes **31 of 60 cores**. Your conclusion survives and gets
sharper; the dataset takes real damage. One action needed from you (§4).

---

## 1. Result

| | |
| --- | --- |
| annotators | 3 humans (Adit, one outside reader, annotator 3) |
| cores retained | **29 / 60 (48 %)** → **348 / 720 scenarios** |
| Fleiss κ | **0.592** (3 raters, 47 fully-rated cores) |
| raw agreement | 0.617 |
| verdicts | 29 agree · 18 disagree · 13 undecidable |

Two-annotator sensitivity, before the third pass: **39/60 retained, κ = 0.722**.
Adding a third reader cost 10 cores and 0.13 of κ. **Report both.** That drop is
a finding about item difficulty, not a reason to drop a rater.

Per-annotator, for the record:

| annotator | committed | agreed with key |
| --- | --- | --- |
| Adit | 59/60 | 0.831 |
| outside reader | 50/60 | 0.840 |
| annotator 3 | 57/60 | **0.684** |

Pairwise: Adit↔outside **0.800**, Adit↔annotator 3 0.625, outside↔annotator 3 0.638. One
careful reader resolves roughly a third of these differently from the other two,
reproduced across two separate sittings. That is not carelessness; it is
interpretive variance in the pragmatic families, and it is worth a sentence in
the chapter.

---

## 2. Your conclusion survives the clean subset, and strengthens

Accuracy on the 29 cores where **all three** humans independently agreed
(first repeat only, your protocol):

| tier | k=2 | k=3 | k=4 | pooled | was | Δ |
| --- | --- | --- | --- | --- | --- | --- |
| `think_tiny` | 0.491 | 0.371 | 0.328 | **0.397** | 0.423 | −0.026 |
| `think_big` | 0.578 | 0.491 | 0.431 | **0.500** | 0.603 | −0.103 |
| `offload` (7B) | 0.804 | 0.776 | 0.794 | **0.791** | 0.782 | +0.009 |
| `think_big_q8` | 0.570 | 0.495 | 0.374 | **0.480** | 0.592 | −0.112 |

Gradient spread **0.359 → 0.395**. Your 7B's flat-in-k result holds on the clean
subset (0.804 / 0.776 / 0.794).

**The 3B lands on exactly 0.500 on items three people found unanimous.** The 7B
is still ~9 points below the ~0.88 it needs to clear the ceiling bar. Cleaning
the dataset did not rescue either tier — it moved them *down*.

### 2.1 Human-hard and model-hard are different axes

We checked whether the models fail on the cores humans could not agree on. They
do not — if anything the reverse:

| tier | audit-passed | audit-failed |
| --- | --- | --- |
| `think_big` | 0.537 | **0.623** |
| `think_big_q8` | 0.549 | **0.703** |
| `offload` | 0.792 | 0.753 |
| `think_tiny` | 0.394 | 0.423 |

**The models are no worse — and the 3B is better — on the items humans found
ambiguous.** So model failure is not tracking item ambiguity. They fail on
expressions three independent people resolve identically. This closes the
strongest objection available to a reviewer: *"your models look bad because your
dataset is ambiguous."* It measurably is not the explanation.

This belongs in the chapter next to your flat-in-k result. Together they say the
failure is pragmatic inference on clear items, not confusion on unclear ones.

---

## 3. The damage, stated plainly

Retention by family:

| family | retained |
| --- | --- |
| `state_blocks` | 6/6 |
| `affordance`, `condition` | 4/6 |
| `already_satisfied`, `purpose_fit`, `salience` | 3/6 |
| `ownership`, `recency` | 2/6 |
| **`proximity`, `exclusion`** | **1/6** |

`proximity` and `exclusion` are effectively gone — one core each, and any
per-family claim about them is now unsupportable. `already_satisfied` was
already thinned by the contested-scenario drop and is down to 3 cores; it should
carry an explicit "underpowered, reported for completeness" note rather than a
number anyone might cite.

No core came back `key_wrong` under three annotators (the two-annotator run had
one, `bowl_own`, where both readers agreed with each other against the key — it
is now `disagree` since the third reader split it). Worth a footnote: no key was
*provably* wrong, but 31 were not defensible.

---

## 4. What we need you to do

**Re-freeze on the audited subset.** `data/r2_core_audit.json` is in the repo
with all 60 verdicts, and your own build already takes it:

```
python scripts/r2_build_dataset.py --audit data/r2_core_audit.json --drop-unaudited
python scripts/r2_analyze.py --raw data/r2_raw_all.csv.gz --scenarios data/r2_scenarios.jsonl --figures
python scripts/r2_freeze.py --analysis data/r2_analysis_real.json
```

No re-querying: your raw table already covers the superset, so this is a filter,
exactly like `--drop-contested`.

Three things to get right in the re-analysis:

1. **The cluster bootstrap is now over 29 cores, not 58.** CIs will widen
   materially and that is correct — say so rather than letting the interval
   quietly halve in width because n_rows stayed large. The rows are not
   independent; the cores are.
2. **γ needs recomputing on the retained rows.** ρ is a within-scenario
   correlation, so it survives the filter fine, but the estimate moves and the
   ICC bootstrap should also be over the 29 surviving cores.
3. **Emit both subsets in `meta_json`** — pooled-on-all-720 and
   pooled-on-audited-348 — with `audit_kappa`, `n_annotators` and
   `n_cores_retained`. R1 will train on the audited one; having both in the
   artefact means no figure can be built from the wrong subset by accident.

**Do not re-key the 31 dropped cores.** Same reasoning you gave for
`already_satisfied`: adjudicating them means choosing between readings three
people already split on, and any adjudication we make is contestable for exactly
the reason the core failed. Dropping needs no adjudication and is defensible in
one sentence.

---

## 5. On the annotator pool, for the methods section

Two of three annotators are project members. Neither had read
`data/r2_scenarios.jsonl` (which carries `true_idx`) before annotating, and the
third is outside the project entirely. State that plainly as a limitation rather
than hoping nobody asks. The mitigation worth reporting: **the outside reader
agreed with the key at 0.840, the highest of the three**, so the pool is not
obviously biased toward the authored answers.

One pass (annotator 3's) was quarantined mid-audit over a provenance doubt and then
reinstated after a redo reproduced it at 0.978 on co-committed items. The
quarantine note is in `data/r2_audit_quarantine/README.md`. Mentioning it costs
nothing and the distinction it records is worth keeping visible: a pass may be
excluded for **provenance**, never for producing inconvenient numbers.

---

## 6. Status

| # | Item | State |
| --- | --- | --- |
| 1–4 | token budget, ceilings, Q8, energy | done |
| 5 | human audit | **done — this document** |
| 6 | re-freeze on the audited subset | **you, §4** |

After §4 lands, R1 retrains on the audited artefact and the measurement phase is
closed. Nothing else is outstanding on either side.
