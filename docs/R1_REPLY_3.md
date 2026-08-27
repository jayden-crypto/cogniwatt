# Reply to R2_REPLY_4 — offload measured, and it does not clear the bar (Adit → Jayden)

Plan A done. The `offload` tier is Qwen2.5-7B-Instruct Q4_K_M hosted locally,
measured on the full N=10 × 660 protocol — not the subset you budgeted for,
because it ran far faster than either of us expected. All three tiers now sit on
one llama.cpp path.

**Your §2 question has an answer, and it is the second one.** The 7B does not
clear the bar, and neither does anything else. `ask` is the only channel that
can support a confident `place()`.

---

## 1. The measurement

| tier | model | k=2 | k=3 | k=4 | pooled | rho | gamma |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `think_tiny` | Qwen2.5-0.5B Q8_0 | 0.541 | 0.400 | 0.327 | **0.423** | 0.100 | **0.840** |
| `think_big` | Llama-3.2-3B Q4_K_M | 0.709 | 0.573 | 0.527 | **0.603** | 0.275 | **0.641** |
| `offload` | Qwen2.5-7B Q4_K_M | 0.800 | 0.768 | 0.777 | **0.782** | 0.525 | **0.401** |

19 800 queries, parse rate 100 % on all three, same prompt, same parser, same
permutations, temperature 0.7. Gradient spread 0.359.

**k=3 single-query accuracy is 0.768** — inside the 0.75–0.80 band your §2 table
brackets, with a measured gamma of 0.401 sitting just above your 0.35 column.
Your table predicted a ceiling near 0.68–0.75 there. Measured: **0.731.** Your
model of the tier was right; the tier just is not good enough.

Your placeholder gamma for `OFFLOAD` (0.35) was close to the measured 0.401.
The accuracy placeholder (0.88) was ten points optimistic.

---

## 2. Ceilings — the decision-relevant number

`scripts/r2_ceiling.py` drives **your** `BeliefState` and **your** likelihood
construction, imported rather than reimplemented so it cannot drift from what
training does. 4 000 trials, 40 queries deep, weight `gamma**n` per query.

| tier | k=2 | k=3 | k=4 | reaches 0.85? |
| --- | --- | --- | --- | --- |
| `think_tiny` | 0.521 | 0.374 | 0.300 | no |
| `think_big` | 0.706 | 0.546 | 0.509 | no |
| `offload` (7B) | 0.760 | 0.731 | 0.750 | **no** |
| `ask` | 0.904 | 0.909 | 0.910 | **yes** |

> No model available to a domestic robot — 0.5B, 3B or 7B — can ground these
> referring expressions confidently enough to act on, at any candidate count,
> with unlimited queries. Only asking the person can.

**Two checks before you build a report on this.**

*It is not an impossible task.* `ask` clears 0.85 comfortably, so the
environment is solvable and the models simply cannot solve it. Had `ask` failed
too, the right conclusion would have been that my dataset is broken.

*It is not knife-edge on your 0.85.* Each tier is usable iff the threshold is at
or below its ceiling: `ask` 0.904, `offload` 0.731, `think_big` 0.509,
`think_tiny` 0.300. So for **any** threshold in `(0.731, 0.904]` — 17 points
wide — the human is the only usable channel. The conclusion survives a wide
range of assumptions about how cautious a robot should be before an
irreversible action, which is what makes it a result rather than an artefact of
one chosen constant.

---

## 3. gamma falls monotonically with scale — now a trend

0.840 → 0.641 → 0.401 across 0.5B → 3B → 7B, with rho going 0.100 → 0.275 →
0.525. Three points, one direction, opposite to what R2_HANDOFF §6.2 predicted.
A more capable model is more self-consistent, so re-querying it buys less; a
weak model looks variable because it is nearly random.

This is worth stating carefully in the report, because gamma taken alone ranks
the tiers exactly backwards — the 0.5B is the "richest" evidence source by
gamma and the worst by every measure that matters.

---

## 4. A result I did not expect: the 7B's accuracy is flat in k

`think_tiny` 0.541 → 0.327 across k. `think_big` 0.709 → 0.527. `offload`
0.800 / 0.768 / **0.777** — flat.

Adding distractors costs the small models heavily and costs the 7B nothing.
That is a *qualitative* change in failure mode. For the small models part of the
difficulty is discrimination: more candidates, more chances to be pulled onto a
distractor. For the 7B it is not — it either makes the pragmatic inference or it
does not, and the number of wrong options beside the right one barely matters.

This is the strongest evidence in the study for "the failure is pragmatics, not
perception", and it is a better sentence for an Edge AI report than any single
accuracy number. It also suggests the ceiling would not improve much by
simplifying the scene, which is the obvious objection to the §2 result.

---

## 5. Status of your §6 priorities

| # | Item | State |
| --- | --- | --- |
| 1 | `offload` accuracy + gamma via local 7B | **done** — this document |
| 2 | Q8_0 rerun of `think_big` | in progress; download is slow on this link |
| 3 | per-tier latency/energy on mains | **blocked** — laptop keeps returning to battery |
| 4 | human audit | sheet exported and blind; needs two humans |
| ~~5~~ | paid API | cancelled; backends remain in the tree behind `--cloud-api`, unused |

On (2): with §4 above, the Q8 test now asks a sharper question than when you
proposed it. If Q8 lands near 0.603, "3B-class is insufficient" is safe. If it
jumps toward 0.78, the story becomes "Q4_K_M destroys grounding in a 3B" — and
given the 7B is *also* Q4_K_M and reaches only 0.782, that would raise the
question of what a Q8 7B could do. I will not speculate further until it runs.

---

## 6. One thing I would like you to check on your side

The ceiling calculation is mine but the belief filter is yours, and the whole
§2 conclusion rests on the composition. `scripts/r2_ceiling.py` is in the zip —
please sanity-check it against `scripts/r1_saturation_check.py`. Your reply 3 §2.1
reported a `think_big` ceiling of 0.543 at k=3 from your own code; I get 0.546
from mine on the same constants. That agreement is reassuring but I would rather
you confirm it than take my word for it, given what the number is now carrying.
