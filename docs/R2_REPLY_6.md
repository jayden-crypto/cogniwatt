# R2 reply 6 — your side is done; a bug on ours changed the answer (Jayden → Adit)

**Paste this whole file to your Claude as the opening prompt.** Reads cold.

All four of your outstanding items landed and all four are accepted. The
token-budget result is the one that matters and it came out the way that makes
the study strongest. Then we found a bug on our side that had invalidated the
conclusion we sent you last time — §2 — so please disregard the degeneracy
framing in `R2_REPLY_5.md` §2.

Only one item remains on the whole project, and it needs people rather than
compute.

---

## 1. Your results — accepted, nothing to push back on

**Token budget (your §1).** Reasoning does not rescue the tiers, and the 3B gets
*significantly worse* (−0.059 paired, McNemar p = 0.016). The 7B's +0.023 is
indistinguishable from zero at 27× the tokens and 68× the latency. **The
self-selected control is the best part** — 34.4 % of the time the 7B ignored the
instruction and answered directly, giving 0.811 skipped vs 0.801 reasoned, so
the gain is absent even where the model chose to reason. No separate experiment
would have produced that, and it closes the obvious reviewer objection.

So the strong form holds: *no model available to a domestic robot can ground
these expressions confidently enough to act, and not because it was not allowed
to think.*

**Q8 ablation (your §3).** 0.603 vs 0.592, p = 0.500, agreeing on 581/660 —
they make the same mistakes. "3B-class is insufficient" is safe; "Q4 destroys
grounding" is not supported. Your deployment note is a good report line: the
2.0 GB Q4 build is at least as accurate as the 3.4 GB Q8, which on a 4 GB card
is the difference between fitting and spilling for no measurable cost.

**Ceilings now agree on all four tiers.** Your `ask` k=3 ceiling of 0.945
against our 0.947 after you switched to recovery — within 0.002. Two
independent implementations, four tiers, no disagreement above 0.010. Thank you
for taking the check seriously rather than treating it as a formality.

**`asr_recovery` in the artefact** — `r1_merge_recovery.py` deleted, we read
your key directly. The freeze guard refusing `think_big_q8` was the right call.

**Energy profile merged.** Handled a schema difference silently on our side:
you ship `latency_ms: {median, iqr, p95, ...}` where
`docs/R2_HANDOFF.md` §7b specified a flat `latency_ms_median`. Yours is better —
it carries the spread — so `apply_measured_profile()` now accepts both and
honours `valid_device_anchor`, which correctly refuses the 7B. No change needed
from you.

**Your measured costs demolish our roofline on this device:**

| op | R1 roofline (modelled) | R2 measured | ratio |
| --- | --- | --- | --- |
| `think_tiny` | 87.0 ms / 7 954 mJ | **19.6 ms / 837 mJ** | 4.4× / 9.5× |
| `think_big` | 519.6 ms / 48 906 mJ | **34.0 ms / 1 675 mJ** | 15× / **29×** |

Our roofline was pessimistic by up to 29× on energy. That is a real correction
to R1's cost model and it goes in the report as such. **It does not change any
conclusion**, and that is itself worth stating: on-device inference was never
rejected for being expensive, it was rejected for being *incapable*. Making it
29× cheaper does not help a channel that cannot reach the required confidence.

---

## 2. Retract our §2 from last time — it was our bug

We told you the learned policy had collapsed onto `always_ask` and that
`greedy_value` beat it everywhere. **That was wrong**, and the cause was on our
side: `scripts/evaluate_policy.py` built its environment *without the dialogue
artefact*, so every policy trained on your measured constants was being scored
inside a world still running our old placeholders — different accuracies,
different decay, different confusion matrices.

This is the **third** instance in this codebase of a training-time setting
silently failing to reach evaluation, after `lambda_cloud` and `net_quality`. It
is now covered by the same guard.

Corrected numbers:

| condition | PPO success | greedy | PPO return | best baseline | return gap |
| --- | --- | --- | --- | --- | --- |
| nominal (λ_ask 2) | 0.967 | 0.964 | +0.266 | +0.311 | −0.045 |
| λ_ask 6 | 0.877 | 0.958 | −2.211 | −3.295 | **+1.084** |
| λ_ask 10 | 0.802 | 0.963 | −2.303 | −5.928 | **+3.624** |
| **offline + alone** | **0.612** | 0.446 | −6.338 | −9.282 | **+2.944** |

**Your degeneracy observation still holds, but only in the nominal condition.**
With a cooperative, always-available human, one `ask` reaches confidence, every
agent ties at ~0.967, and there is genuinely nothing to allocate — exactly as
you said. Once the human is priced or absent, no single action suffices, the
agent has to sequence cheap evidence under a budget, and the learned policy
wins by +1.1 to +3.6 return and by 16.6 points of success offline.

So your §1 conclusion — that rescuing this with `lambda_ask` is now a modelling
constant *chosen in light of* a measurement that closed off the alternative —
was right, and it is better supported than we could show you at the time.

**One thing you should know about `greedy_value`.** It reads
`env.oracle.accuracy()`, `env.oracle.evidence_weight()` and the true confusion
matrices directly — it is handed, for free, the constants that cost you 27 600
model queries. That makes it a Bayes-optimal myopic planner with a perfect
model, not an engineered heuristic. We have added an option to give it the
pre-measurement placeholders instead, which is the honest version of the
comparison, and we will report both.

---

## 3. What is left: the audit, and it needs two people

| # | Item | State |
| --- | --- | --- |
| 1–4 | token budget, ceilings/recovery, Q8, energy | **all done** |
| **5** | **human audit of the cores** | **the only open item on the project** |

You are right that you cannot do it and right about why. R1 will take one
pass on `docs/r2_audit_sheet.md`. We need one more reader who is not either of
us and has not seen the answer key — any coursemate will do; it is ~40 minutes.

Everything else is measured, cross-checked between two independent
implementations, and merged. Unless the audit turns up a systematic ground-truth
problem, the measurement phase of this project is finished.

---

## 4. For your chapter

Three things from our side you may want, since they are downstream of your
numbers:

1. **The roofline correction** (§1) — your energy figures replace modelled ones
   and the 29× gap is worth a line, together with the point that it changes no
   conclusion.
2. **Where the degeneracy sits** (§2) — dialogue dominates absolutely under a
   cooperative human; allocation only becomes a decision when the human is
   priced or absent. Your §1 argument, now with the numbers behind it.
3. **`greedy_value` is not a fair baseline by default** (§2, last paragraph) —
   relevant to how your accuracy constants get used downstream, and to any claim
   about what the learned policy adds.
