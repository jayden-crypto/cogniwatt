# R2 reply 5 — verified, and one measurement could rescue the whole framing (Jayden → Adit)

**Paste this whole file to your Claude as the opening prompt.** Reads cold.

`cogniwatt_full.zip` merged. Offload measurement received, your §6 verification
request done, and it passes. Then it broke something on our side — not your
fault, and §3 is the important part of this reply because there is a measurement
that would fix it properly.

---

## 1. Your §6 request — verified independently, it holds

You asked us to check `scripts/r2_ceiling.py` against our own
`scripts/r1_saturation_check.py`, given what the number now carries. Two
separately written implementations, your constants, 4 000 trials, 40 queries:

| tier | k | R1 (ours) | R2 (yours) | delta |
| --- | --- | --- | --- | --- |
| `think_tiny` | 3 | 0.373 | 0.374 | −0.001 |
| `think_big` | 3 | 0.546 | 0.546 | 0.000 |
| `offload` | 3 | 0.724 | 0.731 | −0.007 |
| `ask` | 3 | **0.947** | 0.909 | **+0.038** |

**Every model tier agrees within 0.010.** The computation is confirmed and the
§2 conclusion stands on it.

**The `ask` gap is your own recommendation, not a disagreement.** We switched to
referent recovery (0.9619) as you argued in `R1_REPLY_2.md` §3.2; your ceiling
script still uses `(1 − WER)` (0.9373). Running both through our filter: 0.907
with WER, 0.947 with recovery — exactly your figure and exactly ours. **Please
switch `r2_ceiling.py` to recovery**, so your own table matches your own advice.
It moves `ask` further above the bar, so it strengthens your §2.

Two small asks while you are in there:

* **`r2_freeze.py` should emit `asr_recovery`** as a key. We support it in
  `interfaces/dialogue.py` (optional, takes precedence over `asr_wer`) and are
  currently bridging it from `r2_asr_curve.json` with
  `scripts/r1_merge_recovery.py`, which we would like to delete.
* Your §4 flat-in-k result for the 7B (0.800 / 0.768 / 0.777) is the best
  sentence in the study. Keep it prominent in the chapter.

---

## 2. What your measurement did to our side

Retrained on the full three-tier artefact, 6 seeds:

| split | PPO | `greedy_value` | **`always_ask`** | channel mix |
| --- | --- | --- | --- | --- |
| train | 0.887 ± 0.001 | 0.900 | **0.886** | ask 1.00, everything else ≈ 0 |
| held-out | 0.887 ± 0.001 | 0.882 | **0.886** | dialogue 97 % |

**The learned policy collapsed onto the `always_ask` baseline** — 0.887 against
0.886. It asks once, acts, and never runs a model. A hand-written heuristic now
beats it.

That is the honest consequence of your numbers and we are not disputing them:
one `ask` reaches ~0.95 confidence, no model tier reaches 0.85, so the optimal
policy is "ask once, deliver". But it means the decision problem is
**degenerate** — there is no allocation left to learn, and an RL project whose
optimal policy is two lines of scripted logic has an obvious problem.

We are sweeping `lambda_ask` and `patience_max` under your measured constants to
find where cognition has to be budgeted again. That works, but it rescues the
project with *our modelling constant* rather than with a measurement, which is
strictly worse. Hence §3.

---

## 3. The measurement that would fix this properly: accuracy vs token budget

**This is the highest-value thing you could run, and it is directly on-theme.**

Every number so far comes from one prompt that asks for a bare answer — roughly
5 output tokens. That measures what these models do when told to answer
immediately. It does not measure what they can do when allowed to *think*.

Your §4 finding is what makes this worth doing: the 7B's accuracy is **flat in
k**, so its failures are pragmatic inference rather than discrimination. A
pragmatic inference is exactly the kind of failure that more reasoning tokens
can fix.

**The experiment.** Same scenarios, same parser, same tiers, one new prompt
variant that permits reasoning before the answer — e.g. "Think step by step,
then give your answer as a single letter on the last line." Then measure
accuracy **and** tokens generated, per tier.

**Why it matters more than any other outstanding item.** The project is called
CogniWatt and its thesis is that thinking has a price. Right now we have one
price point per tier and no way to buy more competence with more compute — which
is precisely why the problem went degenerate. An accuracy-vs-token-budget curve
gives:

* **If the 7B (or 3B) crosses ~0.85 with more tokens** — the framing is *rescued
  by measurement*. There is now a real economic decision: spend 10× the tokens
  (hence latency, hence energy, and on a battery that is a genuine cost) or ask
  the human. That is the compute-versus-dialogue trade-off the whole environment
  was built to study, and it would be *measured* rather than assumed.
* **If it does not** — the §2 conclusion gets much stronger. "No model can do
  this, and not because we did not let it think" is a far harder claim to
  dismiss than "no model can do this at 5 output tokens", which is the obvious
  reviewer objection to the current result.

Either outcome is worth more than anything else on the list. There is no way to
lose this experiment.

**Scope.** k=3 only, N=1 repeat, all 660 scenarios, `offload` and `think_big`
tiers. That is 1 320 queries. Output goes from ~5 to ~150 tokens so it is maybe
10–20× slower per query, but it is still an overnight run at worst, and the
`--tiers` and `--repeats` flags already exist. Record mean output tokens per
tier — R1 needs it to price the extra latency and energy in `ved.py`.

If it looks promising at N=1, then N=10 on a subset for a gamma under the
reasoning prompt. Not before.

---

## 4. Everything else, ranked

| # | Item | Why here |
| --- | --- | --- |
| **1** | **accuracy vs token budget (§3)** | can rescue the project's central framing with a measurement; cannot fail |
| 2 | `r2_ceiling.py` → recovery; `r2_freeze.py` → emit `asr_recovery` (§1) | ~10 minutes, removes a bridge script and an inconsistency |
| 3 | Q8_0 rerun of `think_big` (in progress) | still wanted; §3 may reframe it — if reasoning tokens matter more than quantisation, say so |
| 4 | per-tier latency/energy on mains | needed for the cost table, and §3 makes it *more* important: if reasoning tokens buy accuracy, their energy price is the exchange rate |
| 5 | human audit | ground truth; R1 will do one pass, find a second person |

Note how §3 changes the ranking of (4). If accuracy can be bought with tokens,
then joules-per-accuracy-point becomes the central quantity of the whole
project, and it is measured on your card.

---

## 5. Our side, for context

Since your last drop: merged everything, verified your ceilings, applied both
corrections you argued for (referent recovery over WER; uniform `COLOR_SIM`
after your −0.00006 excess-similarity result), and retrained. 150+ training runs,
`test_env.py` 13/13, `test_r2_estimators.py` 17/17.

Nothing in `cogniwatt/env/`, `agents/`, `analysis/` or `interfaces/ved.py`
changed. Your artefact schema has now absorbed three tiers, a measured ASR
figure and an optional recovery key without a single interface edit, which is
the handoff contract working exactly as intended.
