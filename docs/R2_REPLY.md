# R2 reply — corrections, answers, and what we still need (Jayden → Adit)

**Paste this whole file to your Claude as the opening prompt.** Written to be
read cold, mirroring `docs/R1_HANDBACK.md`. Everything below is either a number
we measured on our side, or an explicit statement that something is blocked.

Handback received and read in full. The measurement work is solid — the cluster
bootstrap over authored cores, validating the ICC estimator against a
beta-binomial generator *before* spending, and refusing to publish a
battery-measured anchor are all the right calls. Four of your findings changed
what we do. Two things in the handback are wrong about our side, and one is
blocking.

---

## 0. BLOCKING: the data files were not delivered

`docs.zip` contained **only `docs/`** — 9 markdown/HTML files, no `data/`. Every
artefact §1 lists as "ready" is missing:

```
data/dialogue_measured.npz          <- the artefact itself
data/r2_color_similarity.npy
data/r2_rtx3060_profile.json
data/r2_scenarios.jsonl
data/r2_raw_responses.csv.gz
```

**Please send `data.zip` with those five.** Also worth including
`scripts/r2_*.py` and `tests/test_r2_estimators.py` so the numbers are
reproducible on our side rather than transcribed.

Nothing below required them — we reconstructed a provisional artefact from your
§2 tables to answer your §8.1 question — but the real merge cannot happen until
they arrive.

---

## 1. Your §3.1 is correct. We verified it, and here is the clean way to show it

You argued that measured gamma ranking the tiers backwards (tiny 0.834 > big
0.605) does not require any env change, because accuracy enters the likelihood
and gamma only enters the weight. **Confirmed.** We ran your constants through
the actual belief filter (`scripts/r1_saturation_check.py`, 4000 Monte-Carlo
trials per cell, 40 queries deep) and measured each tier's *ceiling* — the
accuracy it reaches after unlimited queries:

| k | tier | 1-query acc | effective draws | **ceiling accuracy** |
| --- | --- | --- | --- | --- |
| 3 | `think_tiny` | 0.396 | 6.02 | **0.485** |
| 3 | `think_big` | 0.542 | 2.53 | **0.645** |
| 3 | `offload` *(placeholder)* | 0.880 | 1.54 | 0.888 |
| 3 | `ask` *(placeholder)* | 0.892 | 1.18 | 0.898 |

This is the argument for your §3.1 in one line: **`think_tiny` gets 6.0
effective draws to `think_big`'s 2.5 and still loses by 16 points.** Six draws of
a near-chance channel are worth less than two and a half of a decent one. Gamma
really does measure variability rather than information, exactly as you say, and
the ceiling table is the number to quote instead of gamma alone.

### 1.1 But the project's central claim is now half as strong

Our core economic tension is *"a cheap model cannot be chained up to the
competence of an expensive one."* Measured against placeholder:

| k | placeholder tiny → big ceiling | gap | measured tiny → big ceiling | gap |
| --- | --- | --- | --- | --- |
| 2 | 0.543 → 0.814 | 0.270 | 0.581 → 0.767 | **0.186** |
| 3 | 0.458 → 0.831 | 0.373 | 0.485 → 0.645 | **0.160** |
| 4 | 0.467 → 0.841 | 0.374 | 0.422 → 0.598 | **0.175** |

**The claim survives — the gap is real and the problem is not degenerate — but
it is roughly halved (−57 % at k=3).** That is a genuine result from your
measurement and it goes in the report as such. It is not a problem to fix.

---

## 2. Two things the handback gets wrong about our side

### 2.1 §4 — our off-diagonal is NOT uniform, so your finding implies a *change*

You wrote: *"your current uniform off-diagonal is empirically the better-
supported choice on this dataset."* Our env is not uniform.
`cogniwatt/env/tasks.py` ships a hand-written `COLOR_SIM` with off-diagonal
values from **0.15 to 0.55** (`red`/`striped` 0.55, `blue`/`green` 0.45,
`white`/`striped` 0.50, …), and `sample_task` passes it into every belief update
as `task.similarity`. Measured effect on the likelihood matrix: up to **0.035
per entry**.

So your result — errors do not land on semantically similar distractors, excess
similarity −0.00006 — is evidence that we should **delete that matrix**, not
leave things as they are. We are removing it and going genuinely uniform. Your
conclusion was right; it just applies to a different baseline than you assumed.

Your perceptual CIELAB matrix stays unwired, and we agree with your reasoning
for that.

### 2.2 §6 — the profile is more blocked than "partial" suggests

`ops: {}` with a real `P_idle` means the file cannot override anything: our
`apply_measured_profile()` skips any op with no `latency_ms_median`, so merging
it today is a no-op. That is the correct behaviour and your refusal to publish a
battery-measured anchor is right — we are only flagging that "partial" reads as
"some ops available" when it is actually "no ops available". Plugging the laptop
in is a 15-minute unblock and it is worth doing before the report.

Noted on the card being an **RTX 3050 Laptop 4 GB**, not a 3060, and on keeping
`device: "rtx3060"` for the merge. Reconciling that against the sim's 6 GB
reference part is ours; we will state plainly in the report that the anchor was
measured on a 3050 Laptop.

---

## 3. Answers to your §8

### 3.1 Retrain timing — **wait for the cloud measurement.** Here is the evidence

We built a provisional artefact from your §2 tables
(`scripts/r1_provisional_artefact.py`, `meta_json` records that offload and ask
are placeholders) and trained our canonical configuration on it, 3 seeds.

| constants | success | tiny | **big** | cloud | ask | on-device share |
| --- | --- | --- | --- | --- | --- | --- |
| R1 placeholders | 0.904 | 0.26 | 0.03 | 1.45 | 0.32 | 14 % |
| **R2 measured local tiers** | 0.897 | 0.39 | **0.00** | 1.37 | 0.34 | 19 % |

**`think_big` usage goes to exactly zero.** The 3B tier becomes entirely dead
weight — never selected, on any seed, on any device. `think_tiny` picks up
slightly (0.26 → 0.39, since your measured k=2 accuracy of 0.529 beats our 0.45
placeholder and gamma 0.834 makes it chainable), and overall success barely
moves.

Your §3.2 diagnosis is right and the mechanism is sharper than "over-offload":
unlimited `think_big` reaches 0.645 at k=3, while **one** placeholder cloud call
is 0.880. A 24-point deficit that no amount of local querying can close means
the policy has no reason to ever run the 3B model. We cannot tell from this
whether that is a real finding about 3B models on edge hardware or an artefact
of measured-local-vs-placeholder-cloud — and that is precisely why we are not
retraining for the report yet.

**So: the ~$4 batch for `offload` is the single highest-value thing you can
unblock.** Please push for the spend approval. Everything downstream of it is
already written.

### 3.2 Second annotator — we cannot be it, and we agree with your reasoning

Your point that a model agreeing with model-authored cores measures authorship
memory rather than resolvability is correct, and it applies to us too — R1's
side of this project is also Claude-assisted, so a pass from us would carry the
same defect and would additionally not be independent of the cores' author.
`docs/r2_audit_sheet.md` needs two humans. R1 will do one pass; find a
second person outside the project for the other.

**On `already_satisfied`: drop the 60 contested scenarios, do not re-key them.**
Your own numbers make the case — excluding them moves the gradient spread
0.154 → 0.180, i.e. it *widens* the thing the project depends on, and dropping
8.3 % of a 720-scenario set is cheap. Re-keying requires adjudicating a genuine
ambiguity you have already shown is real ("the mug on the shelf" parsing as an
identifying description), and any adjudication we make will be contested by a
reader for the same reason. Dropping needs no adjudication and is defensible in
one sentence. Record the exclusion and the reason in the report.

### 3.3 `COLOR_SIM` — agreed uniform, and we are changing our side to match

See §2.1. Your recommendation is accepted; it just requires an edit on our side
rather than inaction.

### 3.4 Env changes — confirmed, none needed

We verified the composition end to end (§1). Nothing in `cogniwatt/env/`,
`cogniwatt/agents/` or `interfaces/ved.py` needs to change for your artefact to
drive training, beyond deleting `COLOR_SIM`. If the action mix collapses onto
one channel after the full retrain, we will treat it as the §3.2 artefact until
the cloud tier is measured, as you suggest.

---

## 4. What we need, in priority order

| # | Item | Why it is ranked here |
| --- | --- | --- |
| **1** | **`data.zip`** — the five files in §0 | blocks the merge entirely |
| **2** | **`offload` accuracy + gamma** (~$4 batched) | decides whether `think_big` being dead is a finding or an artefact (§3.1) |
| **3** | **`ask` channel accuracy + `DEFAULT_ASR_WER`** | see §5 — this is the number most likely to overturn our headline |
| **4** | per-tier latency/energy, on mains | needed for the report's cost table; low policy impact since `think_big` is unused |
| **5** | human audit of the 60 cores | gates the ground truth, but the drop-them fix (§3.2) does not wait on it |

Items 2 and 3 are both "written but never run". If you can only unblock one,
make it **3**, for the reason below.

---

## 5. The measurement most likely to overturn our results — please prioritise `ask`

Our R1 headline this cycle is that **the human channel crowds out on-device
inference**. Measured on our side: removing only the human (cloud still
available) doubles the on-device share 14 % → 33 %, while removing only the
cloud barely moves it (14 % → 19 %) and leaves `think_big` at zero. The demand
goes to `ask`, not to local models. Our conclusion — *on-device reasoning earns
its place exactly when the robot is offline and alone* — rests on that.

**That entire result pivots on two numbers we invented:** `lambda_ask = 2.0` and
the `ask` channel's 0.97 accuracy. If your measured ASR pipeline puts effective
`ask` accuracy materially below ~0.85, the human stops being the best buy, local
inference re-enters the portfolio, and our headline changes shape.

So your `ask` measurement is not a loose end on your side — it is the input our
main finding is most exposed to. `faster-whisper` is a `pip install` away and you
estimate 30 minutes. Please do it before the hardware profile.

---

## 6. Corrections to `R2_HANDOFF.md` — all four accepted

1. **The digit-label prompt bias is a real bug and a good catch.** A 0.5B model
   answering "1" on 237/240 prompts at temperature 0 is a token prior, not a
   choice, and the 99.8 % parse rate hiding it is exactly the kind of failure
   that looks like a result. Letter labels for every tier is right, and it is
   not per-tier tuning. Keeping the digit-label evidence file was the correct
   instinct.
2. **Batch-vs-live split accepted** — a queued batch measures Anthropic's queue,
   not a robot's network. Batch for accuracy, live for latency.
3. `output_config` wrapper shape — accepted, our error.
4. §6.2's gamma direction — accepted, and now superseded by §1 above.

Both of your own bugs (randomised `hash()` breaking cross-run permutation
stability; saturation reported as 1/(1−rho)) were caught and fixed before they
reached a number, which is the outcome that matters.

---

## 7. Our side, for context

Since the handoff, R1 has: resolved the bimodal-convergence limitation (it was
`ent_coef`; 1/8 runs converge at 0.01 vs 16/16 annealed), added entropy
annealing, found and fixed a reproducibility break where the published headline
had been generated under a different reward function, replaced the zero-shot
transfer claim with one that is falsifiable and passes, and audited the
portfolio claim. 120 training runs, `tests/test_env.py` 13/13 green.

Two of those touch you:

* **`docs/r1_transfer_audit.md`** — our held-out axis is moving from *device* to
  *operating condition* (offline + no human), because no baseline degraded on
  the device split. Does not affect your interface.
* **`docs/r1_cloud_price.md`** — the `lambda_cloud` derivation and the
  portfolio analysis your §3.2 warning feeds into.

Neither changes anything you have built against. The interface is unchanged and
your artefact schema is exactly right.
