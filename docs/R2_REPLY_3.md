# R2 reply 3 — everything merged, and one number is now the whole project (Jayden → Adit)

**Paste this whole file to your Claude as the opening prompt.** Reads cold.

`data.zip` received, all 39 entries merged. `tests/test_env.py` **13/13**,
`tests/test_r2_estimators.py` **17/17** (after one fix, §1). The drop-contested
rebuild and the ASR measurement both landed and both are used below.

Your §3 was the important part of your reply and you were right that it is not
settled. It is settled now, and not in the direction either of us expected: the
ASR uncertainty turns out not to matter, because the conclusion never depended
on `ask` being good. It depended on `think_big` being able to do the job, and it
cannot.

---

## 1. One bug in the delivery — your test file is stale

`tests/test_r2_estimators.py` shipped 14/17 here. Two were `No module named
'scipy'` (our env, fixed). The third is real:

```
FAIL test_dataset_has_no_position_or_colour_shortcut: k=2 position prior Counter({1: 112, 0: 108})
```

Your §1 says you replaced the `max − min ≤ 1` guard with a chi-square
detectability test. **You updated `scripts/r2_build_dataset.py` (the
`_no_detectable_prior` helper at line ~247) but not the test**, which still
asserts the old strict form. The data is fine — 108/112 gives χ² p = 0.787, and
your own build guard passes it.

We fixed it on our side by making the test call the same chi-square logic, so
there is one definition of "no shortcut" rather than two that drift. **Please
sync your copy**, or the next person to run your suite will think the dataset is
broken. Flagging rather than silently patching because it is your file.

---

## 2. The result: `think_big` is dead, and it is not close

Trained our canonical configuration on your delivered artefact, 4 seeds:

| constants | PPO | greedy | tiny | **big** | cloud | ask | on-device share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R1 placeholders | 0.904 | 0.901 | 0.26 | 0.03 | 1.45 | 0.32 | 14 % |
| **R2 measured** | 0.907 ± 0.002 | 0.904 | 0.24 | **0.00** | 1.55 | 0.44 | **11 %** |

`think_big` usage is **exactly zero** on every seed. Not "rare" — never
selected, on any device, in any episode.

### 2.1 Why: it cannot reach a usable confidence, at any k

An agent has to commit to an irreversible `place`. Below ~0.85 posterior that is
a bad bet (the reward is +10/−10). We ran your artefact through the actual
belief filter and asked what confidence each tier reaches after *unlimited*
queries — the saturation ceiling:

| k | `think_big` 1-query acc | **ceiling** | reaches 0.85? |
| --- | --- | --- | --- |
| 2 | 0.709 | 0.705 | **no** |
| 3 | 0.573 | 0.543 | **no** |
| 4 | 0.527 | 0.499 | **no** |

**There is no candidate count at which the 3B tier can support a confident
delivery decision.** Not "it is expensive", not "the cloud is cheaper" — it
cannot get there at any price, so its cost never enters the decision. That is
why usage is 0.00 rather than 0.05.

For contrast, cheapest total reward cost to reach 0.85 at k=3:
`offload` **1.673** (3 queries) · `ask` **2.228** (1 query) ·
`think_tiny` **∞** · `think_big` **∞**.

---

## 3. Your §3.1 TTS caveat — real, but it does not affect the conclusion

We did your recommendation (3) and swept `asr_wer` from the measured 0.0337 out
to 0.40, well past the 0.124 you calculated as our flip threshold:

| asr_wer | ask acc | ask cost to 0.85 | big cost | cloud cost | cheapest |
| --- | --- | --- | --- | --- | --- |
| **0.0337** (measured) | 0.937 | 2.228 | ∞ | 1.673 | offload |
| 0.080 | 0.892 | ∞ | ∞ | 1.673 | offload |
| 0.124 (your threshold) | 0.850 | ∞ | ∞ | 1.673 | offload |
| 0.200 | 0.776 | ∞ | ∞ | 1.673 | offload |
| 0.400 | 0.582 | ∞ | ∞ | 1.673 | offload |

**No flip anywhere.** Degrading the human does not rescue on-device inference,
because on-device inference cannot reach the threshold either way. `ask` stops
being able to reach 0.85 above WER ≈ 0.05 (γ_ask = 0.15 makes it effectively
one-shot), and after that the cloud is simply the only channel that works.

**Practical consequence for you: deprioritise recording real speech (your §3
option 1).** It was the right instinct and we would have taken it, but the
number it would refine cannot change the conclusion. Spend the time on §5
instead.

One correction of ours, for the pile: our first pass at this sweep ranked
channels by *information per unit reward cost* and concluded `think_big` beats
`ask` everywhere — which contradicted the trained policies. The metric was
wrong: a channel that saturates below the target can never reach it, so its
per-query efficiency is irrelevant. Cost-to-reach-a-threshold is the
decision-relevant quantity. Noted in `scripts/r1_asr_sensitivity.py`.

## 3.1 Your §3.2 (referent recovery vs WER) — accepted

You are right that the env consumes "did `ask` deliver the right referent", not
a word error rate, and that `(1 − wer)` understates the channel in noise
(0 dB: 0.574 by WER vs 0.917 by recovery). We are switching to recovery. At our
operating point it is a 2.6-point change (0.966 → 0.992) so it moves nothing
today, but it is the correct form and the divergence is large exactly where our
degraded-condition held-out axis lives. Thank you for both numbers being in the
curve file — it is a one-line change as you said.

---

## 4. What this does to the report

Our headline was *"the human channel crowds out on-device inference"*. Your
measurement sharpens it into something stronger and less comfortable:

> **At measured accuracy, a 3B Q4 model cannot reach a usable confidence on this
> grounding task at any candidate count, so it is dominated regardless of what
> else is available.** The human is not crowding it out; nothing is. It never
> had a regime.

That is a real, defensible finding for an Edge AI report, and it is grounded in
your measurement rather than our placeholders. But it makes one number
load-bearing for the entire project, which brings us to §5.

---

## 5. The `offload` measurement is now the critical path. It is $4

Everything above is measured-local against **placeholder** cloud. We cannot
distinguish these two readings:

* **(a) real finding** — edge-class models genuinely cannot ground referring
  expressions well enough to act on, and a home robot must offload or ask;
* **(b) artefact** — `think_big` at a measured 0.603 is being compared against
  an *invented* cloud accuracy of 0.88, and the gap is our fiction.

The whole report now turns on which of these it is. **Please escalate the spend
approval again**, with that framing: it is ~$4 batched, the backend is written,
and it decides whether our central result is a finding or a modelling artefact.
If approval genuinely will not come, tell us and we will report the result
conditionally with the placeholder flagged in every figure caption — but that is
a much weaker paper.

---

## 6. A new question worth ~30 minutes: is it the model or the quantisation?

`think_big` is Llama-3.2-3B-Instruct **Q4_K_M**. Before we write "3B-class
models cannot do this", we should know whether we are measuring the *model* or
the *4-bit quantisation*. These support very different claims:

* "edge-class LLMs cannot ground referring expressions" — a statement about model
  scale, interesting and citable;
* "Q4_K_M destroys grounding capability in a 3B model" — a statement about
  deployment practice, arguably *more* useful to an Edge AI audience, and it
  would mean the tier is recoverable by shipping a better quant.

You have ~1 GB of headroom (you measured 2.29 GB resident of 4.0 GB). Cheapest
discriminating test, in preference order:

1. **Llama-3.2-3B-Instruct Q8_0** (~3.4 GB) — same model, better quant. Will not
   co-reside with anything, but you already never co-locate. If accuracy jumps,
   the story is quantisation.
2. **Qwen2.5-1.5B-Instruct Q8_0** (~1.6 GB) — a middle point on scale at a good
   quant, to separate scale from quant.

A single k=3 pass on the 660-scenario set at N=1 repeat is enough to see a
20-point move; no need for the full N=10 protocol unless it looks interesting.
Same prompt template, same parser, same permutations. If Q8 lands near 0.60 as
well, the scale claim is safe and we say so with evidence.

---

## 7. Priorities

| # | Item | Why |
| --- | --- | --- |
| **1** | `offload` accuracy + gamma (~$4) | decides whether §2 is a finding or an artefact — this is the project |
| **2** | Q8 sanity check (§6) | decides whether the claim is about scale or quantisation; ~30 min, no cost |
| 3 | per-tier latency/energy on mains | report cost table; low policy impact now that `think_big` is unused |
| 4 | human audit of the cores | ground truth; the drop-contested fix already landed so it is not blocking |
| ~~5~~ | ~~real-speech ASR~~ | **dropped** — §3 shows it cannot change the conclusion |

Nothing in `cogniwatt/env/`, `agents/`, or `ved.py` changed. Your artefact schema
was exactly right and merged without a single edit to the interface.
