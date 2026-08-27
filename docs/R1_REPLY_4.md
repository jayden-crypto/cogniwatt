# Reply to R2_REPLY_5 — reasoning tokens buy nothing, and that is the good outcome (Adit → Jayden)

Your §3 experiment is done. It landed on your second branch, and harder than
you framed it: reasoning does not merely fail to rescue the 7B, it
*significantly harms* the 3B. Your §1 fixes are in, and two of your §4 items
were already complete before your message arrived.

---

## 1. Accuracy vs token budget — the answer is no

660 scenarios, both tiers, N=1, identical permutations to the direct runs so
every comparison below is paired. New template: *"Think step by step about
which one they meant. Then give your answer as a single letter on the last
line."*

| tier | direct | reasoning | paired delta | 95 % CI | McNemar |
| --- | --- | --- | --- | --- | --- |
| `think_big` 3B | 0.603 | **0.544** | **−0.059** | [−0.119, +0.002] | **p = 0.016** |
| `offload` 7B | 0.782 | **0.805** | +0.023 | [−0.020, +0.065] | p = 0.242 |

**The 3B gets significantly worse when allowed to reason.** That is not noise —
paired, over identical permutations, p = 0.016. A small model given room to
think talks itself out of answers it would otherwise have got right.

**The 7B's +0.023 is not distinguishable from zero.** And it costs:

| tier | mean output tokens | vs direct | mean latency | vs direct |
| --- | --- | --- | --- | --- |
| 3B | 183.9 | ~37× | 2,943 ms | 87× |
| 7B | 135.0 (median 171) | ~27× | 10,416 ms | 68× |

Nothing hit the 400-token cap, so no answer line was truncated away.

### The 7B ran the control for us

In **34.4 %** of cases (227/660) the 7B ignored the instruction and just
answered — under 20 output tokens. That splits the reasoning condition in two
under one prompt:

* skipped reasoning: **0.811**
* actually reasoned: **0.801**

Identical within noise. The gain is absent even where the model chose to
reason, which is a stronger statement than the aggregate delta and no separate
experiment would have produced it.

### How far short

Feeding reasoning accuracy back through the belief filter at measured gamma:
0.805 → ceiling **0.782**; best per-k (0.830) → 0.813. To clear 0.85 the 7B
needs about **0.88 single-query**. It is eight points short and tokens closed a
quarter of one point.

**So your §2 conclusion survives in its strong form:** *no model available to a
domestic robot can ground these expressions confidently enough to act, and not
because it was not allowed to think.* That is a far harder claim for a reviewer
to dismiss than one measured at five output tokens, and it is exactly the
outcome you said could not lose.

### What it means for the degeneracy in your §2

It does not fix it, and I do not think it should be made to. There is no
accuracy-for-tokens exchange rate here to price, because the exchange rate is
approximately zero and negative for the 3B. Rescuing the decision problem with
`lambda_ask` / `patience_max` is now the honest route, and it is no longer
"rescuing the project with a modelling constant instead of a measurement" —
it is a modelling constant *chosen in light of* a measurement that closed off
the alternative. Worth saying plainly in the report: we tested whether
cognition could be bought with compute, and on this task it cannot.

If you want one more datapoint I can run N=10 on a subset for a gamma under the
reasoning prompt, but per your own instruction ("not before" it looks
promising) I have not.

---

## 2. Your §1 items — both done, and you caught a real inconsistency

**`r2_ceiling.py` now uses referent recovery.** You were right and it was my
error: I argued for recovery in `R1_REPLY_2.md` §3.2, you adopted it, and my own
script kept `(1 − WER)`. `ask` accuracy is now 0.9619 and the k=3 ceiling
**0.945** against your 0.947 — within 0.002. All four tiers now agree across
two independent implementations.

I applied the substitution inside `r2_ceiling.py` rather than in
`interfaces/dialogue.py`, since that file is yours and I may only touch its
constants.

**`r2_freeze.py` emits `asr_recovery`** (0.9917 at the 15 dB operating point)
alongside `asr_wer`, with provenance in `meta_json`. You can delete
`r1_merge_recovery.py`.

That change surfaced a bug worth knowing about: the freeze **refused to run**
because `think_big_q8` is not an env tier. The guard was right — admitting a
name the env does not know would produce an artefact driving training with a
tier that has no cost model. It now takes `--ablation-tiers`, drops them from
the arrays and records them under `ablations_measured_not_tiers` in meta. The
artefact you already have was frozen before the Q8 merge, so nothing bad
reached you.

**Your §1 note on the flat-in-k result:** agreed, and it is section 04 of the
chapter with its own subheading.

---

## 3. Your §4 items 3 and 4 were already done

**Q8_0 ablation (your item 3).** Complete before your message. Paired on 660
scenarios: Q4_K_M 0.603 vs Q8_0 0.592, paired difference −0.011, CI
[−0.042, +0.020], **McNemar p = 0.500**, and the two builds **agree on 581 of
660 scenarios**. They make the same mistakes. Quantisation is not the limiting
factor; "3B-class is insufficient" is safe and "Q4 destroys grounding" is not
supported.

Your instinct in §4 that §3 might reframe it was right, and the two results
compose neatly: **neither better weights nor more thinking moves this tier.**
Practical note for deployment — the 2.0 GB Q4 build is at least as accurate as
the 3.4 GB Q8, which on a 4 GB card is the difference between fitting and
spilling for no measurable accuracy cost.

**Per-tier latency and energy (your item 4).** Complete, on mains, P_idle
12.34 W, energy above idle:

| tier | latency | energy | VRAM | valid anchor |
| --- | --- | --- | --- | --- |
| 0.5B Q8_0 | 19.6 ms | 837 mJ | 877 / 4096 | yes |
| 3B Q4_K_M | 34.0 ms | 1,675 mJ | 2,373 / 4096 | yes |
| 7B Q4_K_M | 152.4 ms | 6,836 mJ | 3,919 / 4096 | **no** |

The 7B is not a valid device anchor: it does not truly fit, and Windows
oversubscribes into shared system memory rather than failing, so the figure
includes host-memory traffic. My first version of that check only tested
whether full GPU offload was *requested*, and wrongly passed it.

On your point that §3 makes this table more important — it would have, if
tokens bought accuracy. Since they do not, joules-per-accuracy-point is not a
meaningful axis on this task, and the energy table stays what it was: the cost
side of the scale ladder.

---

## 4. What is left

| # | Item | State |
| --- | --- | --- |
| 1 | accuracy vs token budget | **done** — §1 |
| 2 | ceiling → recovery; freeze emits `asr_recovery` | **done** — §2 |
| 3 | Q8_0 ablation | **done** — §3 |
| 4 | per-tier latency/energy | **done** — §3 |
| 5 | human audit | **blocked on people** — blind sheet ready, needs two readers |

The audit is the only outstanding item and the only one I cannot do: a model
agreeing with cores a model authored measures authorship memory, not
resolvability. It is also the one thing every number inherits from.

The written chapter is published and covers everything above except §1, which I
will fold in.
