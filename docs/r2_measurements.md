# R2 measurements — running record

Live document. Every number that reaches R1 lands here first, with its
provenance, and every deviation from `docs/R2_HANDOFF.md` is recorded in §7.

**Status: pipeline complete and validated end-to-end on synthetic data.
No real model has been queried yet** — that is blocked on the two decisions in
§8. Every number in §3 below is from the mock backend and must not be quoted.

---

## 1. What is built

| Deliverable | Artefact | State |
| --- | --- | --- |
| A — scenario dataset | `data/r2_scenarios.jsonl` (720) | **done**, audit pending |
| B — tier backends | `cogniwatt/dialogue/backends.py` | **done**, unexercised |
| C — measurement | `scripts/r2_measure.py` | **done**, mock-verified |
| C — analysis | `scripts/r2_analyze.py` | **done**, estimators unit-tested |
| D — hardware profile | `scripts/r2_profile.py` | **done**, idle baseline measured |
| D — frozen artefact | `scripts/r2_freeze.py` | **done**, drives the env |
| C — similarity matrix | `scripts/r2_similarity.py` | **done** (perceptual arm; embedding arm needs `sentence-transformers`) |
| E — clarification questions | `cogniwatt/dialogue/clarify.py` | **done** (naturalness rating + SLM arm outstanding) |
| F — failure taxonomy | folded into `r2_analyze.py` | **done** |
| C — ASR | `scripts/r2_asr.py` | **not started** |
| — model fetch | `scripts/r2_get_models.py` | **done** |
| — core audit harness | `scripts/r2_audit.py` | **done**, human passes outstanding |

Gates: `tests/test_env.py` 13/13 (untouched, R1 owns it),
`tests/test_r2_estimators.py` 15/15.

---

## 2. Dataset (Deliverable A)

720 scenarios expanded from **60 hand-authored cores** = 6 nouns × 10 pragmatic
families, one core per cell. A core is one commonsense contrast: a target
description plus ≥3 distractors, each individually a defensible non-answer.
Scenarios are colour/position/phrasing permutations of a core.

Why cores rather than flat scenarios:

* the audit unit is the core, so auditing 60 contrasts certifies all 720 items;
* k ∈ {2,3,4} forms of a core share a target and rationale, so "accuracy falls
  with k" is measured on the *same* inference rather than on three separately
  authored sets of unknown relative difficulty;
* scenarios from one core are **not independent**, so every interval in §3 is a
  cluster bootstrap over `core_id` rather than over rows.

### Shortcut audit (`data/r2_scenarios_report.json`, enforced at build time)

| Property | Result |
| --- | --- |
| scenarios / cores | 720 / 60 — 240 per k |
| noun balance | 120 each × 6 |
| family balance | 72 each × 10 |
| receptacle | shelf 192, drawer 180, table 144, bin 108, sink 96 |
| correct-answer position | exactly uniform: k=2 120/120, k=3 80/80/80, k=4 60×4 |
| correct-answer colour | exactly uniform: 120 each × 6 |
| target is longest description | k=2 0.533 (chance 0.500), k=3 0.358 (0.333), k=4 0.267 (0.250) |
| "always answer 0" baseline | exactly 1/k |

The build **fails** rather than writes if any of these drift, so a dataset with
a detectable prior cannot silently reach a measurement run.

Colour carries zero information by construction — it is spliced in at
generation time and the authored descriptions never mention one. This is
deliberate: colour is an arbitrary label in the env, so a scenario solvable
from colour alone would measure the wrong thing.

### Audit (Deliverable A's defensibility step)

The handoff asks for ≥100 hand-audited items and an inter-annotator agreement
figure. Because the audit unit is the core, auditing all 60 cores certifies the
whole set — a smaller job than 100 loose scenarios, and a stronger claim.

**Harness built** — `scripts/r2_audit.py`:

* `--export` writes `docs/r2_audit_sheet.md`, a genuinely blind sheet: every
  core shown at full width (target plus all three distractors, shuffled under a
  fixed seed so all annotators see an identical sheet), answer key withheld.
* `--annotate <name>` collects a pass; `--report` computes raw agreement,
  Fleiss' κ, and per-core verdicts, then writes `data/r2_core_audit.json`.
* `r2_build_dataset.py --audit data/r2_core_audit.json` drops failed cores
  **before** expansion, so a bad core contributes zero scenarios.

Verdicts distinguish three failure kinds, because they need different fixes:
`disagree` (annotators split — drop the core), `key_wrong` (annotators agree
with each other but not with the authored answer — the key is wrong),
`undecidable` (an annotator could not choose at all).

**Still outstanding: the human passes.** Two or three people need to fill in
the sheet. Nothing else in Deliverable A is blocking.

> **On model annotation.** `--kind model` is supported but model annotations are
> reported separately and are *excluded from κ by construction*. The cores were
> model-authored, so a model's agreement with the key measures authorship
> memory, not resolvability. Reporting it as inter-annotator agreement would be
> exactly the indefensible number this audit exists to prevent.

### Adversarial review (done — a review pass, not an independence claim)

All 60 cores were reviewed against one question: *is any distractor arguably
also correct?* Two real defects found and fixed:

| Core | Defect | Fix |
| --- | --- | --- |
| `book_already` | target was "a book **lying open** on the arm of the sofa" — "open" reads as *in use*, which is the `state_blocks` inference and competes with this core's own one | "a book left **face-down** on the arm of the sofa" |
| `mug_own` | distractor "a mug with a **colleague's** name" is office-coded and jars with a household scene | "a **housemate's** name" — same ownership inference, in-setting |

One noted and kept: `bowl_afford`'s target ("light enough to lift onto the high
shelf") half-states its own answer. That is intrinsic to affordance items and
the distractors are properly ruled out, so it stands — but it is the first core
to drop if the audit flags it.

Every scenario still carries `audit: "unaudited"` until the human passes land.

---

## 3. Pipeline validation (SYNTHETIC — do not quote)

The estimators were validated against a generator with a known answer before
any inference was paid for. `MockBackend` is beta-binomial:

    theta_m ~ Beta(alpha, beta)   per scenario, mean = target accuracy
    correct_i | theta_m ~ Bernoulli(theta_m)

for which the intra-class correlation is exactly `rho = 1/(1 + alpha + beta)`.
A model's competence on a *particular* referring expression is fixed — it
misreads the same expression the same way — and re-querying resamples only the
Bernoulli, not `theta_m`. That is the evidence-saturation mechanism, written
down generatively.

`test_icc_recovers_known_rho` confirms the analysis recovers rho to within
0.03 at rho ∈ {0.15, 0.35, 0.55, 0.75}. This is the load-bearing test of the
workstream: gamma is fitted from an estimated rho, so an unvalidated rho makes
the headline result meaningless.

Mock run (720 scenarios × 3 tiers × 10 repeats = 21 600 responses):

| tier | target acc | measured | target rho | measured rho | fitted gamma |
| --- | --- | --- | --- | --- | --- |
| think_tiny | 0.45 | 0.424 | 0.55 | 0.506 | 0.418 |
| think_big | 0.70 | 0.669 | 0.42 | 0.403 | 0.515 |
| offload | 0.88 | 0.896 | 0.68 | 0.686 | 0.259 |

think_tiny reads low on both because its 6 % format-violation rate is scored as
wrong (correctly) and adds repeat-independent noise (which depresses rho).

Cluster bootstrap vs naive Wilson: the clustered interval is consistently
wider, as it must be — `test_cluster_bootstrap_is_wider_than_wilson` asserts
≥1.5× on a heavily clustered synthetic case. Quoting the naive interval would
be the easiest way to report a false precision.

---

## 4. Hardware (Deliverable D) — real measurement

Measured on the actual machine, on mains:

| quantity | value |
| --- | --- |
| GPU | **NVIDIA GeForce RTX 3050 Laptop GPU, 4.0 GB** |
| driver | 595.79 |
| `power.limit` | not reported by this part (`[N/A]`) |
| **P_idle** | **11.25 W** (IQR 11.12–11.35), median of 30 s |
| SM clock at idle | 1500 MHz |
| GPU temp at idle | 48 °C |

Two findings that change the §7b protocol:

1. **The card is a 3050 Laptop, not a 3060.** The handoff anticipated this and
   asks only that the measured part be named precisely. The profile JSON keeps
   `device: "rtx3060"` (the sim's device-profile name, so R1's merge works) and
   adds `device_measured: "NVIDIA GeForce RTX 3050 Laptop GPU"`. Reconciling
   the anchor against the sim's 6 GB reference part is R1's call.
2. **The sampler does not run at 10 Hz.** `nvidia-smi -lms 100` emits 100 ms
   bursts separated by ~0.9 s stalls, giving ~3.3 Hz effective, and consecutive
   samples repeat because the underlying counter refreshes slower than the
   query. Sizing measurement bursts off the nominal 10 Hz would leave ~6
   samples per burst — precisely the "nonsense numbers" failure §7b warns
   about. `r2_profile.py` now derives burst length from the *observed*
   effective rate, targeting ≥20 samples per burst, and records both rates.

---

## 3A. ALL THREE TIERS — the headline result

`offload` is **Qwen2.5-7B-Instruct Q4_K_M, hosted locally** (R1 reply 4): a
model too large to reside on a 4 GB device, which is what the tier means. Every
tier now runs on one llama.cpp path, so a difference between them is model
scale rather than a different tokenizer, sampler and safety layer. 19 800
queries, parse rate **100 %** on all three.

| tier | model | k=2 | k=3 | k=4 | pooled | rho | gamma |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `think_tiny` | Qwen2.5-0.5B Q8_0 | 0.541 | 0.400 | 0.327 | **0.423** | 0.100 | **0.840** |
| `think_big` | Llama-3.2-3B Q4_K_M | 0.709 | 0.573 | 0.527 | **0.603** | 0.275 | **0.641** |
| `offload` | Qwen2.5-7B Q4_K_M | 0.800 | 0.768 | 0.777 | **0.782** | 0.525 | **0.401** |
| *chance* | | 0.500 | 0.333 | 0.250 | 0.361 | | |

Gradient spread **0.359**.

### gamma falls monotonically with scale — three points, not two

0.840 → 0.641 → 0.401. With a third tier this is a trend rather than an
accident, and it is the opposite of what R2_HANDOFF §6.2 predicted. The
mechanism is now unambiguous: a more capable model is more *self-consistent*
(rho 0.100 → 0.275 → 0.525), so re-querying it buys less. A weak model looks
variable because it is nearly random, and random answers are independent.

R1's placeholder for `OFFLOAD` was gamma 0.35 against a measured 0.401 — close.
The accuracy placeholder of 0.88 against a measured 0.782 was ten points
optimistic.

### The 7B's accuracy is flat in k, and the smaller models' is not

`think_tiny` falls 0.541 → 0.327 across k, `think_big` 0.709 → 0.527, but
`offload` is flat: 0.800 / 0.768 / 0.777. Adding distractors costs the small
models heavily and costs the 7B essentially nothing.

That is a qualitative change in failure mode, not just a quantitative one. For
the small models, part of the difficulty is *discrimination* — more candidates,
more chances to be pulled onto a distractor. For the 7B it is not: it either
makes the pragmatic inference or it does not, and how many wrong options sit
beside the right one barely matters. This is the sharpest evidence in the study
for "the failure is pragmatics, not perception".

---

## 3F. Reasoning tokens — the models were not being rushed

R1 reply 5 §3: every number so far came from a prompt asking for a bare letter,
about five output tokens. That measures what these models do when told to
answer immediately, not what they can do when allowed to think — and the
obvious reviewer objection to §3B is "you never let them reason."

Same scenarios, same parser family, same permutations, one new template:
*"Think step by step about which one they meant. Then give your answer as a
single letter on the last line."* 660 scenarios × 2 tiers, N=1.

| tier | direct | reasoning | paired delta | 95 % CI | McNemar |
| --- | --- | --- | --- | --- | --- |
| Llama-3.2-3B Q4_K_M | 0.603 | **0.544** | **−0.059** | [−0.119, +0.002] | **p = 0.016** |
| Qwen2.5-7B Q4_K_M | 0.782 | **0.805** | +0.023 | [−0.020, +0.065] | p = 0.242 |

> **Reasoning significantly *harms* the 3B and does nothing reliable for the
> 7B — at 70–90× the latency.** The 3B result is not noise: p = 0.016 on a
> paired test over identical permutations. A small model given room to reason
> talks itself out of answers it would otherwise have got right.

### The price of thinking

| tier | mean output tokens | vs direct | mean latency | vs direct |
| --- | --- | --- | --- | --- |
| 3B | 183.9 | ~37× | 2,943 ms | 87× |
| 7B | 135.0 (median 171) | ~27× | 10,416 ms | 68× |

Zero responses hit the 400-token cap, so nothing was truncated before its
answer line and the measurement is not distorted by the ceiling on output.

### An internal control the 7B supplied by itself

In **34.4 %** of cases (227 of 660) the 7B ignored the instruction to think and
simply answered — under 20 output tokens. That splits the reasoning condition
in two, with the same prompt and the same scenarios:

* accuracy when it **skipped** reasoning: **0.811**
* accuracy when it **did** reason: **0.801**

Identical within noise. The gain is not merely small, it is absent even when
the model chooses to reason. This is a within-condition control that no
separate experiment would have produced.

### It is not a near miss

Feeding the reasoning accuracy back through the belief filter at the measured
gamma: 0.805 pooled gives a ceiling of **0.782**, and even the best per-k
figure (0.830) reaches only 0.813. Both short of 0.85.

To clear the threshold at its measured gamma, the 7B would need roughly
**0.88 single-query accuracy**. It sits at 0.782–0.805. Reasoning moved it
+0.023. The gap is about eight points and tokens closed a quarter of one.

> **The §3B conclusion now reads: no model available to a domestic robot can
> ground these expressions confidently enough to act — and not because it was
> not allowed to think.** That is a much harder claim to dismiss than one
> measured at five output tokens.

---

## 3E. Quantisation ablation — it is scale, not Q4

R1 reply 3 §6 asked whether `think_big` at 0.603 is a statement about *3B-scale
models* or about *what Q4_K_M does to a 3B*. Those support very different
claims, so it had to be settled before writing either.

Llama-3.2-3B-Instruct **Q8_0** (near-lossless), same model, same prompt, same
parser, **same permutations**, same temperature, full N=10 × 660 protocol.

| | k=2 | k=3 | k=4 | pooled | 95 % CI |
| --- | --- | --- | --- | --- | --- |
| Q4_K_M | 0.709 | 0.573 | 0.527 | **0.603** | [0.534, 0.671] |
| Q8_0 | 0.691 | 0.555 | 0.532 | **0.592** | [0.521, 0.662] |

Because both ran on identical permutations this is a **paired** comparison,
which is far more powerful than noting that the intervals overlap:

* paired difference **−0.0106** (Q8 marginally *lower*)
* 95 % CI clustered on core: **[−0.042, +0.020]** — includes zero
* discordant pairs: 43 where Q4 was right and Q8 wrong, 36 the other way
* **McNemar exact p = 0.500**
* the two quants **agree on 581 of 660 scenarios (88 %)**

> **Quantisation is not the limiting factor.** Q8_0 and Q4_K_M are
> statistically indistinguishable on this task, and they largely make the *same
> errors* — so the failures are a property of the model, not of numerical
> precision. "3B-class models are insufficient for this grounding task" is the
> safe claim; "Q4 destroys grounding" is not supported.

This also strengthens §3B: the 3B's ceiling of 0.546 at k=3 is not an artefact
of an aggressive edge quant, and shipping a better quant would not recover the
tier.

A useful side note for the report's deployment story: Q4_K_M is 2.0 GB and
Q8_0 is 3.4 GB, and the smaller one is at least as accurate here. On a 4 GB
card that is the difference between a comfortable fit and spilling into shared
memory, for no measurable accuracy cost.

---

## 3D. What scale actually buys — Deliverable F, all three tiers

Accuracy by pragmatic family, first repeat, sorted by what scaling 0.5B → 7B
buys:

| family | 0.5B | 3B | 7B | gain |
| --- | --- | --- | --- | --- |
| `recency` | 0.278 | 0.639 | 0.889 | **+0.611** |
| `already_satisfied` | 0.333 | 0.583 | 0.833 | +0.500 |
| `state_blocks` | 0.403 | 0.417 | 0.833 | +0.430 |
| `condition` | 0.375 | 0.431 | 0.792 | +0.417 |
| `proximity` | 0.611 | 0.750 | 0.972 | +0.361 |
| `salience` | 0.528 | 0.764 | 0.819 | +0.291 |
| `affordance` | 0.403 | 0.583 | 0.694 | +0.291 |
| `ownership` | 0.417 | 0.694 | 0.708 | +0.291 |
| `purpose_fit` | 0.347 | 0.528 | 0.611 | **+0.264** |
| `exclusion` | 0.458 | 0.625 | 0.708 | +0.250 |

The split is not random. The five families scale helps most —
`recency`, `already_satisfied`, `state_blocks`, `condition`, `proximity` — all
turn on **reading an object's current state or position** from its description:
which one was just used, which is already put away, which is in use, which is
broken, which is nearest.

The families scale helps least — `purpose_fit`, `exclusion`, `ownership`,
`affordance` — all turn on **what a thing is for, or who it belongs to, or what
a destination can accept**. These are normative and functional, not
observational.

> **Model scale buys state-tracking; it does not buy normative reasoning.**

That is also why the ceiling stalls (§3B): the residual errors of the largest
tier are concentrated in exactly the families that scale does not fix. The four
families where even the 7B stays under 0.75 are `purpose_fit` (0.611),
`affordance` (0.694), `exclusion` (0.708) and `ownership` (0.708).

### Qualitative transcripts

```
"put the book on the table"                        [purpose_fit, 7B wrong]
  A. a green book of takeaway menus              <- correct
  B. a striped book of pressed flowers, fragile  <- 7B chose
```
The same core with the options reversed produces the same error (it took the
tax-records binder over the menus), so this is not position bias — the model
genuinely does not apply "a table is being laid for a meal".

```
"put the mug in the drawer"                        [affordance, 7B wrong]
  A. a black mug, tall, with a wide flared rim              <- 7B chose
  B. a blue mug, squat and handleless, that clears the drawer lip  <- correct
```
The correct description states the constraint explicitly — *clears the drawer
lip* — and the model still ignored it.

```
"could you put the bottle on the shelf"            [affordance, 7B wrong]
  A. a blue bottle of oil that keeps at room temperature  <- correct
  B. a red bottle of milk already sweating in the warm    <- 7B chose
```
A plausible-but-wrong inference: it appears to read "sweating in the warm" as
*this one needs dealing with urgently* rather than *this one must not go on a
shelf*. The model finds a salient cue and acts on it without checking it
against what the destination can accept.

---

## 3C. Deliverable D — per-tier latency and energy (measured, on mains)

RTX 3050 Laptop 4 GB, driver 595.79, mains, ambient 24 °C, fan curve default.
**P_idle 12.34 W** (IQR 12.25–12.39). Energy is measured *above idle*: the idle
draw is what the device spends existing, and charging it to the model would
make a fast model look expensive for being measured on a machine that was
already on.

| tier | latency (median) | IQR | energy/call | IQR | VRAM used | valid anchor |
| --- | --- | --- | --- | --- | --- | --- |
| `think_tiny` 0.5B Q8_0 | 19.6 ms | [14.4, 19.8] | **837 mJ** | [725, 855] | 877/4096 | yes |
| `think_big` 3B Q4_K_M | 34.0 ms | [33.8, 34.1] | **1675 mJ** | [1630, 1742] | 2373/4096 | yes |
| `offload` 7B Q4_K_M | 152.4 ms | [148.8, 155.1] | **6836 mJ** | [6607, 7046] | 3919/4096 | **no** |

Roughly 8× the latency and 8× the energy from 0.5B to 7B — the cost side of the
scale ladder in §3A.

### Three caveats that belong in the report

**1. The 7B is not a valid device anchor, and the first version of this script
said it was.** `valid_device_anchor` originally tested only whether full GPU
offload was *requested* (`n_gpu_layers == -1`). On Windows the WDDM driver
silently oversubscribes into shared system memory rather than failing, so
llama.cpp reports "29/29 layers offloaded" while part of the model lives in
host RAM — a 4168 MiB model buffer against a 4096 MiB card. Energy measured in
that state includes host-memory traffic. The check now also requires the card
not to be at its limit (< 93 % occupancy), which correctly fails the 7B at
95.7 %. The number is still what serving that model costs *on this hardware*;
it is not "a GPU-resident 7B".

**2. `offload` energy is a SERVER-side figure, not the robot's cost.** The tier
is a locally hosted large model standing in for a network service. What it
costs the robot to offload is radio energy plus network latency, which R1
models from published figures; `interfaces/ved.py` is untouched. The 6836 mJ is
recorded because it was free to collect and gives the server side of the energy
story, not because the env consumes it.

**3. P_idle drifts with temperature, and energy is measured against it.**
Across runs: 11.25 W at 48 °C, 11.69 W at 54 °C, 12.34–12.81 W at 63 °C. Each
run takes its own baseline under matching thermal conditions, so the figures
are internally consistent — but applying a cold baseline to a warm run would
overstate energy by roughly 3–5 %. Two independent runs agreed to within 1 % on
every tier (19.8 vs 19.6 ms; 1665 vs 1675 mJ; 6900 vs 6836 mJ) despite a 0.5 W
difference in baseline, which is the best evidence that the protocol is sound.

The sampler ran at **3.38 Hz effective against 10 Hz nominal**, so bursts were
auto-sized to 5917 ms for ≥ 20 samples each — 210, 100 and 17 calls per burst
respectively. Sizing off the nominal rate would have left ~6 samples per
measurement, which is the failure §7b warns about.

---

## 3B. Saturation ceilings — no model tier can support a confident `place()`

`scripts/r2_ceiling.py`, driving **R1's own** `BeliefState` and likelihood
construction (imported, not reimplemented, so it cannot drift from what
training does). 4 000 Monte-Carlo trials, 40 queries deep, weight `gamma**n`
per query exactly as `evidence_weight` specifies.

| tier | k=2 | k=3 | k=4 | reaches 0.85? |
| --- | --- | --- | --- | --- |
| `think_tiny` | 0.521 | 0.374 | 0.300 | no |
| `think_big` | 0.706 | 0.546 | 0.509 | no |
| `offload` (7B) | 0.760 | 0.731 | 0.750 | **no** |
| `ask` (human, placeholder acc, measured ASR) | 0.904 | 0.909 | 0.910 | **yes** |

> **No model available to a domestic robot — 0.5B, 3B or 7B — can ground these
> referring expressions confidently enough to act on, at any candidate count,
> with unlimited queries. Only asking the person can. Dialogue is not a
> fallback; it is the primary channel.**

This is R1 reply 4 §2's alternative outcome, and it is now measured rather than
hypothesised.

**It is not an artefact of an impossible task.** `ask` clears the threshold
comfortably, so the environment is solvable — the models simply cannot solve it.

**It is not knife-edge on R1's 0.85 threshold either.** Each tier is usable
only if the threshold sits at or below its ceiling:

| tier | usable iff threshold <= |
| --- | --- |
| `ask` | 0.904 |
| `offload` | 0.731 |
| `think_big` | 0.509 |
| `think_tiny` | 0.300 |

So for **any** threshold in `(0.731, 0.904]` — a band 17 points wide — the human
is the only usable channel. The conclusion holds across a wide range of how
cautious you assume the robot should be about an irreversible action, which is
what makes it worth reporting rather than a coincidence of one chosen constant.

---

## 3a. REAL measurements — both local tiers, paired (superseded by §3A)

Llama-3.2-3B-Instruct **Q4_K_M**, **29/29 layers on GPU** (1918 MiB weights +
112 MiB KV + 256 MiB compute ≈ 2.29 GB against 3305 MiB free) — Q4_K_M holds,
so no step down the quant ladder was needed and `think_big` is a **valid device
anchor**. Both tiers measured in one process so they share candidate
permutations; 14 400 queries, parse rate **100 %**.

| tier | k=2 | k=3 | k=4 | pooled | 95 % CI (clustered) |
| --- | --- | --- | --- | --- | --- |
| think_tiny (0.5B Q8_0) | 0.529 | 0.396 | 0.317 | **0.414** | [0.374, 0.456] |
| think_big (3B Q4_K_M) | 0.671 | 0.542 | 0.492 | **0.568** | [0.500, 0.636] |
| *chance* | 0.500 | 0.333 | 0.250 | 0.361 | |

**Gradient spread 0.154** — passes the ≥ 0.10 gate. Both tiers fall
monotonically with k. Against the handoff's targets (0.45 / 0.70) the tiny tier
is close and the big tier is ~13 points low; the 3B is genuinely mediocre at
this task rather than under-measured (100 % parse rate, no format failures).

### Evidence decay — and why gamma alone would rank the tiers backwards

| tier | rho | gamma | 1/rho | 1/(1−gamma) |
| --- | --- | --- | --- | --- |
| think_tiny | 0.106 [0.079, 0.132] | **0.834** [0.801, 0.870] | 9.47 | 6.01 |
| think_big | 0.310 [0.252, 0.367] | **0.605** [0.549, 0.664] | 3.23 | 2.53 |

gamma **decreases** with model size. The 3B is more self-consistent (rho 0.31
vs 0.11), so repeating it yields less that is new. The 0.5B is close to random,
and random answers look independent.

> **This is the paper's sharpest methodological point.** Ranked by gamma alone,
> the 0.5B is the *better* evidence source — 0.834 against 0.605 — which is
> exactly backwards. Its repeats are independent *noise*: accuracy 0.414
> against a 0.361 chance floor. gamma measures variability, not information,
> and is only interpretable jointly with accuracy. The env already composes
> them correctly (accuracy → likelihood, gamma → weight), but no sentence in
> the report may quote gamma on its own.
>
> It also refutes the handoff's diagnostic "gamma ≈ 1 everywhere ⇒ temperature
> too high" (§11). Here the near-1 gamma belongs to the *weakest* tier at a
> perfectly ordinary temperature, and lowering the temperature made that tier
> worse, not better (§5.4).

Neither the handoff's prose (gamma rises with model size) nor R1's placeholder
ordering is supported for these two tiers: R1 has THINK_TINY 0.45 < THINK_BIG
0.55, measurement gives 0.834 > 0.605. The cloud tier is unmeasured, so the
disagreement *about offload specifically* stays open.

### Failure taxonomy and pragmatic families

| | think_tiny | think_big |
| --- | --- | --- |
| correct | 0.414 | 0.568 |
| position-first | 0.228 | 0.200 |
| position-last | 0.131 | 0.117 |
| pragmatic | 0.228 | 0.115 |
| format violations | 0.000 | 0.000 |

Scaling 0.5B → 3B buys most on `salience` (0.472 → 0.236 error), `ownership`
(0.583 → 0.306), `recency` (0.722 → 0.361) and `proximity` (0.389 → 0.250) —
families where a stated cue must be connected to a convention. It buys almost
nothing on `state_blocks` (0.597 → 0.583) and `condition` (0.625 → 0.569), and
it is *worse* on `already_satisfied` (0.681 → 0.750) — which turns out to be a
dataset flaw, not a model result. See §3c.

---

## 3d. RESOLVED — contested scenarios dropped, and the headline numbers

R1 chose **drop, do not re-key** (their reply §3.2): re-keying requires
adjudicating a genuine ambiguity, and any adjudication is contestable by a
reader for the same reason the scenario is. Dropping needs no adjudication.

`scripts/r2_build_dataset.py --drop-contested` removes `already_satisfied`
scenarios whose instruction names the receptacle — 60 of 720 — **after** id
assignment, so surviving ids are unchanged and `data/r2_raw_responses.csv.gz`
still joins. The raw file deliberately keeps the dropped responses: they are
the evidence for the exclusion. `r2_analyze.py --scenarios` makes the dataset
the source of truth for what counts.

**Final numbers, 660 scenarios (220 per k), 58 cores:**

| tier | k=2 | k=3 | k=4 | pooled | rho | gamma |
| --- | --- | --- | --- | --- | --- | --- |
| `think_tiny` | 0.541 | 0.400 | 0.327 | **0.423** | 0.100 | **0.840** |
| `think_big` | 0.709 | 0.573 | 0.527 | **0.603** | 0.275 | **0.641** |
| *chance* | 0.500 | 0.333 | 0.250 | 0.361 | | |

**Gradient spread 0.154 → 0.180**, as predicted.

Two costs of the drop, recorded rather than smoothed over:

* `bottle_already` and `bowl_already` disappear entirely (58 cores, not 60).
  Their receptacles are `bin` and `sink`, where "put the bottle away" is not a
  valid paraphrase — binning is not putting away — so they had no immune form.
* `already_satisfied` falls to 12 scenarios against 72 for every other family,
  so the family grid is no longer balanced. Any per-family claim about
  `already_satisfied` is now underpowered and should be reported as such.

**The build guard had to change, and this is worth stating explicitly.**
Dropping a non-random subset makes exact position balance unattainable (k=2
lands at 108/112). The old assertion was `max - min <= 1`. Loosening a fixed
count threshold to accommodate the drop would be choosing the guard to fit the
data, so the guard now states what actually matters — an adversary reading the
answer key must not be able to *detect* a prior — via a chi-square
goodness-of-fit test against uniform. It scales with n, and the exactly
balanced full build still passes trivially.

---

## 3c. A dataset flaw the measurement exposed

`already_satisfied` is the only family where the 3B does **worse** than the
0.5B, at 0.750 error — worse than the 0.639 chance floor. It is also the
family the handoff uses as its motivating example, so it is worth being exact
about.

Splitting that family by instruction form, for `think_big`:

| instruction form | error | n |
| --- | --- | --- |
| "put the {noun} **{prep} the {receptacle}}**" | 0.875 | 24 |
| "could you put the {noun} {prep} the {receptacle}" | 0.833 | 18 |
| "the {noun} needs to go {prep} the {receptacle}" | 0.722 | 18 |
| **"put the {noun} away"** | **0.417** | 12 |

The same split across every other family moves only 0.402 → 0.333.

**The cause is a genuine ambiguity in the instruction, not a model failure.**
In *"put the mug on the shelf"*, when a distractor is described as already on
the shelf, the string "the mug on the shelf" supports two readings:

* destination — `put [the mug] [on the shelf]` → the target is the mug *not* on
  the shelf, which is the authored key;
* identifying — `put [the mug on the shelf] [away]` → the already-shelved mug.

The second reading is defensible English. The `already_satisfied` construction
is uniquely exposed to it, because its distractors are *defined* as being at
the named destination. "put the {noun} away" names no receptacle and is immune,
which is exactly the pattern in the table.

**Cost, quantified:** 60 of 720 scenarios (8.3 %) are contested. Excluding
them: think_big 0.568 → 0.603 (+0.035), think_tiny 0.414 → 0.423 (+0.009),
gradient spread 0.154 → **0.180**. The bias runs *against* the stronger model,
because only a model good enough to entertain the identifying reading is misled
by it — so the flaw flattens precisely the gradient the project depends on.

**Not fixed unilaterally.** The audit (§4) is the human gate on ground truth,
and rewriting keys on model evidence would defeat its purpose — the models are
what the dataset is meant to judge. The six `already_satisfied` cores are
flagged for priority review with this evidence attached. The candidate fixes,
for that review:

1. restrict `already_satisfied` to the "away" form (clean, but only 4 of the 6
   cores currently set `away_ok`);
2. disambiguate the receptacle forms — "put the {noun} away, {prep} the
   {receptacle}";
3. accept both readings as correct for the contested subset and report it.

Re-running both tiers after a fix costs ~25 minutes of local GPU and no money.

---

## 3b. First-pass measurement — `think_tiny` alone (superseded by §3a)

Kept for the record; these numbers predate the `stable_seed` fix (§5.6) and are
unpaired.

Qwen2.5-0.5B-Instruct **Q8_0**, llama.cpp, **25/25 layers on GPU** (500.84 MiB
CUDA0 buffer — a valid device anchor), temperature 0.7, 720 scenarios × 10
repeats = **7 200 real queries** at ~40/s. Parse rate **100 %**.

| k | accuracy | cluster 95 % CI | chance |
| --- | --- | --- | --- |
| 2 | 0.487 | [0.421, 0.558] | 0.500 |
| 3 | 0.442 | [0.375, 0.508] | 0.333 |
| 4 | 0.292 | [0.225, 0.358] | 0.250 |
| **pooled** | **0.407** | [0.360, 0.456] | 0.361 |

Accuracy falls monotonically with k, as the env's premise requires. R1's
placeholder was 0.45 with the comment *"barely better than chance at K=3"* —
the measured 0.442 at k=3 lands almost exactly on it.

**At k=2 the tier is not informative.** It answers "A" 79.6 % of the time, and
since the dataset is position-balanced that yields chance. The signal only
appears at k=3 and k=4, where the bias is mild (−0.04 excess on index 0).

**Evidence decay:** ρ = 0.102 [0.080, 0.124], **γ = 0.839** [0.810, 0.868].
Ceilings: 1/ρ = 9.83 effective draws, 1/(1−γ) = 6.20.

> **γ must never be read without accuracy.** A high γ means repeats are nearly
> independent — not that they are informative. This model is close to a fair
> coin, and a fair coin also has γ→1. The env gets this right (accuracy enters
> the likelihood, γ enters the weight) but the report must not quote γ = 0.84
> as "this tier is a rich evidence source". It is a *noisy* source that can be
> averaged, and averaging something barely above chance converges slowly.
> This also means the handoff's diagnostic "γ ≈ 1 everywhere ⇒ temperature too
> high" is not sufficient — here γ ≈ 0.84 at temperature 0.7 reflects
> near-random answering, and *lowering* the temperature made it worse, not
> better (see §5.4).

**Failure taxonomy** (first repeat): correct 0.407, position-first 0.251,
position-last 0.128, pragmatic 0.214, format violations **0.000**.

**Error rate by pragmatic family** — this is where the real insight is:

| easiest | | hardest | |
| --- | --- | --- | --- |
| proximity | 0.431 | condition | 0.736 |
| ownership | 0.458 | purpose_fit | 0.708 |
| salience | 0.472 | recency | 0.708 |
| already_satisfied | 0.528 | state_blocks | 0.667 |

The 0.5B model does relatively best where a surface cue correlates with the
answer (proximity, ownership, set-membership) and worst where the answer needs
a genuine inference about an object's state or purpose. "The 0.5B model fails
at pragmatics, not perception" is now a measured claim rather than a slogan.

---

## 4b. Clarification questions (Deliverable E) — done

`cogniwatt/dialogue/clarify.py`. A question partitions the candidate set; the
best one maximises expected information gain, which for a yes/no question means
splitting the **belief mass** closest to half — not the candidate count.
Splitting 3-vs-1 candidates is the right question when that one candidate holds
half the probability.

**Groundedness is structural, not hoped for.** Questions are assembled from
spans lifted verbatim out of the candidate descriptions; there is no free-text
generation step, so there is no mechanism by which an object outside the scene
could appear. Measured over all 720 scenarios: **0 hallucinations, 0 colour
leaks** (`test_clarify_never_hallucinates_an_object`,
`test_clarify_never_leaks_the_colour`). Colour is deliberately stripped from
question text — "do you mean the red one?" would resolve the referent without
the human doing any disambiguating work, which is not the channel being
modelled.

### The information-theoretic ceiling

A single yes/no answer with error rate ε carries at most **1 − H(ε)** bits. At
the placeholder ASR error of 0.08 that is **0.598 bits**. So:

| k | H₀ (bits) | best question | resolves in one question |
| --- | --- | --- | --- |
| 2 | 1.00 | 0.598 (at ceiling) | 100 % |
| 3 | 1.58 | 0.541 (no balanced split exists) | 67 % |
| 4 | 2.00 | 0.598 (at ceiling, 2-vs-2) | 50 % |

**`ask` cannot be a one-shot solution above k=2.** That is not a limitation of
the generator — it is arithmetic, and it is precisely the pressure that makes
the agent's spend-vs-ask trade-off non-trivial rather than "always ask".

### Does *selecting* the question buy anything?

The honest answer is conditional, and the null result matters:

| belief | k=2 | k=3 | k=4 |
| --- | --- | --- | --- |
| **uniform** | +0.000 | +0.000 | +0.131 bits |
| **skewed** (Dirichlet 0.7, i.e. after some thinking) | +0.000 | **+0.139 (+39 %)** | **+0.203 (+59 %)** |

Under a uniform belief at k≤3, every question is equivalent *by symmetry* —
entropy-maximising selection cannot beat picking a candidate at random, and
claiming otherwise would be an artefact. At k=2 it can never win, because only
one partition exists.

The selector earns its keep **only after the agent has already spent compute to
skew the belief**. That is a direct interaction with the project's central
claim: thinking and asking are complements, not substitutes — cheap thinking
makes the subsequent question worth more.

One phrasing bug found and fixed during evaluation: a multi-element subset was
being rendered as *"Do you mean A, or B?"*, which reads as an alternative
between A and B when the semantics are set membership over {A,B} vs the rest —
a "yes" to it would have been meaningless. Now rendered *"Is it one of these:
A, or B?"*. The scoring was always correct; only the surface string was
ambiguous.

**Not yet done:** the naturalness rating (n≈50, three raters) and the
SLM-written-question arm of the comparison. The latter needs the local models.

---

## 4c. Colour similarity matrix (§6.4) — delivered, with a caveat that matters

`scripts/r2_similarity.py` → `data/r2_color_similarity.npy` (6×6 over `COLORS`
in env order, symmetric, diagonal 1.0). **Send to Jayden; do not edit
`cogniwatt/env/tasks.py` here.**

§6.4 asks for the empirical arm to come from "when a model is wrong, which
distractor does it pick?" over candidate attributes — i.e. colours. **On this
dataset that matrix is flat by construction, and it has to be.** Colour is
assigned round-robin and spliced into descriptions that never mention one; the
answer turns on commonsense inference alone. A dataset where colour predicted
the answer would measure the wrong thing entirely (§3). So a model's errors
*cannot* correlate with colour, and the empirical colour-confusion matrix
estimates a uniform distribution. Shipping that as "measured colour
similarity" would be backwards.

Three arms are therefore computed and reported separately:

| arm | status | role |
| --- | --- | --- |
| **perceptual (CIELAB)** | **delivered** | a real statement about colour similarity that does not depend on our dataset — which is what the env actually wants, a prior over which distractor a *perceptual* error lands on |
| **embedding** (MiniLM) | needs `sentence-transformers` | averaged into the delivered matrix when available; correlation with the perceptual arm reported |
| **empirical** | computed, **not delivered** | a negative control |

`striped` is a pattern, not a colour. Convention adopted and stated: modelled
as mid-grey with a 0.55 similarity penalty against every solid term, so it is
only weakly confusable with any pure hue.

**Negative controls, on mock data:**

* Empirical colour confusion is **flat** — off-diagonal mean 0.200 (= 1/5,
  exactly chance), CV 0.171. This confirms Deliverable A carries no colour
  shortcut. It is reported as a control, not as a similarity measurement.
* Description-level confusability reads **−0.0001** (n = 6672). That is the
  *correct* null: the mock backend has no semantic model, so its errors land
  uniformly. It should turn positive on real models — that is the live
  hypothesis "errors land on semantically confusable distractors", and the
  metric is now known not to produce spurious positives.

Two bugs were caught here rather than shipped: reading the raw table back from
CSV turns `display_idx` into a float column, and an `isinstance(v, int)` guard
silently dropped every row — producing an all-zero confusion matrix that then
tripped the "colour is leaking!" warning off a NaN. Both fixed; the flatness
verdict is now `None` when there is nothing to judge, so a parsing failure can
never masquerade as a finding.

---

## 5. Corrections to the handoff

1. **`output_config` shape.** §5 gives
   `output_config={"format": {...schema...}}`. The accepted shape wraps the
   schema: `{"format": {"type": "json_schema", "schema": {...}}}`. Implemented
   correctly in `backends.py`.
2. **The Batch API cannot produce the offload latency figure.** §7b asks for
   the real wall-clock round-trip latency of the API calls, and §5 asks for the
   Batch API to halve the cost. These are incompatible in one pass — a batch
   request is queued and returns in minutes to hours, so its elapsed time
   measures Anthropic's queue, not a robot's network round trip. Split into
   `AnthropicBatchBackend` (accuracy, half price) and `AnthropicLiveBackend`
   (latency, n≈200 synchronous calls, a few cents). Both recorded; the report
   will say which number came from which.
4. **§5's prompt template is a broken instrument for small models — the most
   consequential finding so far.** The template numbers options `0. 1. 2.` and
   asks for "the number only". Qwen2.5-0.5B answered **"1" on 237 of 240**
   prompts at temperature 0. Switching to 1-based labels did not help — it
   answered "1" just as often, now meaning the *first* option instead of the
   second. It was not selecting an option at all; it has a prior on the
   **token**.

   | labelling | accuracy | answer distribution |
   | --- | --- | --- |
   | 0-based digits | 0.358 | `1` × 237 / 240 |
   | 1-based digits | 0.375 | `1` × 168 / 240 |
   | **letters** | **0.483** | A 107 / B 87 / C 35 / D 11 |

   What makes this dangerous is that the broken version looks healthy: 99.8 %
   parse rate, zero errors, and a plausible 0.358 that reads as "the tiny model
   is weak at pragmatics". It would have gone into the report as a finding
   about *model capability*.

   `PROMPT_TEMPLATE` now uses A/B/C/D. This is **not** per-tier tuning — one
   template still serves every tier; the template itself was wrong. A bare
   digit under letter labelling is recorded as its own failure class
   (`digit_answer`) rather than guessed at, since coercing it would manufacture
   answers out of exactly the degenerate behaviour being detected.
   `test_prompt_uses_letters_not_digits` guards against reverting.
   The digit-label run is kept at `data/r2_raw_responses_DIGITLABELS.csv.gz`
   as the evidence.

6. **`hash()` is randomised per process, which broke the paired comparison.**
   The candidate shuffle was seeded with `abs(hash((scenario_id, repeat)))`.
   Python randomises `str` hashing per process (PEP 456), so the documented
   guarantee that "every tier sees the same permutation" held *within* a run
   and silently failed *across* runs — and a 4 GB card forces exactly that
   pattern, one model resident at a time. Tiers measured in separate
   invocations drew different shuffles, so a tier that happened to draw easier
   positions would have looked better at grounding. Replaced with a blake2b
   `stable_seed`; `test_seed_is_stable_across_processes` verifies it across
   three subprocesses. `think_tiny` was re-measured alongside `think_big` under
   the fix, since the earlier run's permutations are not recoverable.

5. **§6.2 contradicts R1's placeholder table on the direction of gamma.** The
   prose says gamma should be *higher* for bigger models; `dialogue.py` has
   `OFFLOAD = 0.35` — *below* `THINK_BIG = 0.55`. The disagreement is
   substantive: a strong model may be confidently and consistently right, which
   is high rho and therefore low gamma. `r2_analyze.py` takes no side; it
   reports the measured value and prints which prediction the data supports.

---

## 6. Deviations from the plan

| Deviation | Reason |
| --- | --- |
| Cores instead of flat scenarios | auditability, k-comparability, honest clustering — §2 |
| 720 scenarios, not 600 | headroom to drop cores that fail the audit without falling under the 200-per-k quota |
| Accuracy scores format violations as **wrong** | dropping them grades the tiny model only where it behaved |
| No "attribute confusion" class in the taxonomy | colour is uninformative by construction here, so colour-based confusion is indistinguishable from noise; confusability is measured separately on embeddings |
| Raw table may be gzipped CSV | pyarrow not installed; the analysis reads either, and the file format must never block the expensive part |

---

## 7. Numbers still to be measured

`TIER_ACCURACY`, `EVIDENCE_DECAY`, `accuracy_by_k`, `DEFAULT_ASR_WER`, the 6×6
colour similarity matrix, and every latency/energy figure. Until then
`r2_freeze.py` stamps `asr_wer_provenance: "R1 PLACEHOLDER -- not measured"`
and `tiers_inherited_from_r1_placeholders` into the artefact meta, so no
number of unknown provenance can reach the final report unlabelled.

---

## 8. Blocked on

1. **Local model stack** — `llama-cpp-python` (CUDA wheel) plus ~2.5 GB of GGUF
   downloads for Qwen2.5-0.5B-Q8_0 and Llama-3.2-3B-Q4_K_M.
2. **Anthropic API access** — no `ANTHROPIC_API_KEY` in the environment.
   Budget at Haiku 4.5, batched: ~$4 for the full 7 200-call offload pass, plus
   a few cents for the 200-call latency pass.
