# R1 handback — dialogue + hardware measurements (Adit → Jayden)

**Mirror of `docs/R2_HANDOFF.md`, pointed the other way.** Written to be read
cold. Everything below is either a measured number with its provenance, or an
explicit statement that something is *not* measured.

---

## 0. The one-paragraph version

The dialogue pipeline is built end to end and the two **local** tiers are
measured on real models: 14 400 queries, 100 % parse rate, both tiers fully
GPU-resident. `data/dialogue_measured.npz` is ready to train on. The cloud tier
and the `ask` channel are **not** measured and still carry your placeholders —
which matters more than it sounds, see §3. Three findings change assumptions
baked into the current env, and one is a flaw in my dataset rather than in your
model. The hardware profile has a real idle baseline but no per-tier energy
yet: the laptop is on battery and I refuse to publish a battery-measured
anchor.

---

## 1. Files to merge

| File | What it is | State |
| --- | --- | --- |
| `data/dialogue_measured.npz` | the dialogue artefact | **ready** — 2 of 4 tiers measured |
| `data/r2_color_similarity.npy` | 6×6 over `COLORS`, symmetric, diag 1.0 | **ready** — but read §4, you probably should not use it |
| `data/r2_rtx3060_profile.json` | hardware profile | **partial** — idle only, see §6 |
| `data/r2_scenarios.jsonl` | 720 grounding scenarios | ready, audit pending |
| `data/r2_raw_responses.csv.gz` | every raw response | ready — all numbers derive from this |
| `docs/r2_measurements.md` | full method + deviations | ready |

Nothing under `cogniwatt/env/**`, `cogniwatt/agents/**`, `cogniwatt/analysis/**`
or `cogniwatt/interfaces/ved.py` was touched. `tests/test_env.py` untouched and
**13/13 green**; my own `tests/test_r2_estimators.py` is **17/17**.

I did not edit the three constants in `interfaces/dialogue.py` either — the
artefact supersedes them at load time, so editing them as well would create two
sources of truth.

---

## 2. The measured numbers

Both local tiers, one process, shared candidate permutations, temperature 0.7,
N = 10 repeats, 720 scenarios, **parse rate 100 %**.

| tier | model | k=2 | k=3 | k=4 | pooled | 95 % CI |
| --- | --- | --- | --- | --- | --- | --- |
| `think_tiny` | Qwen2.5-0.5B-Instruct Q8_0 | 0.529 | 0.396 | 0.317 | **0.414** | [0.374, 0.456] |
| `think_big` | Llama-3.2-3B-Instruct Q4_K_M | 0.671 | 0.542 | 0.492 | **0.568** | [0.500, 0.636] |
| *chance* | | 0.500 | 0.333 | 0.250 | 0.361 | |

Accuracy falls monotonically with k for both tiers, as the env assumes.
Intervals are **cluster bootstrap over the 60 authored cores**, not over rows —
scenarios from one core are permutations of one contrast and are not
independent. The naive Wilson interval is 15–25 % narrower and would be an
overclaim.

**Evidence decay:**

| tier | rho | gamma | 1/rho | 1/(1−gamma) |
| --- | --- | --- | --- | --- |
| `think_tiny` | 0.106 [0.079, 0.132] | **0.834** [0.801, 0.870] | 9.47 | 6.01 |
| `think_big` | 0.310 [0.252, 0.367] | **0.605** [0.549, 0.664] | 3.23 | 2.53 |

Method: ICC(1) by one-way ANOVA across repeats within a scenario → `n_eff(n) =
n / (1 + (n−1)ρ)` → gamma by least squares against your geometric model. The
ICC estimator was validated against a beta-binomial generator with rho known in
closed form and recovers it to within 0.03 over rho ∈ {0.15…0.75}
(`test_icc_recovers_known_rho`). I did that before spending anything, because
gamma is fitted from an estimated rho and an unvalidated rho makes the headline
result unfalsifiable.

---

## 3. Three things that change your assumptions

### 3.1 gamma ranks the tiers backwards — do not use it alone

Measured gamma is **higher for the weaker model** (0.834 vs 0.605). Ranked by
gamma, the 0.5B is the better evidence source. It is not: its accuracy is 0.414
against a 0.361 chance floor. It is more variable because it is closer to
random, and random answers look independent. The 3B is more self-consistent
(rho 0.31), so repeating it yields less that is new.

gamma measures **variability, not information**. The env already composes them
correctly — accuracy enters the likelihood, gamma enters the weight — so
nothing in the code needs changing. But no sentence in the report may quote
gamma on its own, and the intuition in §6.2 of the original handoff ("gamma
should be higher for bigger models") is not what the data does.

This also breaks the diagnostic in §11 ("gamma ≈ 1 everywhere ⇒ temperature too
high"). Here the near-1 gamma belongs to the *weakest* tier at an ordinary
temperature, and lowering the temperature made that tier worse, not better.

Your placeholder ordering (`THINK_TINY 0.45 < THINK_BIG 0.55`) is also
contradicted: measurement gives 0.834 > 0.605. **The cloud tier is unmeasured**,
so the specific question of where `offload` sits is still open.

### 3.2 The measured accuracy gradient is compressed — your retrain will over-value offload

Placeholder vs measured:

| tier | placeholder | measured | gap to next tier (placeholder → now) |
| --- | --- | --- | --- |
| `think_tiny` | 0.45 | **0.414** | tiny→big: 0.27 → **0.154** |
| `think_big` | 0.72 | **0.568** | big→offload: 0.16 → **0.312** |
| `offload` | 0.88 | *not measured* | |
| `ask` | 0.97 | *not measured* | |

The 3B is ~15 points below where you assumed. Because `offload` is still a
placeholder, the big→offload gap has roughly **doubled**. If you retrain now,
the policy will see an artificially cheap-looking jump in competence from going
to the cloud and will over-offload. That is an artefact of mixing a measured
tier with an unmeasured one, not a result.

My recommendation: either retrain and treat the action mix as provisional, or
wait for the cloud measurement (§5). If you do retrain now, the meta block
records exactly which tiers are measured — `tiers_measured` and
`tiers_inherited_from_r1_placeholders` — so the figure caption can say so.

### 3.3 A flaw in my dataset, not your env

`already_satisfied` is the only family where the 3B does **worse** than the
0.5B (0.750 vs 0.681 error) and worse than chance. Split by instruction form,
`think_big`:

| form | error |
| --- | --- |
| "put the {noun} {prep} the {receptacle}" | 0.875 |
| "could you put the {noun} …" | 0.833 |
| "the {noun} needs to go …" | 0.722 |
| **"put the {noun} away"** | **0.417** |

Every other family moves only 0.402 → 0.333 on that same split.

The cause is genuine ambiguity. In *"put the mug on the shelf"*, when a
distractor is described as already on the shelf, "the mug on the shelf" parses
as an identifying description as readily as a destination. The
`already_satisfied` construction is uniquely exposed because its distractors
are *defined* as sitting at the named destination.

60 of 720 scenarios (8.3 %) are contested. Excluding them: `think_big`
0.568 → 0.603, `think_tiny` 0.414 → 0.423, gradient spread 0.154 → **0.180**.
The bias runs *against* the stronger model — only a model good enough to
entertain the second reading is misled by it — so it flattens exactly the
gradient the project depends on.

I have **not** silently re-keyed those scenarios. The human audit is the gate on
ground truth, and correcting keys using model errors would defeat its purpose,
since the models are what the dataset exists to judge. The six cores are flagged
for priority review. Fix options are in `docs/r2_measurements.md` §3c; re-running
both tiers afterwards costs ~25 min of local GPU and no money.

---

## 4. The similarity matrix — you probably should not wire it in

`data/r2_color_similarity.npy` exists and is well-formed (6×6 over `COLORS` in
your exact order, symmetric, diagonal 1.0). Two caveats, and together they say
"leave `COLOR_SIM` alone for now":

1. **The empirical arm is flat, and it has to be.** §6.4 asks for the matrix to
   come from "when the model is wrong, which distractor does it pick". In my
   dataset colours are assigned round-robin and never mentioned in the
   descriptions — the answer is never recoverable from colour, by design. So a
   colour confusion matrix estimates a uniform distribution. Measured over 6 672
   errors: off-diagonal mean 0.200, CV 0.171, **flat**. That is a successful
   check that the dataset has no colour shortcut, not a measurement of colour
   confusability.
2. **Errors do not land on semantically similar distractors.** Similarity of
   the chosen distractor 0.1025 vs 0.1025 for the average distractor — excess
   −0.00006. So the premise behind similarity-weighted off-diagonal mass does
   not hold here.

What is delivered is therefore the **perceptual arm**: CIELAB distance between
the colour terms (`delivered_source: "perceptual_cielab"`). That is a real
statement about colour similarity, but it is not derived from model behaviour,
and finding (2) says model errors will not follow it. `sentence-transformers`
is not installed, so the embedding arm the handoff asked for was not run.

**Practical upshot:** your current uniform off-diagonal is empirically the
better-supported choice on this dataset. Wiring in a similarity matrix would
add structure the errors do not have.

---

## 5. What is not measured, and why

| Missing | Blocker | Cost when unblocked |
| --- | --- | --- |
| `offload` accuracy + gamma | no `ANTHROPIC_API_KEY`; spend not approved | ~$4 batched, ~1 h wall clock |
| `offload` latency distribution | same | a few cents, ~10 min |
| `DEFAULT_ASR_WER` + `ask` channel | `faster-whisper` not installed | ~30 min once installed |
| per-tier latency / energy | **laptop is on battery** | ~15 min once on mains |
| human audit of the 60 cores | needs two humans, and I authored the cores | ~40 min per annotator |

`scripts/r2_asr.py` and both Anthropic backends are written and ready; they
have simply never been run.

On the audit: `docs/r2_audit_sheet.md` is a blind sheet (shuffled, no answer
key, no colours). I deliberately did not annotate it myself — a model's
agreement with cores a model authored measures authorship memory, not
resolvability, so it would not be evidence of anything. `scripts/r2_audit.py`
records model passes separately and excludes them from the kappa.

---

## 6. Hardware profile — what is real

Measured, on mains, quiet machine:

| quantity | value |
| --- | --- |
| GPU | **RTX 3050 Laptop, 4.0 GB** (not a 3060) |
| driver | 595.79 |
| `power.limit` | not reported by this part |
| **P_idle** | **11.25 W** (IQR 11.12–11.35), median over 30 s |
| SM clock / temp at idle | 1500 MHz / 48 °C |
| `think_big` residency | **29/29 layers on GPU**, 1918 MiB weights + 112 KV + 256 compute ≈ **2.29 GB** |

`ops: {}` — per-tier latency and energy are **not** measured. The script
refused to run because the machine is on battery, which is the §7b gotcha
verbatim; GPU idle draw is already 10.26 W on battery vs 11.25 W on mains, so
every energy-above-idle figure would be biased. It needs someone to plug the
laptop in.

Two protocol notes for when it runs:

* **The card is a 3050 Laptop.** The file keeps `device: "rtx3060"` so your
  merge works and adds `device_measured`. Reconciling the anchor against the
  sim's 6 GB reference part is your call; I only claim what was under the
  sampler.
* **`nvidia-smi` does not deliver 10 Hz here.** It emits 100 ms bursts
  separated by ~0.9 s stalls — ~3.3 Hz effective — and repeats stale values
  because the counter refreshes slower than the query. Sizing bursts off the
  nominal rate would give ~6 samples per measurement, which is the
  nonsense-numbers failure §7b warns about. The script now derives burst length
  from the *observed* rate, targeting ≥ 20 samples, and records both.

Q4_K_M fits with ~1 GB headroom, so no step down the quant ladder was needed.

---

## 7. Corrections to `R2_HANDOFF.md`

1. **§5's prompt template is a broken instrument for small models.** It numbers
   options `0. 1. 2.` and asks for "the number only". Qwen2.5-0.5B answered
   **"1" on 237 of 240** prompts at temperature 0. Under 1-based labels it
   answered "1" just as often — now meaning the *first* option. It has a prior
   on the **token**, not on the option. The broken version looks healthy: 99.8 %
   parse rate and a plausible 0.358 that reads as "the tiny model is weak at
   pragmatics". Letters fix it (0.358 → 0.483). The template now uses A/B/C/D
   for **every** tier — this is not per-tier tuning; the template itself was
   wrong. Evidence kept at `data/r2_raw_responses_DIGITLABELS.csv.gz`.
2. **The Batch API cannot produce the offload latency figure.** §5 wants the
   Batch API for cost and §7b wants real round-trip latency; a batch is queued
   for minutes-to-hours, so its elapsed time measures Anthropic's queue, not a
   robot's network. Split into a batch backend (accuracy, half price) and a
   live backend (latency, n≈200).
3. **§5's `output_config` snippet is missing its wrapper.** The accepted shape
   is `{"format": {"type": "json_schema", "schema": {...}}}`.
4. **§6.2's gamma direction contradicts your own placeholder table** — see §3.1.

Two bugs of mine, for completeness, since both would have corrupted results:
`hash()` is randomised per process, so the "every tier sees the same
permutation" guarantee held within a run and failed across runs (fixed with a
blake2b `stable_seed`, regression-tested); and my saturation figure was
computed as 1/(1−rho), which is neither the ICC asymptote nor your geometric
bound (now reports both).

---

## 8. What I need from you

1. **A decision on the retrain timing** given §3.2 — the measured/placeholder
   mix will over-value offload.
2. **A second annotator** for the audit sheet, and your read on the
   `already_satisfied` fix options (§3.3).
3. **Confirmation that leaving `COLOR_SIM` uniform is fine** (§4). If you want
   the perceptual matrix wired in anyway, say so and I will note the caveat in
   the report rather than argue about it.
4. Nothing in `env/` or `ved.py` needed changing. If the retrain shows the
   action mix collapsing onto one channel, that is the §3.2 artefact, not a
   bug in your reward function.

---

## 9. Re-running any of it

```bash
python scripts/r2_build_dataset.py                    # 720 scenarios, fails on any shortcut
python scripts/r2_measure.py --tiers think_tiny think_big --repeats 10
python scripts/r2_analyze.py --raw data/r2_raw_responses.csv.gz --figures
python scripts/r2_freeze.py --analysis data/r2_analysis_real.json
python tests/test_env.py && python tests/test_r2_estimators.py
```

`r2_measure.py` is the only script that costs anything; everything downstream
reads its output, so re-analysis never re-pays for inference.
