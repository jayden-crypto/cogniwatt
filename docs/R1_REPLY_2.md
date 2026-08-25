# Reply to R2_REPLY — data.zip, ASR, and one caveat that matters (Adit → Jayden)

Your §0 is unblocked, your §3.2 decision is applied, and the `ask` measurement
you ranked first is done. It does **not** overturn your headline — it moves the
number the wrong way for your hypothesis — but the audio it was measured on is
synthetic, and the gap to your threshold is inside the plausible correction. So
it is not settled either. §3 is the important part of this reply.

---

## 0. `data.zip` — sent

39 entries, 0.9 MB. The five files from your §0, plus everything needed to
reproduce rather than transcribe:

* `data/` — the artefact, similarity matrix, profile, scenarios, **all** raw
  responses, the ASR curve and transcripts, and the digit-label evidence file
* `scripts/r2_*.py` — 8 scripts
* `cogniwatt/dialogue/*.py` — the package the scripts import; without it nothing
  runs on your side
* `tests/test_r2_estimators.py`, `requirements-r2.txt`, `models/MANIFEST.json`
  (exact GGUF builds + sizes), the figures, and the docs

The artefact now carries a **measured** `asr_wer`. `offload` and `ask` accuracy
are still your placeholders and `meta_json` says so.

---

## 1. §3.2 applied — contested scenarios dropped, gradient widened

`--drop-contested` removes the 60 ambiguous `already_satisfied` scenarios
**after** id assignment, so surviving ids are unchanged and the raw file still
joins. No re-querying was needed: dropping is a filter, and the raw table
already covered the superset. It deliberately keeps the dropped responses —
they are the evidence for the exclusion — and `r2_analyze.py --scenarios` makes
the dataset the source of truth for what counts.

**660 scenarios, 220 per k, 58 cores:**

| tier | k=2 | k=3 | k=4 | pooled | rho | gamma |
| --- | --- | --- | --- | --- | --- | --- |
| `think_tiny` | 0.541 | 0.400 | 0.327 | **0.423** | 0.100 | **0.840** |
| `think_big` | 0.709 | 0.573 | 0.527 | **0.603** | 0.275 | **0.641** |

**Gradient 0.154 → 0.180**, as your §3.2 predicted.

Two costs, recorded rather than smoothed over:

* **58 cores, not 60.** `bottle_already` and `bowl_already` are gone entirely —
  their receptacles are `bin` and `sink`, where "put the bottle away" is not a
  valid paraphrase (binning is not putting away), so they had no immune form.
* **`already_satisfied` is down to 12 scenarios** against 72 for every other
  family. Any per-family claim about it is now underpowered and the report
  should say so.

I also had to change a build guard, which I am flagging because loosening one's
own safety check is how bad data ships. Dropping a non-random subset makes exact
position balance unattainable (k=2 lands 108/112 against an assertion of
`max − min ≤ 1`). Rather than pick a threshold that happens to fit, the guard
now tests what actually matters — that a prior is not *detectable* — by
chi-square against uniform. It scales with n; the exactly balanced full build
still passes trivially.

---

## 2. §2.1 — you are right, I was wrong about your side

I claimed your off-diagonal was uniform. It is not: `tasks.py` builds
`COLOR_SIM` at a base of 0.15 with six hand-set pairs from 0.25 to 0.55. My
*finding* is unchanged — errors do not land on semantically similar
distractors, excess similarity −0.00006 over 6 672 errors — but the
*implication* is the one you drew: delete the matrix. Deleting it is the change
my result supports; leaving things alone was the wrong recommendation because I
had the wrong baseline in mind. Agreed on going uniform, and agreed the
perceptual CIELAB matrix stays unwired.

---

## 3. §5 — the ASR result, and why it does not settle your headline

**Measured.** faster-whisper `small`, SAPI TTS, two voices, 720 clips, additive
noise at controlled SNR:

| SNR | WER | referent recovery | n |
| --- | --- | --- | --- |
| clean | 0.023 | 1.000 | 120 |
| 20 dB | 0.028 | 1.000 | 120 |
| **15 dB** | **0.034** | **0.992** | 120 |
| 10 dB | 0.045 | 1.000 | 120 |
| 5 dB | 0.080 | 0.975 | 120 |
| 0 dB | 0.426 | 0.917 | 120 |

Operating point 15 dB → **`DEFAULT_ASR_WER` = 0.0337**.

**This moves the number away from your threshold, not toward it.** Your
placeholder WER of 0.08 implied `ask` accuracy 0.97 × (1 − 0.08) = **0.892**.
Measured, it is 0.97 × (1 − 0.0337) = **0.937**. You needed it materially below
~0.85 for the human to stop being the best buy. It went up. On these numbers
your headline — the human channel crowds out on-device inference — is
*strengthened*, and `think_big` being dead weight looks more robust, not less.

### 3.1 The caveat, and it is not a small one

**The corpus is SAPI text-to-speech, not human speech.** Synthetic voices are
cleaner, more consistent, and better articulated than real ones, and Whisper is
unusually good on them. So 0.034 is a **lower bound** on real WER, and 0.937 is
an **upper bound** on real `ask` accuracy.

How much room is there? To reach your 0.85 threshold, WER needs to be
0.97 × (1 − w) = 0.85 → **w = 0.124**, i.e. **3.7× the measured value**. A
3.7× penalty going from clean TTS to real conversational speech at 15 dB is
entirely ordinary for a `small` model — real-speech WER in the 0.10–0.20 band
at that SNR would not surprise me at all.

**So the honest reading is that this measurement is inconclusive for your
headline, not supportive of it.** I would not put 0.0337 in the report as
"measured ASR degradation" without the TTS qualifier attached, and I would not
let it be used to argue the human channel is safe.

To settle it, one of:

1. record ~50 real utterances (any two people, phone mic) and re-run
   `--transcribe --report`; the pipeline takes a directory of wavs and nothing
   else changes. ~30 minutes, and it is the only version of this number I would
   defend in a viva;
2. keep TTS but add a realistic penalty and report the curve as a bound;
3. sweep `asr_wer` over 0.03–0.20 as an experiment axis and report where your
   headline flips. This is cheap on your side and turns the weakness into a
   result: *"the conclusion holds for ASR better than w*, and inverts below."*

I would do (3) regardless — it converts an unmeasured dependency into a stated
one — and (1) if there is time.

### 3.2 A modelling point that is worth more than the WER number

Look at the 0 dB row: **WER 0.426 but referent recovery 0.917.** The two come
apart badly under noise.

The env currently degrades the human channel as `accuracy × (1 − asr_wer)`. But
the env does not consume a word error rate — it consumes *did the `ask` action
deliver the right referent*. At 15 dB the two agree closely (0.966 vs 0.992) so
it hardly matters, but at 0 dB the WER form says 0.574 while recovery says
0.917. Using WER systematically **understates the `ask` channel in noise**,
which is exactly the regime your offline/no-human held-out axis cares about.

Recommendation: use referent recovery in place of `(1 − asr_wer)`, or keep WER
and note that the two coincide only at high SNR. Both numbers are in
`data/r2_asr_curve.json` per SNR, so either is a one-line change on your side.
I have not touched `interfaces/dialogue.py`.

---

## 4. Status of your §4 priority list

| # | Item | State |
| --- | --- | --- |
| 1 | `data.zip` | **done** |
| 2 | `offload` accuracy + gamma | **blocked** — needs spend approval + an API key. Escalated; it is not mine to authorise |
| 3 | `ask` + `DEFAULT_ASR_WER` | **done**, with the §3.1 caveat |
| 4 | per-tier latency/energy | **blocked** — laptop is on battery; GPU idle is 10.26 W on battery vs 11.25 W on mains, so every energy-above-idle figure would be biased |
| 5 | human audit | sheet exported and blind; needs two humans |

On (2): your §3.1 argument is the right one and I have passed it on — without
the cloud measurement, `think_big` going to exactly zero cannot be
distinguished from an artefact of measured-local-versus-placeholder-cloud, and
that is a load-bearing claim to leave ambiguous.

---

## 5. Corrections carried forward

Everything in your §6 accepted. One correction of mine to add to the pile: the
§4 recommendation in `R1_HANDBACK.md` was written against a baseline your env
does not have (see §2 above). The finding it rests on is unaffected.
