# R2 Handoff — Conversational AI workstream (Adit)

**Paste this whole file to your Claude as the opening prompt, or point it at
`docs/R2_HANDOFF.md` once you have the repo.** It is written to be read cold —
it assumes no prior context.

---

## 0. What you are joining

**CogniWatt** is a course project for Edge AI & Robotics / Conversational AI.
A household robot in simulation is given an ambiguous spoken instruction
(*"put the mug on the shelf"*, in a house with several mugs) and a finite
hardware budget. A single RL policy decides not just what to do, but **how much
compute to spend thinking** and **when to interrupt the human** — with energy,
latency and human patience all priced in the same reward function.

**This is a two-person team.** There is no third member, so the edge-hardware
workstream is split between us by one rule: **whoever runs a model profiles it.**

| | Owner | Owns |
| --- | --- | --- |
| **R1** | Jayden (M2 Air) | RL environment, belief filter, PPO, baselines, analysis — **complete**. Plus: M2 Air power anchor, thermal fit, actuation + network models, roofline extrapolation to Jetson / Pi / Coral |
| **R2** | **Adit — you** (RTX 3060) | The dialogue side: language models, ASR, and the measured constants that make the simulation real. Plus: **per-inference latency and energy on the 3060**, captured on the same passes that measure accuracy |

You are already loading these models to measure grounding accuracy. Wrapping
those runs in a power sampler costs you almost nothing and produces the project's
second hardware anchor — which is exactly why this piece is yours rather than
Jayden's.

R1 is done and its results are in `docs/r1_report.html`. The learned policy
reaches 0.955 success against a 0.895 hand-engineered heuristic, and 0.923
zero-shot on unseen devices. **Every dialogue and hardware number behind those
results is a placeholder that gets replaced with a measurement.** That is the
entire point of the remaining work: right now the simulation is structurally
right but empirically invented.

### Your headline contribution

Everyone can measure "how accurate is a 0.5B model". The novel measurement —
the one that makes the project's core claim work — is **evidence saturation**:

> Querying the same language model twice does not give you two independent
> opinions. A quantised SLM at low temperature is nearly deterministic, and even
> when sampled its errors are *systematically correlated* — it misreads the same
> referring expression the same way. So information from a given model
> **saturates**. A cheap model cannot be chained up to the competence of an
> expensive one.

R1 models this as geometric decay: the *n*-th query to tier *t* contributes
`γ_t^(n-1)` of a full observation. Nobody has measured `γ` for quantised SLMs on
a grounding task. **You are going to.** Without saturation the whole problem is
degenerate — the optimal policy is just "spam whichever channel has the best
information-per-Joule". With it, the optimum is a portfolio, which is the
interesting result.

---

## 1. Hard rules — read before writing any code

### 1.1 The performance contract (most important thing in this document)

`DialogueOracle.sample_evidence()` is called **inside the PPO rollout loop,
millions of times per training run**. It must never call a language model.
A single 200 ms model call there turns a 10-minute training run into three
weeks.

You deliver a **precomputed artefact**: the models are queried offline, once,
and the measurements are frozen to a `.npz` file. Training reads numbers from
that file. Live model calls happen only in (a) your measurement scripts and
(b) the Tier-2 demo.

If you find yourself importing `transformers` or `anthropic` anywhere under
`cogniwatt/env/` or `cogniwatt/agents/`, stop — you have taken a wrong turn.

### 1.2 File ownership

| Path | You may |
| --- | --- |
| `cogniwatt/interfaces/dialogue.py` | **Edit the constants only** — `TIER_ACCURACY`, `EVIDENCE_DECAY`, `DEFAULT_ASR_WER`. Do not change function signatures. |
| `cogniwatt/interfaces/ved.py` | **Do not edit.** You deliver `data/r2_rtx3060_profile.json` (§7b) and Jayden merges it. One owner per file avoids merge pain. |
| `cogniwatt/dialogue/**` | **Create freely** — this is your package |
| `scripts/r2_*.py` | **Create freely** — your measurement scripts |
| `docs/r2_*.md` | **Create freely** — your reports |
| `cogniwatt/env/**`, `cogniwatt/agents/**`, `cogniwatt/analysis/**` | **Do not touch.** If you need a change here, write it down and send it to Jayden. |
| `tests/test_env.py` | **Do not edit.** You may add `tests/test_r2_*.py`. |

### 1.3 The gate

```bash
python3 tests/test_env.py
```

13 tests must stay green at every commit. One of them,
`test_dialogue_artefact_roundtrip`, is the exact contract your artefact must
satisfy — read it first, it is the clearest statement of what you owe.

---

## 2. Environment setup

You are getting the project as a **zip** (`cogniwatt_handoff.zip`), not a git
clone. Unpack it wherever you like:

```bash
unzip cogniwatt_handoff.zip -d cogniwatt && cd cogniwatt
```

Everything is there: source, tests, docs, trained policies, result CSVs and
figures. It excludes `__pycache__` and virtualenvs.

**Because there is no shared repo, agree a sync convention with Jayden before
you start** — you will both be editing the same tree on different machines and
there is no merge tooling to save you. The safe convention, given the file
ownership table above: you only ever create files under `cogniwatt/dialogue/`,
`scripts/r2_*`, `data/r2_*`, `docs/r2_*`, plus the three constants in
`dialogue.py`. Send those files back individually rather than a whole tree, and
Jayden drops them in. If either of you needs to edit anything else, say so first.

Your machine is the discrete-GPU box — the project's model rig. **Your card has
4 GB of VRAM**, which is the binding constraint on this workstream, so §5.1
below is not optional reading. Everything is chosen to fit; where it is tight,
the fallback ladder is spelled out.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy torch pandas matplotlib
pip install llama-cpp-python faster-whisper sentence-transformers soundfile anthropic
```

Existing code needs only `numpy`, `torch`, `pandas`, `matplotlib`. Keep your
extra dependencies in `requirements-r2.txt` so Jayden's Mac never has to install
CUDA-specific packages.

---

## 3. The design constraint that will break your work if you miss it

**The referring expressions must be resolvable by commonsense reasoning, and
harder models must resolve them better.**

Here is the trap. The environment's instruction is *"put the mug on the shelf"*
with three identical-looking mugs and a randomly chosen true referent. If you
hand that to a language model, **no model can beat chance** — there is genuinely
no information in the text. You would measure 33% for every tier, the accuracy
gradient would vanish, and the whole project would collapse.

So the scenarios you author must contain a **commonsense-determined** answer that
is *inferable but not trivial*:

> **Instruction:** "put the mug away"
> **Scene:** (a) a red mug on the kitchen counter, half full of coffee
> (b) a blue mug already inside the cupboard
> (c) a white mug in the dishwasher
> **Answer:** (a) — the others are already "away".

> **Instruction:** "bring me something to read"
> **Scene:** (a) a paperback on the nightstand (b) a bottle on the desk
> (c) a remote on the sofa
> **Answer:** (a)

The difficulty knob is how much inference the answer takes. Calibrate so you
land roughly at **0.5B ≈ 0.45, 3B ≈ 0.70, cloud ≈ 0.88** — if the tiny model is
at 0.9 the task is too easy and there is no economics left; if the big model is
at 0.4 it is too hard and the agent can never become confident. **Pilot 50
scenarios and check this gradient before generating the full set.** This is the
single highest-risk step in your workstream.

### The environment's exact vocabulary

Match it, so scenarios map onto R1's task generator (`cogniwatt/env/tasks.py`):

```python
NOUNS       = ("mug", "book", "bottle", "bowl", "remote", "cushion")
COLORS      = ("red", "blue", "green", "white", "black", "striped")
RECEPTACLES = ("shelf", "sink", "table", "drawer", "bin")
# instruction   = f"put the {noun} on the {receptacle}"        <- underspecified
# disambiguated = f"put the {color} {noun} on the {receptacle}" <- ground truth
```

Candidate counts are **k ∈ {2, 3, 4}**. Measure everything per-k — resolving one
of four is strictly harder than one of two, and the interface now supports
per-k accuracy.

---

## 4. Deliverable A — the scenario dataset

**File:** `data/r2_scenarios.jsonl` · **Script:** `scripts/r2_build_dataset.py`

Target **≥ 600 scenarios** (200 per k), balanced across nouns and receptacles.

```json
{
  "id": "sc_0041",
  "k": 3,
  "noun": "mug", "receptacle": "shelf",
  "instruction": "put the mug away",
  "candidates": [
    {"color": "red",   "desc": "a red mug on the kitchen counter, half full of coffee", "room": "kitchen"},
    {"color": "blue",  "desc": "a blue mug already inside the cupboard",                "room": "kitchen"},
    {"color": "white", "desc": "a white mug sitting in the dishwasher",                 "room": "kitchen"}
  ],
  "true_idx": 0,
  "rationale": "the other two are already put away",
  "difficulty": "easy"
}
```

Generate them with a strong model (Claude Opus 5 is fine here — this is authoring,
not measurement), then **hand-audit at least 100**. If two humans disagree on the
answer, the scenario is bad — drop it. A dataset whose ground truth is not
defensible invalidates every number you produce downstream. Record inter-annotator
agreement; it goes in your report.

---

## 5. Deliverable B — the three model tiers

| Tier | Constant | Model | Notes |
| --- | --- | --- | --- |
| `think_tiny` | on-device small | **Qwen2.5-0.5B-Instruct**, GGUF **Q8_0** | ~0.6 GB. This is the "INT8 on-device" tier |
| `think_big` | on-device large | **Llama-3.2-3B-Instruct**, GGUF **Q4_K_M** | ~1.9 GB — the tight one on 4 GB, see §5.1 |
| `offload` | cloud | **Claude Haiku 4.5** (`claude-haiku-4-5`) | see below |
| `ask` | the human | scripted human simulator + real ASR | §7.3 |

### 5.1 Fitting the models in 4 GB — read before installing anything

4 GB VRAM makes two of your deliverables behave differently, and conflating them
is the trap:

**For accuracy measurement (Deliverable C):** speed does not matter at all — you
are running offline batches, not real time. llama.cpp lets you offload only some
layers to the GPU (`n_gpu_layers=N`) and keep the rest on CPU. So even if a model
does not fully fit, you set `n_gpu_layers` to whatever fits and the rest runs on
CPU. It will be slower; the *outputs are identical*. Accuracy is never blocked by
VRAM. Use `n_gpu_layers=-1` (all on GPU) when it fits, and dial it down until it
loads when it does not.

**For hardware profiling (Deliverable D, §7b):** the opposite. A `think_big`
latency/energy number is only a valid *device anchor* if the model runs
**fully on the GPU** — a partial-CPU-offload measurement characterises "half a
3060 plus your CPU", which is meaningless as a device profile. So for the
profiling pass specifically, the model must fit entirely in VRAM.

**Does Llama-3.2-3B fit fully in 4 GB?** Q4_K_M weights are ~1.9 GB; CUDA context
is ~0.4 GB; the KV cache for a 150-token prompt is negligible. Total ≈ 2.4 GB —
**it fits, with headroom, as long as nothing else is resident.** The rule in the
gotchas ("never co-locate models") is therefore mandatory for you, not advisory.

**If it OOMs anyway** (driver overhead, a desktop environment eating VRAM), drop
the quant — this ladder is in *decreasing* memory:

| Quant | 3B size | Use |
| --- | --- | --- |
| `Q4_K_M` | ~1.9 GB | first choice |
| `Q4_K_S` | ~1.7 GB | if Q4_K_M won't fully load for the profiling pass |
| `Q3_K_M` | ~1.5 GB | last resort |

A lower quant is *not* a compromise — aggressive quantisation is exactly what
ships on 4 GB edge hardware, so it is arguably more realistic. **Just record the
quant you actually used** in `meta_json` and the report, because it changes both
the measured accuracy and the measured energy, and both must be traceable to the
same model build.

**Whisper and the embedding model are not a problem** — `faster-whisper small`
is ~0.5 GB and MiniLM is ~0.1 GB, each fine alone. The only tight thing is the
3B model, and only when something else is co-resident.

Run the local tiers through **llama.cpp** (`llama-cpp-python`), not raw
`transformers` — the project's claim is about *quantised on-device* models, and
GGUF Q8/Q4 is what actually ships on edge hardware. Using fp16 transformers
weights would measure a model nobody deploys.

### Why Haiku 4.5 for the cloud tier

The decay protocol (§7.2) requires querying the same model repeatedly **at a
controlled sampling temperature**. On the current frontier models
(`claude-opus-5`, `claude-sonnet-5`) the `temperature` / `top_p` / `top_k`
parameters have been **removed and return HTTP 400**. `claude-haiku-4-5` still
accepts them. That is the deciding reason — it is the tier where the measurement
is actually well-defined.

If you would rather use `claude-opus-5` for maximum capability, you can — but you
must then measure decay from repeated identical calls without temperature
control, and say so explicitly in the report.

Cost, so you can decide with real numbers: ~20k calls × (~300 in / ~20 out
tokens) ≈ 6M input + 0.4M output. Haiku 4.5 is $1/$5 per MTok → **≈ $8**, and the
**Message Batches API halves that to ≈ $4**. Opus 5 ($5/$25) would be ≈ $40, or
≈ $20 batched. Use the Batch API either way:

```python
import anthropic
client = anthropic.Anthropic()
batch = client.messages.batches.create(requests=[...])   # each item needs a custom_id
```

Poll `client.messages.batches.retrieve(id).processing_status` until `"ended"`,
then stream results. **Results come back in arbitrary order — key them by
`custom_id`, never by position.** Set `max_tokens=256`; this is classification.

Use structured outputs so you get a clean index rather than prose:
`output_config={"format": {...}}` with a JSON schema like
`{"type":"object","properties":{"choice":{"type":"integer"}},"required":["choice"],"additionalProperties":false}`.
Do **not** use assistant prefill — it returns 400 on all current models.

### Prompt template (identical across all tiers — this matters)

Any prompt difference between tiers contaminates the comparison. One template,
one parser, candidate order **shuffled per query** with the permutation recorded,
so you can detect position bias.

```
You are a home robot. The person said: "{instruction}"
Objects that could match:
0. {desc_0}
1. {desc_1}
2. {desc_2}
Which one did they mean? Answer with the number only.
```

---

## 6. Deliverable C — the measurements

**Script:** `scripts/r2_measure.py` → writes `data/r2_raw_responses.parquet`

Query every tier on every scenario **N = 10 times** at the deployment temperature
(use 0.7 for local tiers; record whatever you use). Store every raw response —
all downstream numbers are derived from this one file, so you never re-pay for
inference.

### 6.1 Accuracy → `TIER_ACCURACY` and `accuracy_by_k`

Use **only the first of the N repeats** per scenario, otherwise you contaminate
accuracy with the repeat structure you are about to measure separately.

```
accuracy[tier][k] = mean over scenarios of 1[model_choice == true_idx]
```

Report Wilson 95% confidence intervals. With 200 scenarios per k, the half-width
is about ±0.07 — adequate. Also report **position bias** (does the model favour
index 0?); if it is strong, say so, because it inflates apparent accuracy.

### 6.2 Evidence decay → `EVIDENCE_DECAY` (your headline result)

For each tier and k:

1. For scenario *m*, repeat *i*, record correctness `c[m,i] ∈ {0,1}`, `i = 1..N`.
2. Estimate the **intra-class correlation** ρ across repeats within a scenario,
   by one-way ANOVA:

   ```
   MSB = N * var_between_scenario_means      (between-scenario mean square)
   MSW = mean_m var_within_scenario_m        (within-scenario mean square)
   rho = (MSB - MSW) / (MSB + (N - 1) * MSW)
   ```

   ρ → 1 means the model always gives the same answer (no new information from
   repeating). ρ → 0 means repeats are independent draws.
3. Effective independent draws after *n* queries:
   `n_eff(n) = n / (1 + (n - 1) * rho)`
4. Fit γ by least squares over `n = 1..N`, matching the geometric model R1 uses:

   ```
   minimise  sum_n [ (1 - gamma**n) / (1 - gamma)  -  n_eff(n) ]**2
   ```

   `scipy.optimize.minimize_scalar` on (0, 1) is fine.

Report γ per tier with a bootstrap CI, plus the marginal-information curve
(`n_eff` vs `n`) as a figure — that plot **is** the contribution. Sanity check:
γ should be **higher for bigger models** (more genuine variability to harvest)
and lowest for `ask` (asking a person the same question twice tells you nothing).
If you get γ ≈ 1 everywhere, your temperature is too high; if γ ≈ 0, too low.

### 6.3 ASR → `DEFAULT_ASR_WER`

**Script:** `scripts/r2_asr.py`

1. Write the human's answers to clarification questions as short utterances
   ("the red one", "the one on the counter").
2. Synthesise audio — `piper` TTS, or macOS `say` — several voices.
3. Inject noise at **SNR ∈ {20, 15, 10, 5, 0} dB** (kitchen/TV/babble noise).
4. Transcribe with `faster-whisper` (`small` or `base` — the size that would
   plausibly run on an edge device; note which).
5. Compute **WER vs SNR**, and separately the metric that actually matters:
   **referent recovery rate** — how often the pipeline recovers the *correct
   referent* from the noisy transcript.

Pick a defensible operating point (15 dB is a reasonable living room) and report
that WER as `DEFAULT_ASR_WER`. Deliver the whole curve as a figure — R1 can then
sweep ASR quality as an experiment axis.

### 6.4 Semantic confusability → the similarity matrix

**Deliverable:** `data/r2_color_similarity.npy`, a **6×6** matrix over the
environment's `COLORS` tuple *in that exact order*, symmetric, diagonal 1.0.

Two ways in, and you should do both:
- **Empirical:** when a model is wrong, which distractor does it pick? Build the
  confusion matrix over candidate attributes from your raw responses.
- **Embedding:** cosine similarity between sentence embeddings of the full
  referring expressions (`sentence-transformers`, `all-MiniLM-L6-v2` or
  `bge-small-en`).

Report the correlation between the two. If empirical errors track embedding
similarity, that is a nice result and justifies the model; if not, ship the
empirical matrix and say why. This replaces the hand-written `COLOR_SIM` in
`cogniwatt/env/tasks.py` — **send it to Jayden, do not edit that file yourself.**

---

## 7b. Deliverable D — RTX 3060 compute profile (the hardware half)

**Script:** `scripts/r2_profile.py` · **Output:** `data/r2_rtx3060_profile.json`

Run this **on the same inference passes** as §6 — do not schedule a separate
measurement campaign. Wrap each call in a timer and sample GPU power alongside.

### Protocol

1. **Idle baseline.** Sample 60 s with nothing running:
   ```bash
   nvidia-smi --query-gpu=power.draw,clocks.sm,temperature.gpu --format=csv -lms 100
   ```
   Take the median as `P_idle`.
2. **Per-call measurement.** Start the sampler, run one inference, stop.
   Energy above idle:
   `energy_mJ = Σ (P_sample − P_idle) × Δt × 1000`, with `Δt` in seconds.
3. **Repeat ≥ 50 times per tier**, discarding the first 5 (warm-up / weight
   paging). Report **median and IQR**, not mean — first-call outliers are large
   and not representative of steady-state behaviour.
4. Do this for `think_tiny` (0.5B Q8) and `think_big` (3B Q4_K_M).

### The `offload` tier is a special case

A cloud call spends **network radio** energy on the robot, not GPU energy — and
your desktop 3060 has no radio. So split it honestly:

- **You measure:** the real wall-clock round-trip **latency** distribution of the
  API calls (median, p95, and the failure/timeout rate). That is a genuine
  measurement and it feeds the env's offload latency directly.
- **Jayden models:** the radio energy, from published Wi-Fi transmit power
  figures. Label it as modelled, not measured, in the report.

### Output schema

```json
{
  "device": "rtx3060",
  "vram_gb": 4,
  "p_idle_w": 14.2,
  "sampler_hz": 10,
  "ops": {
    "think_tiny": {"latency_ms_median": 6.1, "latency_ms_iqr": [5.8, 6.9],
                   "energy_mJ_median": 398.0, "energy_mJ_iqr": [370.0, 431.0], "n": 50},
    "think_big":  {"latency_ms_median": 41.3, "latency_ms_iqr": [39.1, 45.0],
                   "energy_mJ_median": 4180.0, "energy_mJ_iqr": [3950.0, 4460.0], "n": 50},
    "offload":    {"latency_ms_median": 148.0, "latency_ms_p95": 402.0,
                   "failure_rate": 0.004, "energy_mJ_median": null, "n": 200}
  },
  "models": {"think_tiny": "qwen2.5-0.5b-instruct-q8_0.gguf",
             "think_big": "llama-3.2-3b-instruct-q4_k_m.gguf",
             "offload": "claude-haiku-4-5"},
  "notes": "offload energy is radio-side and modelled by R1, not measured here"
}
```

### Gotchas specific to this

- `nvidia-smi` power readings update at roughly **10 Hz**. A 6 ms `think_tiny`
  call is *shorter than one sample period* — you cannot measure it directly.
  **Fix:** run the call in a tight loop 200× and divide total energy by 200.
  Do the same for anything under ~200 ms. This is the mistake that produces
  nonsense numbers.
- Lock the clocks if you can (`nvidia-smi -lgc`), or at minimum record
  `clocks.sm` alongside power so boost-clock variation is visible.
- A laptop GPU behaves very differently on battery vs mains. **Measure on mains
  and say so.**
- Your card is 4 GB. The sim's discrete-GPU device profile (`rtx3060`) assumes a
  6 GB reference part, so in your `meta`/notes state your actual card and VRAM —
  the anchor is characterised by your *measured per-op numbers*, and the device
  it is named after is Jayden's problem to reconcile, not yours. Just be precise
  about what you measured on.
- Report ambient temperature and whether the fan curve was default.

---

## 7. Deliverable D — the frozen artefact

**File:** `data/dialogue_measured.npz` · **Script:** `scripts/r2_freeze.py`

This is the actual handoff back to R1. Exact schema:

| Key | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `tiers` | (T,) | str | must be a subset of `think_tiny`, `think_big`, `offload`, `ask` |
| `accuracy` | (T,) | float | top-1 grounding accuracy, pooled over k |
| `evidence_decay` | (T,) | float | γ per tier, in (0, 1] |
| `asr_wer` | scalar | float | measured word error rate |
| `k_values` | (Kn,) | int | e.g. `[2, 3, 4]` |
| `accuracy_by_k` | (T, Kn) | float | accuracy per tier per candidate count |
| `n_samples` | (T, Kn) | int | provenance for CIs |
| `meta_json` | scalar | str | JSON: model ids, quantisation, temperature, date |

First four keys are **required** — `load_precomputed` raises `KeyError` without
them rather than silently falling back, which is deliberate.

Verify:

```python
from cogniwatt.interfaces.dialogue import load_precomputed
o = load_precomputed("data/dialogue_measured.npz", strict=True)
print(o.accuracy("think_big", k=4), o.evidence_decay, o.meta)
```

Then confirm it drives the environment end to end:

```python
from cogniwatt.env.homegrid import HomeGridEdge, TRAIN_DEVICES
env = HomeGridEdge(devices=TRAIN_DEVICES, oracle=o, seed=0)
env.reset(seed=1)
```

---

## 8. Deliverable E — clarification questions

The `ask` action currently costs reward and returns an abstract observation; it
never generates language. For the demo and the qualitative results you need real
questions.

**Module:** `cogniwatt/dialogue/clarify.py`

Given the instruction and the candidate set, produce the question that best
splits the belief — *"Do you mean the one on the counter or the one in the
cupboard?"* — grounded strictly in objects that exist in the scene.

Evaluate on: **information gain** (does the answer actually disambiguate?),
**groundedness** (does it ever hallucinate an object? this should be 0%), and
**naturalness** (small human rating, n ≈ 50, you and two others).

A nice extra result if you have time: compare an entropy-maximising question
selector against just asking the SLM to write a question, and show the
information gain difference.

---

## 9. Deliverable F — failure taxonomy

Categorise every error your models make. Expect something like: *position bias*,
*attribute confusion* (red vs striped), *pragmatic failure* (missing the "already
away" inference), *format violation* (didn't return an index), *hedging*. Report
the distribution per tier.

This is where the real insight lives — "the 0.5B model fails at pragmatics, not
perception" is a far better report sentence than "the 0.5B model got 45%".

---

## 10. Lab-by-lab plan

| Lab | Milestone | Your individual output |
| --- | --- | --- |
| 2 | Scope + related work | Survey: LLM robot agents that ask questions (SayCan, Inner Monologue, DialFRED) — and why none of them price reasoning |
| 3 | Dataset v0 | 100-scenario pilot + **the accuracy-gradient check** (§3). Report the gradient before going further |
| 4 | Tiers running | All three tiers responding through one prompt template; first accuracy table with CIs |
| 5 | Full measurement | 600 scenarios × 3 tiers × 10 repeats; `r2_raw_responses.parquet` |
| 6 | **Evidence decay** | γ per tier + marginal-information curves. **Your headline figure** |
| 7 | ASR | WER-vs-SNR curve, referent recovery rate, operating point |
| 7b | **Hardware profile** | `r2_rtx3060_profile.json` — latency + energy per tier, offload latency distribution |
| 8 | Artefact + integration | `dialogue_measured.npz` + profile delivered; Jayden retrains on measured constants |
| 9 | Clarification + taxonomy | Question generator, groundedness eval, failure taxonomy |
| 10 | Final | Dialogue chapter, qualitative transcripts, demo video |

**Tier-2 AI2-THOR is now a stretch goal, not a commitment.** With two people it
is the first thing to cut. The full novelty claim lives in Tier 1 plus the
measured hardware model; the photoreal port is a demo bonus. Decide by Lab 8 and
do not let it eat the measurement work.

---

## 11. Gotchas that will cost you a day each

- **The accuracy gradient is everything.** Check it at Lab 3 with 100 scenarios.
  If all tiers land within 10 points of each other, redesign the scenarios before
  spending money on the full run. Do not discover this at Lab 6.
- **Never call a model inside `sample_evidence`.** See §1.1.
- **Shuffle candidate order per query** and record the permutation, or position
  bias silently inflates your accuracies.
- **Store raw responses.** Re-running 20k calls because you only kept aggregates
  is a genuinely bad afternoon.
- **Use the same prompt for every tier.** Any per-tier prompt tuning makes the
  accuracy comparison meaningless.
- **`temperature` returns 400 on Opus 5 / Sonnet 5.** Haiku 4.5 accepts it (§5).
- **Batch API results are unordered** — key by `custom_id`.
- **4 GB VRAM is the whole ballgame (§5.1).** Never hold two models resident at
  once — load, measure, free, load the next. On 4 GB this is mandatory, not a
  nicety: the 3B model needs ~2.4 GB fully-resident for a valid hardware anchor,
  and anything co-resident pushes it to partial CPU offload, which silently
  invalidates the `think_big` device measurement.
- **γ ≈ 1 for every tier** means your sampling temperature is effectively zero —
  the models are deterministic and you have measured nothing. Check early.

---

## 12. What to send back to Jayden

1. `data/dialogue_measured.npz` — the dialogue artefact
2. `data/r2_rtx3060_profile.json` — the hardware profile (§7b)
3. `data/r2_color_similarity.npy` — the 6×6 matrix (he wires it into `tasks.py`)
4. `docs/r2_measurements.md` — accuracy table with CIs, γ with bootstrap CIs,
   WER curve, latency/energy table with IQRs, failure taxonomy, and **every
   deviation from this plan**
5. Figures: marginal-information curve, WER vs SNR, per-tier failure
   distribution, energy-per-query by tier
6. A one-paragraph note on anything in `cogniwatt/env/` or `ved.py` that you
   needed changed but did not change yourself

When the artefact lands, R1 retrains on measured constants and the headline
result stops being "structurally realistic" and becomes "grounded in measurement".
That retrain is the moment the project becomes defensible — get the artefact to
him by Lab 8, not Lab 10.

---

## 13. First three commands

```bash
python3 tests/test_env.py
```

```bash
python3 -c "import inspect, cogniwatt.interfaces.dialogue as d; print(inspect.getsource(d))"
```

```bash
sed -n '/def test_dialogue_artefact_roundtrip/,/^@test/p' tests/test_env.py
```

Read the interface and that test before writing anything. They define, precisely
and without ambiguity, what you owe.
