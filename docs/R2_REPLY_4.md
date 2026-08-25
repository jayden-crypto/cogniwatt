# R2 reply 4 — the offload tier, for free, and better (Jayden → Adit)

**Paste this whole file to your Claude as the opening prompt.** Reads cold.
Supersedes §5 and §6 of `docs/R2_REPLY_3.md`.

**The $4 is not happening — this is a course project and we are not expensing an
API.** That is fine. The paid measurement was never the right instrument anyway,
and the free version is a cleaner experiment. Nothing about the project is
blocked. Details below, then a concrete plan.

---

## 1. The `offload` tier does not have to be a commercial API

Re-read what the environment actually consumes. `offload` is a tier with:

* an accuracy the belief filter samples from,
* an evidence-decay constant,
* a cost profile of **network latency + radio energy**.

Nowhere does it require the model to be Anthropic's, or commercial, or even
remote. The definition that matters for an Edge AI project is:

> **a model too large to reside on the edge device, reached over a network.**

A 7B or 14B GGUF is exactly that. A 4 GB card cannot hold one; a home server or
an edge gateway can. That is a *physical* justification for the tier, where
"Claude Haiku" was a commercial one — and physical is what this project is
about.

### 1.1 We lose no measurement we could have defended

You already made this argument yourself, in `R1_HANDBACK.md` §7 correction 2:
the Batch API measures Anthropic's queue, not a robot's network. The live
backend has the same problem one level down — it measures **your laptop's ISP
and Wi-Fi**, which is not a robot's radio either. R1 has always modelled offload
latency and radio energy from published figures and labelled them *modelled*, in
`docs/r1_hardware_calibration.md`. That does not change.

So the paid run would have bought us one number — cloud grounding accuracy — and
a latency figure neither of us would have defended in a viva. We can get the
first for free.

### 1.2 It is a better controlled experiment

Current tiers mix two stacks: local GGUF via llama.cpp, and a commercial API
with its own tokenizer, sampler and safety layer. Any accuracy difference
confounds *model scale* with *serving stack*.

Replacing the top tier with a larger GGUF on the same llama.cpp path gives:

| tier | model | resident size | fits on 4 GB edge device? |
| --- | --- | --- | --- |
| `think_tiny` | Qwen2.5-0.5B-Instruct Q8_0 | ~0.6 GB | yes |
| `think_big` | Llama-3.2-3B-Instruct Q4_K_M | ~2.3 GB | yes, barely |
| `offload` | **Qwen2.5-7B or 14B-Instruct Q4_K_M** | ~4.7 / ~9 GB | **no — must be offloaded** |

Same stack, same prompt template, same parser, same permutations, same
temperature. **A clean scale ladder**, which answers a better question than we
were asking: *how large does a model have to be before it can ground a referring
expression well enough to act on?* That is a genuinely interesting Edge AI
result and it subsumes the Q8-vs-Q4 question from `R2_REPLY_3.md` §6.

---

## 2. The number to beat

R1 computed what the offload tier must clear for the tier to be *usable at all*
— i.e. to reach the ~0.85 posterior an irreversible `place()` needs, at k=3:

| single-query accuracy | γ = 0.35 | γ = 0.60 |
| --- | --- | --- |
| 0.70 | 0.613 ✗ | 0.723 ✗ |
| 0.75 | 0.680 ✗ | 0.796 ✗ |
| **0.80** | 0.751 ✗ | **0.862 ✓** |
| **0.85** | 0.816 ✗ | **0.916 ✓** |
| **0.88** | **0.859 ✓** | 0.945 ✓ |

**So the question is simply: does the large model clear ~0.80–0.85 single-query
accuracy at k=3?** Above that the current framing holds. Below it, we get a
different and arguably more striking result — *no model available to a domestic
robot can ground these expressions confidently, so dialogue is not a fallback,
it is the primary channel*. Either outcome is publishable. That is why this
measurement is worth doing and why neither answer is a failure.

---

## 3. Plan A — local large model, zero accounts, zero cost

Your own §5.1 argument applies: **accuracy is never blocked by VRAM.** Set
`n_gpu_layers` to whatever fits and let the rest run on CPU. Slower, identical
outputs. You are not measuring energy on this tier — R1 models radio energy —
so speed genuinely does not matter here.

**Model:** `Qwen2.5-7B-Instruct-Q4_K_M` (~4.7 GB) as first choice. If you have
16 GB system RAM, `Qwen2.5-14B-Instruct-Q4_K_M` (~9 GB) is the stronger test and
still runs. Either is a legitimate `offload` target; record which.

**Query budget** — you do not need the full N=10 × 660 protocol:

| purpose | scope | queries |
| --- | --- | --- |
| accuracy + accuracy_by_k | all 660 scenarios, N=1 | 660 |
| evidence decay γ (needs repeats) | 200-scenario stratified subset, N=10 | 2 000 |
| **total** | | **~2 660** |

At ~3 s/query on a 7B with partial offload that is **~2 hours unattended**; a 14B
mostly on CPU is ~6 hours. Overnight either way. Nothing to babysit, no rate
limits, no key, and it re-runs deterministically forever — which the API version
never would.

Subsetting for γ is statistically fine: γ is a within-scenario correlation, so it
needs *repeats per scenario*, not *many scenarios*. 200 scenarios × 10 repeats
gives 2 000 rows and a cluster bootstrap over ~58 cores, same estimator you
already validated. Stratify the subset across k and family so it stays balanced.

---

## 4. Plan B — free API tiers, if you would rather have a genuinely large model

If you want a 70B-class model rather than 7B/14B, several providers give real
free tiers with **no card required**. Any of these is legitimate for a course
project:

* **Groq** — Llama-3.3-70B-Versatile, generous free RPM/RPD, very fast
* **Google AI Studio** — Gemini 2.0/2.5 Flash, ~1 500 requests/day free
* **Cerebras**, **OpenRouter** (`:free` models), **Mistral** — all have free tiers

2 660 queries fits inside a day on any of them; Groq would finish in minutes.
Cost: an email address.

**Caveat, and it is why this is Plan B:** it reintroduces the stack confound
from §1.2 — different tokenizer, sampler and serving path — and the model can be
deprecated or rate-limited out from under us, so the result stops being
reproducible. Plan A is worth more to the report even though the model is
smaller. If you do Plan B, do it *in addition to* Plan A, not instead.

---

## 5. What changes in the code

Almost nothing, which is the point.

* `cogniwatt/dialogue/backends.py` — the local backend already exists; point a
  third tier at a larger GGUF. The Anthropic backends stay in the tree, unused
  and documented as "written, never run, no budget approved".
* `data/dialogue_measured.npz` — `tiers_measured` becomes all three;
  `meta_json` records the offload model id and quant, and states plainly that
  the offload tier is *a locally-hosted large model standing in for a network
  service*, with the reasoning from §1.
* `interfaces/ved.py` — untouched. Offload cost stays modelled radio energy +
  network latency, exactly as now.
* Nothing in `cogniwatt/env/`, `agents/`, or `analysis/`.

**In the report** the tier is described as *"a model that cannot reside on the
device, reached over a network; measured on a locally-hosted instance because
the accuracy of a model does not depend on which machine serves it."* That is
true, defensible, and stronger than a footnote about API credits.

---

## 6. Revised priorities

| # | Item | Cost | Why |
| --- | --- | --- | --- |
| **1** | `offload` accuracy + γ via **local 7B/14B** (§3) | free, one overnight run | decides whether `think_big` being dead is a finding or an artefact — this is the project |
| 2 | Q8_0 rerun of `think_big` (`R2_REPLY_3.md` §6) | free, ~30 min | separates "3B-class is insufficient" from "Q4 destroys grounding" — and §3 gives you the scale ladder to put it on |
| 3 | per-tier latency/energy on mains | free, 15 min | report cost table; still worth doing when the laptop is plugged in |
| 4 | human audit of the cores | time only | ground truth; not blocking |
| ~~5~~ | ~~paid API~~ | — | **cancelled** |
| ~~6~~ | ~~real-speech ASR~~ | — | cancelled, `R2_REPLY_3.md` §3 |

Items 1 and 2 together give a four-point scale ladder — 0.5B Q8, 3B Q4, 3B Q8,
7B/14B Q4 — on one stack, with one prompt and one parser. That is a better
Deliverable B than the original handoff asked for, and it costs nothing but a
night of compute.

---

## 7. To be explicit about scope

Nothing here is a retreat. The paid cloud tier was one line in a table; the
measurement it was supposed to produce is fully recoverable for free, on a
better-controlled experiment, with the only genuine loss being an API latency
figure that neither of us would have defended. The project is not blocked and
does not need re-scoping.
