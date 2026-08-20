# CogniWatt

**Learning the economics of cognition for edge robots.**

A household robot in simulation is given an ambiguous spoken instruction and a
finite hardware budget. A single RL policy decides not just *what to do*, but
*how much compute to spend thinking* and *when to interrupt the human* — under
an energy, latency and thermal model calibrated to real edge devices.

Course project — Edge AI & Robotics / Conversational AI. Simulation only; no
physical hardware.

---

## The idea in one table

Every decision step, the agent picks one action from a single space:

| Action | Pays in |
| --- | --- |
| `goto` / `pick` / `place` | battery, wall-clock (locomotion dominates: ~10 J per grid cell) |
| `think_tiny` — 0.5B INT8 on-device | little energy, capped at **0.55** confidence |
| `think_big` — 3B Q4 on-device | ~10× energy, thermal headroom, reaches **0.95** |
| `offload` — cloud model | radio energy, network latency, can fail |
| `ask` — clarifying question | the human's finite patience |

`reward = success − λ_E·energy − λ_T·latency − λ_Q·questions`

Nobody hand-codes the trade-off. It emerges: when the battery is low the agent
stops reasoning and starts asking; when the human has already been interrupted
it burns compute instead; on a device that cannot fit the 3B model it shifts to
cloud and dialogue.

## Novelty

Two literatures have never been joined:

- **LLM-driven robot agents that ask questions** (SayCan, Inner Monologue,
  DialFRED) treat reasoning as *free* — no compute cost, no energy, no device.
- **Edge inference optimisation** (model cascades, early-exit nets, RL
  offloading) optimises a fixed workload with no embodiment and no human.

CogniWatt prices **human interaction and compute allocation in the same
currency, inside one RL action space, against a calibrated hardware cost
model**, and shows that the resulting policy discovers a channel portfolio
rather than being told one.

It also **transfers zero-shot to an operating condition it has never seen** —
no network and no human available — matching a policy trained in that condition
from scratch (0.7225 ± 0.0061 vs 0.7113 ± 0.0027, n = 8 seeds), re-allocating
**39× more** evidence purchasing onto the on-device models at runtime, and
beating every fixed rule by **21 points** in a condition where those rules
collapse.

> **On the earlier version of this claim.** It previously read "transfers
> zero-shot to an edge *device* it has never seen, conditioned only on datasheet
> features". Both halves were audited and neither held: device-feature
> conditioning has no measurable effect (Δ = +0.13 pt held-out, Welch *t* = 0.42,
> n = 16 per arm), and no baseline — including device-blind ones — degrades on
> the device-based split, so it never tested hardware generalisation at all. The
> claim above is the measured replacement, on a split that can actually detect
> failure. See `docs/r1_transfer_audit.md`.

## Repository layout

```
cogniwatt/
  env/         homegrid.py   the POMDP            [R1]
               belief.py     Bayesian filter      [R1]
               tasks.py      floorplans + tasks   [R1]
  interfaces/  ved.py        device cost model    [R1 defines / R1+R2 measure]
               dialogue.py   evidence channels    [R1 defines / R2 implements]
  agents/      ppo.py        recurrent PPO        [R1]
               baselines.py  9 scripted agents    [R1]
  analysis/    evaluate.py, pareto.py             [R1]
scripts/       train.py, run_baselines.py, evaluate_policy.py, make_figures.py
tests/         test_env.py   12 tests, no pytest dependency
docs/          lab1_mdp_formalization.md      problem formalisation      [Lab 1]
               r1_hardware_calibration.md     M2 anchor, roofline, thermal
               r1_stability.md                training stability + annealing
               r1_cloud_price.md              deriving lambda_cloud
               r1_lab_plan.md                 per-lab deliverable record
               r1_results.md                  generated results tables
               r1_report.html                 generated full report
               R2_HANDOFF.md                  Adit's workstream brief
```

## Reproducing the results

```bash
python3 scripts/r1_sweep.py --arms base bigbatch gru ent03 ent01 entanneal --jobs 4
```

```bash
python3 scripts/r1_stability.py && python3 scripts/r1_cloud_price.py
```

```bash
python3 scripts/evaluate_policy.py --tags entanneal_s0 -n 1000
```

```bash
python3 scripts/make_figures.py && python3 scripts/make_report.py && python3 scripts/make_html_report.py
```

Both reports are **generated**, never hand-edited — `docs/r1_report.html` once
drifted months out of date from the code it described, which is how the
reproducibility problem in `docs/r1_stability.md` §6.3 went unnoticed.

## Running it

```bash
python3 tests/test_env.py
```

```bash
python3 scripts/run_baselines.py 600
```

```bash
python3 scripts/train.py --steps 2000000 --no-recurrent --tag main
```

```bash
python3 scripts/evaluate_policy.py --tags main -n 1000
```

Dependencies: `numpy`, `torch`, `pandas`, `matplotlib`. Deliberately **no**
gymnasium or stable-baselines3 — the env is pure NumPy so it runs bit-identically
on arm64 macOS and x86_64+CUDA, and PPO is implemented from scratch in
`agents/ppo.py`.

## Team interfaces

The two files in `cogniwatt/interfaces/` are frozen contracts. Both currently
ship R1 placeholder numbers so the RL workstream is never blocked, and both are
swapped for measured values without touching `cogniwatt/env/`:

| File | Owner | Replaces placeholder with |
| --- | --- | --- |
| `dialogue.py` | **R2 — Adit** | per-tier grounding accuracy of real quantised SLMs, evidence-decay constants, Whisper WER, embedding-based confusability |
| `ved.py` (compute ops) | **R2 — Adit** | per-inference latency and energy on the RTX 3060, captured on the same passes that measure accuracy |
| `ved.py` (everything else) | **R1 — Jayden** | M2 Air anchor via `powermetrics`, thermal fit on the fanless chassis, actuation and network models, roofline extrapolation to Jetson / Pi / Coral |

Two-person team: the split rule is **whoever runs a model profiles it**. Adit is
already loading the SLMs to measure grounding accuracy, so capturing latency and
power on those same runs costs him almost nothing and yields the RTX 3060 anchor.

**Performance contract:** `sample_evidence` runs millions of times inside the
PPO loop and must never call a language model. R2 delivers a *precomputed*
artefact — SLM outputs enumerated offline once and frozen to disk. Live model
calls happen only in the Tier-2 photoreal demo and final evaluation.

## Status

- [x] Tier-1 environment, belief filter, task generator
- [x] Frozen VED and dialogue interfaces with working placeholders
- [x] 9 scripted baselines incl. a strong information-value heuristic
- [x] Recurrent PPO from scratch, action masking, potential-based shaping
- [x] Test suite 13/13
- [x] R1 M2 Air anchor, thermal fit, roofline extrapolation
- [x] **Training stability study** — the bimodal-convergence limitation is
      resolved and its cause identified (`docs/r1_stability.md`)
- [x] **Entropy annealing** (`--ent-coef-final`) — converges on every seed *and*
      produces a policy efficient enough to beat the heuristic on return
- [ ] Multi-seed ablations (they no longer collapse, so the old single-seed
      numbers no longer establish anything)
- [ ] R2 measured dialogue constants + RTX 3060 compute profile
- [ ] Tier-2 AI2-THOR port (stretch goal, RTX 3060 only — drop if time is short)

### Current headline (8 seeds, current reward, `entanneal`)

| split | PPO | `greedy_value` |
| --- | --- | --- |
| train | 0.909 ± 0.004 success, −0.30 ± 0.07 return | 0.898, −0.58 |
| held-out | 0.898 ± 0.003 success, −0.47 ± 0.06 return | 0.892, −0.64 |

The learned policy wins on both metrics at **equal information spend** (2.19
queries/episode vs 2.17). Margins are ~1 point of success — smaller than the
previously reported ~6, which was measured before `lambda_cloud` priced cloud
calls. See `docs/r1_stability.md` §6.3.

> **Note on the legacy results.** `results/runs_legacy/` and
> `results/eval_legacy/` hold the pre-recalibration run and evaluation. They
> were produced under a 40-dim observation, a `coral_micro` held-out device and
> a reward without `lambda_cloud`, and are **not comparable** to anything
> current. `scripts/evaluate_policy.py` now refuses to evaluate across a reward
> change.
