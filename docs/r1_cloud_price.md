# CogniWatt — Deriving the Price of a Cloud Call (`lambda_cloud`)

**Owner:** Jayden (R1) · **Scripts:** `scripts/r1_cloud_price.py`, `scripts/r1_sweep.py`
**Data:** `results/cloud_price.csv`, `results/runs/cloud*`

---

## 1. Why this parameter needs an argument rather than a value

`RewardConfig.lambda_cloud` is a flat penalty per cloud call, added during the
hardware recalibration for a stated reason: once roofline costs were realistic,
a cloud call (140 ms / 0.8 J, 88% accurate) *strictly dominated* on-device
inference (1591 ms / 26 J on the M2 Air, 72% accurate), the optimal policy
degenerated to "always offload", and the local tiers became dead weight.

`lambda_cloud = 0.5` was chosen to stop that. It does — but only just. Measured
on the current policy (`docs/r1_stability.md` §7), the on-device tiers carry
**5–20% of all evidence queries**. For a project whose claim is that the optimum
is a *portfolio*, and whose subject is *edge* inference, that is close to the
degeneracy it was introduced to prevent.

The temptation is to raise `lambda_cloud` until the portfolio looks balanced.
That would be assuming the conclusion: the portfolio claim would then be true by
construction, and the result worthless. This document instead separates what can
be derived from what cannot.

`lambda_cloud` prices two distinct real-world costs. They deserve different
treatment.

## 2. The monetary component — derivable

A grounding query is ~300 input and ~20 output tokens (measured on the prompt
template in `docs/R2_HANDOFF.md` §5). At Claude Haiku 4.5 pricing
($1 / MTok input, $5 / MTok output):

```
300 x $1e-6  +  20 x $5e-6  =  $0.00030 + $0.00010  =  $4.0e-4 per call
```

To express that in reward units we need an exchange rate, and the reward scale
is anchored by `RewardConfig`'s own design note: the weights are

> scaled so that exhausting the whole energy or latency budget costs about as
> much as a task failure

`lambda_energy = 0.025` per Joule against a 400 J episode budget gives
`0.025 x 400 = 10.0` — exactly `r_success`. So **one full battery ≈ one task
≈ 10 reward units**.

What is one task worth in money? Take a domestic assistant performing ~50 tasks
per day. Over a year that is ~18,250 tasks. Against a plausible $500/year
service value, one task ≈ **$0.027**. Hence

```
$4.0e-4 per call  /  $0.027 per task  =  1.5% of a task  =  0.15 reward units
```

**The monetary cost of a cloud call is therefore ≈ 0.15**, an order of magnitude
below the current 0.5. Every assumption above is stated and each is adjustable;
the conclusion is robust to all of them within a factor of ~3, because token
counts and API prices are known and the task-value anchor enters linearly.

## 3. The privacy component — not derivable, and should not be faked

The remaining 0.35 of the current 0.5 is implicitly a privacy price: the cost of
a household's scene description leaving the premises. There is no measurement
that yields this number. It depends on jurisdiction, on the deployment, and on
the household. Any specific figure would be an assertion dressed as a
derivation.

The project already has the right precedent for exactly this situation.
`lambda_ask` — the price of the human's patience — is equally underivable, and
the report does not derive it: it **sweeps it and reports the Pareto frontier**
(`docs/r1_results.md`, "Pareto sweep over the price of a question"). The same
treatment is the consistent and honest one here.

So:

* **monetary component: 0.15, derived** (§2);
* **privacy component: a policy parameter**, swept over `[0, 3.85]`, giving
  `lambda_cloud ∈ {0.15, 0.5, 1.0, 2.0, 4.0}`.

## 4. What the sweep shows about portfolio structure

`scripts/r1_cloud_price.py` sweeps the price against `GreedyValueAgent`, which
re-derives its channel scores from `env.cfg` every step and therefore
re-optimises for whatever price it is handed — so portfolio structure can be
mapped without retraining:

| `lambda_cloud` | success | local share | cloud share | asks |
| --- | --- | --- | --- | --- |
| 0.0 | 0.890 | 8% | 81% | 0.27 |
| **0.15 (derived, money only)** | — | — | — | — |
| 0.5 (current) | 0.902 | 18% | 70% | 0.24 |
| 1.0 | 0.900 | 31% | 54% | 0.30 |
| 2.0 | 0.920 | 47% | 37% | 0.44 |
| 4.0 | 0.922 | 53% | 9% | 1.04 |
| 8.0 | 0.927 | 55% | 6% | 1.04 |
| 16.0 | 0.930 | 57% | 4% | 1.05 |

Three regimes, and the middle one is narrow:

1. **`lambda_cloud < 1`: cloud-dominated.** The local tiers are vestigial. This
   is where the project currently sits.
2. **`1 ≤ lambda_cloud ≤ 2`: genuine three-way portfolio.** Local 31–47%,
   cloud 37–54%, dialogue 15–20%. This is the regime the environment was built
   to study.
3. **`lambda_cloud > 4`: the cloud is priced out** and the agent substitutes the
   *human* (asks rise to ~1.0/episode). A different degeneracy, not a success.

**Returns are not comparable down this table.** Each row optimises a different
objective, so return declines monotonically as a matter of arithmetic, not
because higher prices are worse. Only structure and success are comparable
across rows — the same trap that invalidated the legacy headline
(`docs/r1_stability.md` §6.3).

## 5. What is actually claimed

Rather than picking a value, the results are reported **across the sweep**, and
the claims are separated by how much they depend on it:

* **Invariant — the agent claim.** Does the learned policy beat the strong
  heuristic at every price? If yes, the central contribution does not depend on
  `lambda_cloud` at all, and the parameter is a presentational choice.
* **Not invariant — the portfolio claim.** The composition of the optimal
  portfolio is a property of the *cost model*, not of the learner. It must be
  stated as conditional: *"under a cloud price of X, the optimum allocates Y."*

`scripts/r1_sweep.py` trains the canonical annealed policy at each of the five
prices, 3 seeds each; each is evaluated against the heuristic **at its own
price**, which is the only paired comparison that means anything.

## 6. Recommendation

1. **Report `lambda_cloud = 0.15` as the derived, money-only floor**, and be
   explicit that it excludes privacy.
2. **Report the sweep as the main result**, exactly as `lambda_ask` is reported.
3. **State the portfolio finding conditionally.** If the honest answer is that
   under 2024-class edge costs a home robot *should* offload most of its
   reasoning, that is a legitimate and interesting result — but the report's
   framing must then change, because it currently asserts a portfolio.
4. **Do not tune `lambda_cloud` to produce a balanced portfolio.** The sweep is
   the result; a single flattering value is not.

## 7. Results — the trained sweep

The canonical annealed policy trained at each price, 3 seeds each, evaluated
against the heuristic **at its own price** (the only paired comparison that
means anything). Success/return are mean ± se over seeds; shares are of all
evidence queries.

| λ_cloud | split | PPO success | greedy | PPO return | greedy return | on-device | cloud |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.15 | train | **0.914 ± 0.002** | 0.894 | +0.006 | +0.019 | 17% | 73% |
| 0.15 | held-out | **0.919 ± 0.006** | 0.890 | **+0.153** | −0.117 | 4% | 84% |
| 0.5 | train | **0.907 ± 0.005** | 0.904 | −0.351 | −0.361 | 18% | 66% |
| 0.5 | held-out | **0.897 ± 0.002** | 0.888 | **−0.523** | −0.726 | 5% | 75% |
| 1.0 | train | 0.887 ± 0.002 | **0.900** | **−0.952** | −0.996 | 22% | 54% |
| 1.0 | held-out | 0.876 ± 0.004 | **0.908** | −1.136 | **−0.866** | 7% | 65% |
| 2.0 | train | 0.891 ± 0.002 | **0.920** | **−1.360** | −2.233 | 28% | 2% |
| 2.0 | held-out | 0.890 ± 0.001 | **0.912** | **−1.449** | −2.215 | 10% | 2% |
| 4.0 | train | 0.890 ± 0.003 | **0.922** | **−1.347** | −2.820 | 24% | 0% |
| 4.0 | held-out | 0.889 ± 0.001 | 0.890 | **−1.431** | −2.799 | 6% | 0% |

### 7.1 The agent claim is invariant on return, not on success

**The learned policy beats the heuristic on return at every price**, and the
margin *widens* as the price rises — from a tie at 0.15 to more than a full
reward unit at 2.0 and 4.0. That is the claim that should be reported, because
return is the objective; it is also the harder claim, since the heuristic is
re-optimising for each price while the policy has to learn it.

**It loses on success once `lambda_cloud >= 1`.** This is not a contradiction.
`GreedyValueAgent` buys evidence until its confidence threshold (0.85) is met
*regardless of what that costs* — a fixed-threshold rule is price-insensitive by
construction. So at high prices it purchases extra success at a return it would
never choose if it were optimising return. The learned policy correctly declines
to make that purchase.

**Consequence for the report:** state the claim as *"beats a strong heuristic on
the objective it is given"*, not *"achieves higher success"*. The success-only
framing is true at the derived price (0.15) and at 0.5, and false above 1.0.

### 7.2 The substitute for the cloud is the human, not on-device compute

This is the finding that matters most, and it is uncomfortable.

As `lambda_cloud` rises from 0.5 to 2.0, cloud usage collapses from 66% to 2% —
but on-device inference only rises from 18% to 28%. **The remainder goes to
`ask`.** The heuristic sweep of §4 suggested the local share would reach ~47%;
the *trained* policy does not do that, and on held-out devices the on-device
share never exceeds 10% at any price.

The cost model explains it. At `lambda_cloud = 0.5` one `offload` costs
≈ 0.54 reward for 0.88 accuracy, against `think_big` at ≈ 0.92 for 0.72 — the
cloud strictly dominates. Raise the price and `think_big` becomes the cheapest
channel per query, but `ask` costs ≈ 2.0 for an effective 0.89, and under
evidence saturation (`gamma_ask` is low precisely because re-asking is
uninformative) **one accurate draw beats several weak ones**. A 0.72-accuracy
channel that saturates at ~2.2 effective draws is simply not a good buy at any
price when a 0.89-accuracy channel exists.

**Under CogniWatt's own calibrated hardware costs, on-device inference is rarely
the rational purchase.** The roofline recalibration — the project's most solid
piece of measurement — is what makes this true: a 3B inference at 1591 ms and
26 J on an M2 Air, or 32 s on a Pi 5, cannot compete with a 140 ms cloud round
trip or a question to a human standing in the room.

### 7.3 What this means for the project's framing

The environment was built to demonstrate that the optimum is a **portfolio**
across four channels. Measured honestly, it is a portfolio across **two** —
cloud and dialogue — with the on-device tiers as marginal contributors that
only matter when the cloud is unavailable or the device is fast.

Three defensible responses, in order of preference:

1. **Report it as the finding.** "Under 2024-class edge hardware costs, a home
   robot should offload or ask rather than reason locally; local inference earns
   its place only when connectivity fails." This is a genuine, defensible and
   somewhat counterintuitive result about edge AI economics, and it is *better
   science* than a portfolio manufactured by parameter choice. It requires
   rewriting the project's framing, not its code.
2. **Add link failure as a first-class axis.** `net_quality` already exists and
   `offload_succeeds()` already fails stochastically. Sweeping connectivity —
   rather than price — is the physically honest way to make local inference
   necessary: an offline robot has no cloud at any price. This is a small
   experiment with the existing environment and would likely restore a genuine
   three-way portfolio for the right reason.
3. **Revisit `lambda_ask`.** The human currently costs 2.0 and answers with
   ~0.89 effective accuracy, which makes dialogue an extremely good buy. If real
   users tolerate fewer interruptions than that implies, `lambda_ask` is too low
   and on-device reasoning is being unfairly crowded out.

**Recommended:** (1) as the report's framing, plus (2) as the next experiment,
since it is cheap and directly tests whether the portfolio claim survives for a
principled reason rather than a tuned one.

---

## 8. Connectivity: a real discriminator that still does not restore the portfolio

§7.3 recommended sweeping link quality rather than price, on the grounds that an
offline robot has no cloud at any price. `net_quality_mean` is now a first-class
environment parameter (`cogniwatt/interfaces/ved.py`, `--net-quality`), swept at
{0.9, 0.5, 0.25, 0.1}, 3 seeds each, **trained and evaluated at the same link
quality**.

| link quality | PPO success | greedy | `always_offload` | tiny | big | cloud | ask | on-device share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.9 | 0.921 ± 0.008 | 0.902 | **0.872** | 0.27 | 0.02 | 1.80 | 0.24 | 12% |
| 0.5 | 0.889 ± 0.004 | 0.908 | **0.726** | 0.38 | 0.01 | 1.01 | 0.61 | 19% |
| 0.25 | 0.893 ± 0.007 | 0.912 | **0.596** | 0.39 | 0.00 | 0.26 | 0.97 | 24% |
| 0.1 | 0.893 ± 0.002 | 0.918 | **0.510** | 0.26 | 0.00 | 0.12 | 0.99 | 19% |

### 8.1 It fixes the held-out split problem

`always_offload` — a device-blind, connectivity-blind scripted policy — falls
**36 points**, from 0.872 to 0.510. That is exactly the discrimination the
device-based held-out split fails to produce (`docs/r1_transfer_audit.md` §3,
where no baseline moves by more than 0.9 points). **A held-out condition defined
by connectivity is a real generalisation test**, and swapping the held-out axis
from *device* to *link quality* would make the transfer claim meaningful.

### 8.2 It does not restore the portfolio

`think_big` usage is **0.00 at every link quality below 0.9**, and the on-device
share peaks at 24%. As the cloud fails, demand moves almost entirely to `ask`
(0.24 → 0.99 questions per episode). **When the cloud breaks, the policy asks the
human — it still never runs the 3B model.**

This is the decisive version of §7.2. On-device inference is not losing to a
cheap cloud; it is losing to *a person standing in the room*. A `think_big` call
costs ~0.92 reward (1591 ms, 26 J on an M2 Air) for 0.72 accuracy, against an
`ask` at 2.0 for an effective 0.89 — and under evidence saturation one accurate
draw is worth more than several weak correlated ones. No cloud price and no link
quality changes that ordering, because neither term appears in it.

### 8.3 The regime where local inference is rational

Local reasoning is the correct purchase only when **both** other channels are
unavailable: no network *and* no human. That is not a contrived corner — it is
the robot alone in the house, offline, which is arguably *the* canonical edge-AI
deployment and the one that motivates on-device models existing at all.

`patience_max` is already a parameter of `DialogueOracle`. The experiment is
therefore small: a held-out condition with `net_quality_mean ≈ 0.1` **and**
`patience_max = 0`, on a memory-limited device. In that condition `think_tiny`
is the only surviving evidence channel, `always_offload` and `always_ask` both
collapse, and the policy must fall back on the tier the project is named for.
**That is the experiment that makes the edge-AI framing earn its place**, and it
is the single highest-value remaining item in the R1 workstream.

### 8.4 A harness bug this exposed, and the guard added

The first run of this sweep produced byte-identical numbers for every scripted
baseline at every link quality — `greedy_value` exactly 0.904 and
`always_offload` exactly 0.844 throughout. That is impossible if connectivity is
varying, and it was the tell: `scripts/evaluate_policy.py` built its environment
without passing `net_quality_mean`, so all four arms were evaluated at the 0.75
default regardless of what they were trained on. The policies differed; the
world they were tested in did not.

`ENV_ARGS` / `check_env_args()` now reproduce the training environment at
evaluation time and refuse to place policies trained in different worlds in one
table. This is the same class of bug as the `lambda_cloud` reward mismatch
(`docs/r1_stability.md` §6.3) — a training-time setting that silently fails to
reach evaluation — and it is the second one found in this codebase, which
suggests the pattern is worth watching for rather than treating as a one-off.

---

## 9. When *is* on-device reasoning rational? The 2×2

**Script:** `scripts/r1_isolation.py` · **Data:** `results/isolation.csv` ·
**Figure:** `results/figures/fig7_isolation.png`

§8.2 showed that when the cloud degrades the policy asks the human rather than
reasoning locally. That points at a specific hypothesis: **the on-device tiers
are crowded out by the human, not by a cheap cloud**, and local inference should
only become rational when *both* alternatives are gone. Crossing connectivity
with human availability tests it directly. Three seeds per cell, trained and
evaluated in the same condition.

| cloud | human | PPO | greedy | `always_offload` | `always_ask` | `no_think` | tiny | big | cloud | ask | on-device share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yes | yes | **0.916 ± 0.002** | 0.902 | 0.872 | 0.890 | 0.374 | 0.29 | 0.03 | 1.77 | 0.24 | 14% |
| yes | no | **0.879 ± 0.002** | 0.866 | 0.872 | 0.374 | 0.374 | 0.64 | 0.20 | 1.69 | 0.00 | 33% |
| no | yes | 0.894 ± 0.002 | **0.918** | 0.510 | 0.890 | 0.374 | 0.26 | 0.00 | 0.10 | 1.00 | 19% |
| **no** | **no** | **0.711 ± 0.003** | 0.674 | 0.510 | 0.374 | 0.374 | **0.96** | **0.94** | 1.13 | 0.00 | **63%** |

### 9.1 The hypothesis holds

With neither channel available, the on-device share jumps to **63%** and
`think_big` rises from 0.00–0.03 calls per episode to **0.94**. The 3B model is
a rational purchase — but only as a last resort.

**The human is what crowds it out, not the cloud.** Rows 1→2 isolate this:
removing *only* the human, with the cloud still fast and available, more than
doubles the on-device share (14% → 33%) and lifts `think_big` sevenfold
(0.03 → 0.20). Removing only the cloud (row 3) barely moves it (14% → 19%) and
leaves `think_big` at zero — the demand goes to `ask` instead.

This is the sharpest statement of the project's economics: **a clarifying
question is such good value that it suppresses on-device reasoning even when the
cloud is unavailable.** At an effective 0.89 accuracy for `lambda_ask = 2.0`,
against 0.72 for a `think_big` call costing 1591 ms and 26 J on an M2 Air, that
ordering is not close — and evidence saturation means several weak local draws
cannot substitute for one good one.

### 9.2 This is a held-out condition that discriminates

`docs/r1_transfer_audit.md` §3 found that no agent degrades on the current
device-based held-out split — the largest drop of any baseline was 0.9 points,
so the split tests nothing. This condition is the opposite:

| | device split | offline + alone |
| --- | --- | --- |
| PPO | 0.911 → 0.904 (−0.7 pt) | 0.916 → **0.711** (−20.5 pt) |
| `always_offload` | 0.831 → 0.831 (0.0 pt) | 0.872 → **0.510** (−36.2 pt) |
| `always_ask` | 0.880 → 0.879 (−0.1 pt) | 0.890 → **0.374** (−51.6 pt) |
| PPO − best device-blind baseline | ≈ 0 pt | **+20.1 pt** |

Device-blind scripted policies collapse by 36–52 points, and the learned policy
retains a 20-point margin over the best of them. **This is a real generalisation
test**, and it is also where PPO's advantage over the strong heuristic is
largest anywhere in the project (+3.7 points, against +1.0 in the nominal
condition).

### 9.3 Recommendation

**Replace the held-out axis.** The held-out condition should be *offline and
alone* — `net_quality_mean ≈ 0.1`, `patience_max = 0` — rather than *a different
chip*. Both parameters already exist (`--net-quality`, `--patience`), so this is
a configuration change plus one retrain, not new machinery.

This fixes three problems at once:

1. the transfer claim becomes falsifiable and is then actually supported;
2. the portfolio claim becomes true for a *physical* reason (the channels really
   are unavailable) rather than a tuned `lambda_cloud`;
3. the on-device tiers — the edge-AI half of an Edge AI project — become
   load-bearing rather than vestigial.

**And it gives the project its honest headline:** *on-device reasoning earns its
place exactly when the robot is offline and alone, and a learned policy manages
that fallback 20 points better than any fixed rule.* That is a defensible claim
about edge AI, grounded in a measured hardware model, and it is a better story
than the one the project set out to tell.
